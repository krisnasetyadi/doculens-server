"""Bounded cache for loaded search indexes (MS-657).

The processor used to keep every FAISS index it ever loaded in a plain dict,
so memory only grew. IndexCache keeps the dict interface the callers already
use (in, [], get, pop, del, clear) but holds at most `max_bytes` of
estimated index memory, dropping the least recently used entries first. A
dropped index simply reloads from disk (or Supabase Storage) on its next
query.
"""

import logging
import threading
from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Chunk text is held twice more after loading: once in the docstore, and
# again as tokens in the BM25 index built from it (see get_bm25_index).
TEXT_COPIES = 3


def estimate_index_bytes(vector_store: Any) -> int:
    """Approximate memory of a loaded FAISS vector store: its float32 vectors
    plus its chunk text (and the BM25 copy of it). An estimate, not a
    measurement — good enough to keep the cache within a budget."""
    try:
        index = vector_store.index
        vectors = int(index.ntotal) * int(index.d) * 4
    except Exception:
        vectors = 0
    try:
        text = sum(len(doc.page_content) for doc in vector_store.docstore._dict.values())
    except Exception:
        text = 0
    return vectors + text * TEXT_COPIES


class IndexCache(MutableMapping):
    """A least-recently-used mapping with a memory budget.

    Reading an entry marks it as recently used. Storing one evicts the least
    recently used entries until the estimated total fits `max_bytes`; the
    entry just stored is never evicted, so one index larger than the whole
    budget still loads (and is the first to go when another arrives).
    `on_evict(key)` runs for each evicted key, so state derived from an
    index (its BM25 index) can be dropped with it.
    """

    def __init__(
        self,
        max_bytes: int,
        size_of: Callable[[Any], int] = estimate_index_bytes,
        on_evict: Optional[Callable[[str], None]] = None,
    ):
        self.max_bytes = max_bytes
        self._size_of = size_of
        self._on_evict = on_evict
        self._entries: "OrderedDict[str, tuple[Any, int]]" = OrderedDict()
        self._lock = threading.RLock()

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return sum(size for _, size in self._entries.values())

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            value, _ = self._entries[key]
            self._entries.move_to_end(key)
            return value

    def __setitem__(self, key: str, value: Any) -> None:
        size = self._size_of(value)
        with self._lock:
            self._entries[key] = (value, size)
            self._entries.move_to_end(key)
            total = self.total_bytes
            while total > self.max_bytes and len(self._entries) > 1:
                old_key, (_, old_size) = next(iter(self._entries.items()))
                del self._entries[old_key]
                total -= old_size
                logger.info("Index cache over budget: dropped %s (%.1f MB)", old_key, old_size / 1_048_576)
                if self._on_evict:
                    self._on_evict(old_key)

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._entries[key]

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._entries  # a membership check is not a use

    def __iter__(self):
        with self._lock:
            return iter(list(self._entries))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
