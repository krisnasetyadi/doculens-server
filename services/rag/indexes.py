"""Loading and caching FAISS indexes (PDF and chat), the BM25 indexes built from them, reciprocal rank fusion, and cache invalidation.

Mixin of services.rag.processor.PDFQAProcessor; the methods were moved
verbatim from that class by MS-657."""
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from config import config
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class IndexMixin:
    # A collection whose index can't be loaded (missing/incomplete files) is
    # skipped for a while instead of re-trying the Supabase download on EVERY
    # question — a network round trip per broken collection per query, for
    # an answer that usually never changes. Backoff, not a flat TTL: the
    # first failure may just be a network blip (e.g. right after an HF Space
    # restart, when every index has to be re-downloaded), so it's retried
    # soon; only repeated failures settle at the long ceiling.
    MISSING_VECTOR_STORE_BACKOFF_SECONDS = (30, 120, 300)

    def get_vector_store(self, collection_id):
        """Get vector store from cache, local disk, or Supabase Storage."""
        with self._cache_lock:
            retry_at, _ = self._missing_vector_stores.get(collection_id, (0.0, 0))
            if collection_id not in self.vector_store_cache and time.monotonic() < retry_at:
                logger.debug(f"Skipping {collection_id}: index failed to load recently")
                return None

        vector_store = self._load_vector_store(collection_id)

        with self._cache_lock:
            if vector_store is not None:
                self._missing_vector_stores.pop(collection_id, None)
            elif self.embeddings is not None:
                # (Embeddings not loaded yet at startup is transient — not the index's fault.)
                self._mark_missing(collection_id)
        return vector_store

    def _mark_missing(self, collection_id):
        """Skip this collection until its next backoff step; caller holds _cache_lock."""
        _, failures = self._missing_vector_stores.get(collection_id, (0.0, 0))
        backoff = self.MISSING_VECTOR_STORE_BACKOFF_SECONDS
        delay = backoff[min(failures, len(backoff) - 1)]
        self._missing_vector_stores[collection_id] = (time.monotonic() + delay, failures + 1)

    def _download_lock(self, key):
        with self._cache_lock:
            return self._download_locks.setdefault(key, threading.Lock())

    def _ensure_index_on_disk(self, collection_id, chat=False) -> bool:
        """Download the collection's index from Supabase Storage if it isn't on
        local disk yet. Runs outside _cache_lock: a download is a network round
        trip per file, and holding the cache lock through it stalled every
        other question (after an HF Space restart the disk is empty, so every
        index is downloaded). The per-collection lock keeps two requests from
        downloading the same index at once."""
        folder = config.chat_index_folder if chat else config.index_folder
        index_path = os.path.join(folder, collection_id)
        index_file = os.path.join(index_path, "index.faiss")
        if os.path.exists(index_file):
            return True
        with self._download_lock(("chat" if chat else "pdf", collection_id)):
            if os.path.exists(index_file):
                return True
            try:
                from services.storage import supabase as supabase_storage
                if supabase_storage.is_enabled():
                    logger.info(f"📥 Downloading {'chat ' if chat else ''}index from Supabase: {collection_id}")
                    download = supabase_storage.download_chat_index if chat else supabase_storage.download_index
                    if not download(collection_id, index_path):
                        logger.warning(f"❌ Supabase download incomplete for {collection_id}")
            except Exception as dl_err:
                logger.warning(f"Supabase download error: {dl_err}")
        return os.path.exists(index_file)

    def prefetch_indexes(self, max_workers: int = 8):
        """Download every active collection's index to local disk, a few at a
        time, so the first question after a restart doesn't wait for them.
        Disk only — nothing is loaded into memory (IndexCache does that on
        demand). Collections whose index can't be downloaded start their
        backoff now instead of on the first question."""
        try:
            jobs = [(cid, False) for cid in self.get_all_collections()]
            jobs += [(cid, True) for cid in self.get_all_chat_collections()]
        except Exception as exc:
            logger.warning("prefetch_indexes: could not list collections: %s", exc)
            return
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(lambda job: self._ensure_index_on_disk(*job), jobs))
        missing = [cid for (cid, chat), ok in zip(jobs, results) if not ok and not chat]
        with self._cache_lock:
            for cid in missing:
                if cid not in self.vector_store_cache:
                    self._mark_missing(cid)
        logger.info(
            "prefetch_indexes: %d/%d indexes on disk in %.1fs",
            sum(results), len(jobs), time.monotonic() - started,
        )

    def _load_vector_store(self, collection_id):
        with self._cache_lock:
            if collection_id in self.vector_store_cache:
                logger.debug(f"Returning cached vector store for {collection_id}")
                return self.vector_store_cache[collection_id]

        self._ensure_index_on_disk(collection_id)

        with self._cache_lock:
            # Another request may have loaded it while this one downloaded.
            if collection_id in self.vector_store_cache:
                return self.vector_store_cache[collection_id]

            logger.info(f"🔍 Loading vector store for collection: {collection_id}")
            index_path = os.path.join(config.index_folder, collection_id)

            if not os.path.exists(index_path):
                logger.warning(f"❌ Index path not found: {index_path}")
                return None

            if not all(os.path.exists(os.path.join(index_path, f))
                       for f in ["index.faiss", "index.pkl"]):
                logger.error(f"❌ Incomplete index files for {collection_id}")
                return None

            if self.embeddings is None:
                logger.error("❌ Embeddings model not loaded!")
                return None

            try:
                logger.info(f"🔄 Loading FAISS vector store from {index_path}")
                vector_store = FAISS.load_local(
                    index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✅ Successfully loaded vector store for {collection_id}")

                try:
                    test_results = vector_store.similarity_search("test", k=1)
                    logger.info(f"🧪 Test search returned {len(test_results)} results")
                except Exception as test_e:
                    logger.error(f"❌ Test search failed: {test_e}")
                    return None

                self.vector_store_cache[collection_id] = vector_store
                return vector_store

            except Exception as e:
                logger.error(
                    f"❌ Failed to load vector store for {collection_id}: {str(e)}"
                )
                import traceback
                logger.error(f"🔍 Traceback: {traceback.format_exc()}")
                return None

    def get_all_collections(self):
        """Return list of available PDF collection IDs (Supabase DB → S3 scan → local disk fallback)."""
        # ── Supabase DB first ───────────────────────────────────────────────
        try:
            from services.storage import supabase as supabase_storage
            if supabase_storage.has_database():
                rows = supabase_storage.list_collections()
                if rows:
                    # Only collections the user has left active are used as a
                    # default source — same "active" concept as public links
                    # and external DB connections.
                    ids = [r["collection_id"] for r in rows if r.get("status", "active") == "active"]
                    logger.info("get_all_collections: %d active of %d from Supabase DB", len(ids), len(rows))
                    return ids
        except Exception as e:
            logger.warning("Supabase list_collections failed: %s", e)

        # ── S3 bucket scan fallback ─────────────────────────────────────────
        try:
            from services.storage import supabase as supabase_storage
            if supabase_storage.is_enabled():
                ids = supabase_storage.list_collection_ids_from_s3()
                if ids:
                    logger.info("get_all_collections: %d from S3 scan", len(ids))
                    return ids
        except Exception as e:
            logger.warning("S3 collection scan failed: %s", e)

        # ── Local disk fallback ────────────────────────────────────────────
        collections = []
        if not os.path.exists(config.index_folder):
            return collections
        for entry in os.listdir(config.index_folder):
            entry_path = os.path.join(config.index_folder, entry)
            if (os.path.isdir(entry_path) and
                    os.path.exists(os.path.join(entry_path, "index.faiss"))):
                collections.append(entry)
        logger.info("get_all_collections: %d from local disk", len(collections))
        return collections

    def get_all_chat_collections(self):
        """Return list of available chat collection IDs (Supabase DB → S3 scan → local disk)."""
        # ── Supabase DB first ─────────────────────────────────────────────────
        try:
            from services.storage import supabase as supabase_storage
            if supabase_storage.has_database():
                rows = supabase_storage.list_chat_collections()
                if rows:
                    ids = [r["collection_id"] for r in rows if r.get("status", "active") == "active"]
                    logger.info("get_all_chat_collections: %d active of %d from Supabase DB", len(ids), len(rows))
                    return ids
        except Exception as e:
            logger.warning("Supabase list_chat_collections failed: %s", e)

        # ── S3 bucket scan fallback ───────────────────────────────────────────
        try:
            from services.storage import supabase as supabase_storage
            if supabase_storage.is_enabled():
                ids = supabase_storage.list_chat_collection_ids_from_s3()
                if ids:
                    logger.info("get_all_chat_collections: %d from S3 scan", len(ids))
                    return ids
        except Exception as e:
            logger.warning("S3 chat collection scan failed: %s", e)

        # ── Local disk fallback ───────────────────────────────────────────────
        collections = []
        if not os.path.exists(config.chat_index_folder):
            return collections
        for entry in os.listdir(config.chat_index_folder):
            entry_path = os.path.join(config.chat_index_folder, entry)
            if (os.path.isdir(entry_path) and
                    os.path.exists(os.path.join(entry_path, "index.faiss"))):
                collections.append(entry)
        logger.info("get_all_chat_collections: %d from local disk", len(collections))
        return collections

    def get_chat_vector_store(self, collection_id: str):
        """Get or load chat vector store from cache"""
        cache_key = f"chat_{collection_id}"
        
        with self._cache_lock:
            if cache_key in self.vector_store_cache:
                return self.vector_store_cache[cache_key]
        
        index_path = os.path.join(config.chat_index_folder, collection_id)
        self._ensure_index_on_disk(collection_id, chat=True)
        if not os.path.exists(index_path):
            logger.warning(f"Chat index not found: {index_path}")
            return None
        
        try:
            from langchain_community.vectorstores import FAISS
            vector_store = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            with self._cache_lock:
                self.vector_store_cache[cache_key] = vector_store
            
            logger.info(f"📱 Loaded chat vector store: {collection_id}")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load chat vector store {collection_id}: {e}")
            return None

    # ── Hybrid sparse+dense retrieval helpers (BM25 + FAISS via RRF) ────────

    def _bm25_tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _build_bm25(self, docs: List[Document]):
        """Build a BM25 index over documents; returns None when unavailable."""
        if not docs:
            return None
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed; falling back to dense-only retrieval")
            return None
        corpus = [self._bm25_tokenize(d.page_content) for d in docs]
        if not any(corpus):
            return None
        try:
            return BM25Okapi(corpus)
        except Exception as exc:
            logger.warning("BM25 index build failed: %s", exc)
            return None

    def _rrf_fuse(
        self,
        query: str,
        faiss_ranked: List[Tuple[Document, float]],
        bm25,
        corpus_docs: List[Document],
        top_k: int,
        rrf_k: int = 60,
    ) -> List[Tuple[Document, float]]:
        """Fuse FAISS and BM25 rankings with Reciprocal Rank Fusion.

        Returns (doc, display_similarity) pairs ordered by fused score. The
        display similarity stays FAISS-based so UI "% match" keeps its meaning;
        BM25-only docs get a conservative floor value.
        """
        scores: Dict[int, float] = {}
        doc_by_key: Dict[int, Tuple[Document, float]] = {}

        def key_of(d: Document) -> int:
            return hash(d.page_content[:200])

        for rank, (doc, sim) in enumerate(faiss_ranked):
            k = key_of(doc)
            doc_by_key.setdefault(k, (doc, sim))
            scores[k] = scores.get(k, 0.0) + 1.0 / (rrf_k + rank + 1)

        bm25_top_keys: List[int] = []
        if bm25 is not None and corpus_docs:
            tokens = self._bm25_tokenize(query)
            if tokens:
                bm25_scores = bm25.get_scores(tokens)
                order = sorted(
                    range(len(corpus_docs)),
                    key=lambda i: bm25_scores[i],
                    reverse=True,
                )
                floor_sim = max(
                    min((s for _, s in faiss_ranked), default=0.1), 0.1
                )
                for rank, idx in enumerate(order[: max(top_k * 3, 10)]):
                    if bm25_scores[idx] <= 0:
                        break
                    d = corpus_docs[idx]
                    k = key_of(d)
                    doc_by_key.setdefault(k, (d, floor_sim))
                    scores[k] = scores.get(k, 0.0) + 1.0 / (rrf_k + rank + 1)
                    if rank < 2:
                        bm25_top_keys.append(k)

        fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        # Per-file diversity cap: without it a couple of large documents can
        # occupy every slot and crowd out the file that actually answers the
        # question.
        max_per_source = 2
        selected: List[int] = []
        overflow: List[int] = []
        per_source: Dict[Any, int] = {}
        for k, _ in fused:
            src = doc_by_key[k][0].metadata.get("source")
            if per_source.get(src, 0) < max_per_source:
                selected.append(k)
                per_source[src] = per_source.get(src, 0) + 1
            else:
                overflow.append(k)
            if len(selected) >= top_k:
                break
        if len(selected) < top_k:
            selected.extend(overflow[: top_k - len(selected)])

        # Guaranteed lexical slot: plain RRF favors docs present in BOTH
        # rankings, which can push the strongest exact-keyword match out of a
        # small top_k. Keep BM25's single best hit in the result (one slot
        # only — two slots crowded out dense hits when top_k is small).
        guaranteed = [k for k in bm25_top_keys[:1] if k not in selected]
        if guaranteed and top_k > len(guaranteed):
            keep = [k for k in selected if k not in guaranteed]
            selected = (keep[: top_k - len(guaranteed)] + guaranteed)[:top_k]

        return [doc_by_key[k] for k in selected]

    def get_bm25_index(self, collection_id: str):
        """Lazily build and cache a BM25 index over an indexed collection's chunks."""
        with self._cache_lock:
            if collection_id in self.bm25_cache:
                return self.bm25_cache[collection_id]
            vector_store = self.vector_store_cache.get(collection_id)
            if vector_store is None:
                return None, []
            try:
                docs = list(vector_store.docstore._dict.values())
            except Exception as exc:
                logger.warning("BM25: could not read docstore for %s: %s", collection_id, exc)
                return None, []
            entry = (self._build_bm25(docs), docs)
            self.bm25_cache[collection_id] = entry
            return entry

    def invalidate_cache(self, collection_id=None):
        """Invalidate cache for specific collection or all"""
        with self._cache_lock:
            if collection_id:
                self._missing_vector_stores.pop(collection_id, None)
                if collection_id in self.vector_store_cache:
                    del self.vector_store_cache[collection_id]
                if collection_id in self.bm25_cache:
                    del self.bm25_cache[collection_id]
                # Also check chat cache
                chat_key = f"chat_{collection_id}"
                if chat_key in self.vector_store_cache:
                    del self.vector_store_cache[chat_key]
            else:
                self.vector_store_cache.clear()
                self.bm25_cache.clear()
                self._missing_vector_stores.clear()
