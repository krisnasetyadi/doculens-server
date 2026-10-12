"""MS-657: the memory cap on loaded search indexes (services/rag/index_cache.py)
and how the processor uses it: least recently used indexes are dropped past
the budget, their BM25 index goes with them, and they reload on demand."""

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import config
from services.rag.index_cache import TEXT_COPIES, IndexCache, estimate_index_bytes
from services.rag.processor import PDFQAProcessor


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.0, 2.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.0, 2.0]


def sized(n):
    """A stand-in index whose size is given directly."""
    return SimpleNamespace(size=n)


def cache(max_bytes, evicted=None):
    return IndexCache(max_bytes, size_of=lambda v: v.size,
                      on_evict=(evicted.append if evicted is not None else None))


class IndexCacheTest(unittest.TestCase):
    def test_least_recently_used_goes_first(self):
        evicted = []
        c = cache(100, evicted)
        c["a"], c["b"] = sized(40), sized(40)
        c["a"]  # reading marks "a" as used, so "b" is now the oldest
        c["c"] = sized(40)
        self.assertEqual((sorted(c), evicted, c.total_bytes), (["a", "c"], ["b"], 80))

    def test_membership_checks_do_not_count_as_use(self):
        c = cache(100)
        c["a"], c["b"] = sized(40), sized(40)
        self.assertIn("a", c)
        c["c"] = sized(40)
        self.assertNotIn("a", c)

    def test_an_index_larger_than_the_budget_still_loads_and_goes_first(self):
        evicted = []
        c = cache(100, evicted)
        c["small"] = sized(10)
        c["huge"] = sized(500)
        self.assertEqual((list(c), evicted), (["huge"], ["small"]))
        c["next"] = sized(10)
        self.assertEqual((list(c), evicted), (["next"], ["small", "huge"]))

    def test_dict_operations_callers_use(self):
        c = cache(1000)
        c["a"] = sized(1)
        self.assertEqual((c.get("a").size, c.get("missing"), len(c)), (1, None, 1))
        self.assertEqual(c.pop("a").size, 1)
        self.assertIsNone(c.pop("a", None))
        c["b"] = sized(1)
        del c["b"]
        c["c"] = sized(1)
        c.clear()
        self.assertEqual((len(c), c.total_bytes), (0, 0))
        with self.assertRaises(KeyError):
            c["gone"]

    def test_replacing_an_entry_counts_its_new_size_only(self):
        c = cache(100)
        c["a"] = sized(60)
        c["a"] = sized(30)
        self.assertEqual(c.total_bytes, 30)

    def test_estimate_of_a_real_faiss_store(self):
        texts = ["first chunk of text", "second, slightly longer chunk of text"]
        store = FAISS.from_documents([Document(page_content=t) for t in texts], FakeEmbeddings())
        expected = len(texts) * 4 * 4 + sum(len(t) for t in texts) * TEXT_COPIES
        self.assertEqual(estimate_index_bytes(store), expected)
        self.assertEqual(estimate_index_bytes(object()), 0)


class ProcessorCacheTest(unittest.TestCase):
    def test_the_processor_cache_uses_the_configured_budget(self):
        self.assertEqual(PDFQAProcessor().vector_store_cache.max_bytes, config.index_cache_max_bytes)

    def test_an_evicted_index_takes_its_bm25_index_along(self):
        p = PDFQAProcessor()
        p.vector_store_cache.max_bytes = 0  # every new index pushes the previous one out
        docs = [Document(page_content="leave policy allows twenty days")]
        for cid in ("c1", "c2"):
            p.vector_store_cache[cid] = FAISS.from_documents(docs, FakeEmbeddings())
            bm25, chunks = p.get_bm25_index(cid)
            self.assertEqual(len(chunks), 1)
        self.assertEqual((list(p.vector_store_cache), list(p.bm25_cache)), (["c2"], ["c2"]))
        self.assertEqual(p.get_bm25_index("c1"), (None, []))  # reloads with its index, not on its own

    def test_invalidate_still_works(self):
        p = PDFQAProcessor()
        p.vector_store_cache["c1"] = sized(1)
        p.vector_store_cache["chat_c1"] = sized(1)
        p.bm25_cache["c1"] = ("bm25", [])
        p.invalidate_cache("c1")
        self.assertEqual((len(p.vector_store_cache), p.bm25_cache), (0, {}))


if __name__ == "__main__":
    unittest.main()
