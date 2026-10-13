# tests/test_index_prefetch.py
"""Index downloads from Supabase Storage: off the cache lock, once per
collection, and prefetched in the background at startup."""
import os
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from config import config
from services.rag.processor import processor
from services.storage import supabase as supabase_storage


def _write_index(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for name in ("index.faiss", "index.pkl"):
        with open(os.path.join(dest_dir, name), "wb") as f:
            f.write(b"x")
    return True


class IndexDownloadTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = self.stack.enter_context(tempfile.TemporaryDirectory())
        for name in ("index_folder", "chat_index_folder"):
            folder = os.path.join(root, name)
            os.makedirs(folder)
            self.stack.enter_context(patch.object(config, name, folder))
        self.stack.enter_context(patch.object(supabase_storage, "is_enabled", return_value=True))
        self.stack.enter_context(patch.object(processor, "_missing_vector_stores", {}))
        self.stack.enter_context(patch.object(processor, "_download_locks", {}))

    def test_download_does_not_hold_the_cache_lock(self):
        lock_was_free = []

        def download(collection_id, dest_dir):
            # Another thread must be able to take the cache lock meanwhile.
            probe = threading.Thread(
                target=lambda: lock_was_free.append(processor._cache_lock.acquire(timeout=1) and
                                                    (processor._cache_lock.release() or True)))
            probe.start()
            probe.join()
            return _write_index(dest_dir)

        with patch.object(supabase_storage, "download_index", side_effect=download):
            self.assertTrue(processor._ensure_index_on_disk("c1"))
        self.assertEqual(lock_was_free, [True])

    def test_concurrent_requests_download_once(self):
        calls = []

        def download(collection_id, dest_dir):
            calls.append(collection_id)
            time.sleep(0.2)
            return _write_index(dest_dir)

        with patch.object(supabase_storage, "download_index", side_effect=download):
            threads = [threading.Thread(target=processor._ensure_index_on_disk, args=("c1",)) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(calls, ["c1"])

    def test_prefetch_downloads_pdf_and_chat_and_backs_off_missing(self):
        def download_pdf(collection_id, dest_dir):
            return collection_id != "gone" and _write_index(dest_dir)

        with patch.object(processor, "get_all_collections", return_value=["p1", "gone"]), \
                patch.object(processor, "get_all_chat_collections", return_value=["ch1"]), \
                patch.object(supabase_storage, "download_index", side_effect=download_pdf) as pdf, \
                patch.object(supabase_storage, "download_chat_index",
                             side_effect=lambda cid, dest: _write_index(dest)) as chat:
            processor.prefetch_indexes()

        self.assertEqual(sorted(c.args[0] for c in pdf.call_args_list), ["gone", "p1"])
        self.assertEqual([c.args[0] for c in chat.call_args_list], ["ch1"])
        self.assertTrue(os.path.exists(os.path.join(config.index_folder, "p1", "index.faiss")))
        self.assertTrue(os.path.exists(os.path.join(config.chat_index_folder, "ch1", "index.faiss")))
        self.assertEqual(list(processor._missing_vector_stores), ["gone"])

    def test_index_already_on_disk_is_not_downloaded(self):
        _write_index(os.path.join(config.index_folder, "c1"))
        with patch.object(supabase_storage, "download_index") as download:
            self.assertTrue(processor._ensure_index_on_disk("c1"))
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
