"""MS-907: a public link is indexed once, in the background, and a question only
loads that index. The build job and Sync run against real SQL in a throwaway
schema (skipped when no database is reachable); downloads, Drive and Supabase
Storage are faked, so nothing leaves the machine. The index itself is built
with a small word-count embedding instead of the real model."""

import math
import os
import re
import zlib
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET", "ms907-local-test-secret-not-for-production")

from langchain_core.embeddings import Embeddings

from config import config
from services import public_links as link_service
from services.rag.processor import PDFQAProcessor
from services.rag.public_link_search import SKIP_DOWNLOAD, public_link_cache_key
from services.storage import supabase as supabase_storage
from tests.test_public_links import FOLDER_URL, PublicLinksTestCase, drive_pages, user

FILES = {
    "FILE1": ("report.pdf.txt", "Quarterly revenue grew because the Jakarta warehouse opened in March."),
    "FILE2": ("notes.txt", "Safety training is held every Monday for all warehouse staff."),
}


def fake_download(self, item, destination_dir):
    """Writes a known text file for FILE1/FILE2 and fails anything else, as an unreadable file would."""
    for file_id, (name, text) in FILES.items():
        if file_id in item["url"]:
            path = os.path.join(destination_dir, name.replace(".pdf.txt", ".txt"))
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            return path, os.path.basename(path)
    return None


class WordEmbeddings(Embeddings):
    """Unit-length word counts: texts that share words are close, as with the real model."""

    def _embed(self, text):
        vector = [0.0] * 64
        for word in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(word.encode()) % 64] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def new_processor():
    processor = PDFQAProcessor()
    processor.embeddings = WordEmbeddings()
    return processor


class IndexFolderTestCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.mkdtemp(prefix="ms907-index-")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        for p in (patch.object(config, "public_link_index_folder", folder),
                  patch("services.rag.public_link_search.PublicLinkSearchMixin._download_public_link_item", fake_download)):
            p.start()
            self.addCleanup(p.stop)


class SearchTest(IndexFolderTestCase):
    """A question searches the saved index and downloads nothing."""

    def build(self, processor, link_id, items):
        store, skipped = processor.build_public_link_index(
            {"link_id": link_id, "title": "Shared", "url": FOLDER_URL, "items": items}
        )
        processor.save_public_link_index(link_id, "v1", store)
        return skipped

    def test_built_once_then_searched_from_disk_without_downloading(self):
        items = [{"name": "report.pdf", "url": "https://drive.google.com/file/d/FILE1/view", "item_type": "file"},
                 {"name": "notes.txt", "url": "https://drive.google.com/file/d/FILE2/view", "item_type": "file"},
                 {"name": "broken.pdf", "url": "https://drive.google.com/file/d/GONE/view", "item_type": "file"}]
        skipped = self.build(new_processor(), "link-a", items)
        self.assertEqual(skipped, [{"name": "broken.pdf", "url": items[2]["url"], "reason": SKIP_DOWNLOAD}])

        # Another process: nothing in memory, the index comes from disk.
        processor = new_processor()
        with patch("services.rag.public_link_search.PublicLinkSearchMixin._download_public_link_item") as download, \
                patch("requests.get") as http:
            results = processor.search_public_link_indexes(
                "warehouse safety training", [{"link_id": "link-a", "index_version": "v1"}], top_k=5
            )
        download.assert_not_called()
        http.assert_not_called()
        self.assertTrue(results)
        self.assertEqual(results[0].metadata["source"], "notes.txt")  # the file about safety training
        self.assertEqual({doc.metadata["source_kind"] for doc in results}, {"public_link"})
        self.assertEqual({doc.metadata["public_link_id"] for doc in results}, {"link-a"})
        self.assertNotIn("file_path", results[0].metadata)
        # Held in the index cache, so it counts toward its memory cap.
        self.assertIn(public_link_cache_key("link-a"), processor.vector_store_cache)

    def test_a_link_without_an_index_is_not_searched(self):
        processor = new_processor()
        self.assertEqual(processor.search_public_link_indexes("anything", [{"link_id": "x", "index_version": None}]), [])

    def test_a_newer_version_replaces_the_cached_one(self):
        processor = new_processor()
        self.build(processor, "link-b", [{"name": "report.pdf", "url": "https://drive.google.com/file/d/FILE1/view"}])
        old = processor.get_public_link_vector_store("link-b", "v1")
        store, _ = processor.build_public_link_index(
            {"link_id": "link-b", "title": "Shared", "url": FOLDER_URL,
             "items": [{"name": "notes.txt", "url": "https://drive.google.com/file/d/FILE2/view"}]}
        )
        store.save_local(os.path.join(config.public_link_index_folder, "link-b", "v2"))
        newer = processor.get_public_link_vector_store("link-b", "v2")
        self.assertIsNot(newer, old)
        self.assertEqual(processor._public_link_versions["link-b"], "v2")
        # Loading a version deletes nothing; the build prunes once the database moved on.
        link_dir = os.path.join(config.public_link_index_folder, "link-b")
        self.assertEqual(sorted(os.listdir(link_dir)), ["v1", "v2"])
        processor.prune_public_link_versions("link-b", keep_version="v2")
        self.assertEqual(os.listdir(link_dir), ["v2"])

    def test_an_index_that_cant_be_loaded_is_not_retried_on_every_question(self):
        processor = new_processor()
        with patch.object(supabase_storage, "download_public_link_index", return_value=False) as download:
            self.assertIsNone(processor.get_public_link_vector_store("gone", "v1"))
            self.assertIsNone(processor.get_public_link_vector_store("gone", "v1"))
        self.assertEqual(download.call_count, 1)

    def test_dropping_forgets_memory_and_disk(self):
        processor = new_processor()
        self.build(processor, "link-c", [{"name": "report.pdf", "url": "https://drive.google.com/file/d/FILE1/view"}])
        processor.drop_public_link_index("link-c")
        self.assertNotIn(public_link_cache_key("link-c"), processor.vector_store_cache)
        self.assertFalse(os.path.exists(os.path.join(config.public_link_index_folder, "link-c")))


class FingerprintTest(unittest.TestCase):
    ITEMS = [{"name": "a.pdf", "url": "https://drive.google.com/file/d/A/view"},
             {"name": "b.pdf", "url": "https://drive.google.com/file/d/B/view"}]

    def test_order_and_url_shape_do_not_matter(self):
        reordered = [{"name": "b.pdf", "url": "https://drive.google.com/open?id=B"}, self.ITEMS[0]]
        self.assertEqual(link_service.fingerprint_items(self.ITEMS), link_service.fingerprint_items(reordered))

    def test_added_removed_or_renamed_files_change_it(self):
        base = link_service.fingerprint_items(self.ITEMS)
        self.assertNotEqual(base, link_service.fingerprint_items(self.ITEMS[:1]))
        self.assertNotEqual(base, link_service.fingerprint_items(
            self.ITEMS + [{"name": "c.pdf", "url": "https://drive.google.com/file/d/C/view"}]))
        self.assertNotEqual(base, link_service.fingerprint_items(
            [self.ITEMS[0], {"name": "b-final.pdf", "url": self.ITEMS[1]["url"]}]))


class StorageScanTest(unittest.TestCase):
    def test_public_link_indexes_are_not_read_as_a_collection(self):
        resp = {"CommonPrefixes": [{"Prefix": "abc/"}, {"Prefix": "public-links/"}]}
        self.assertEqual(supabase_storage._collection_prefixes(resp), ["abc"])


class BuildJobTest(PublicLinksTestCase, IndexFolderTestCase):
    """The background job, run in the test's own thread."""

    def setUp(self):
        PublicLinksTestCase.setUp(self)
        IndexFolderTestCase.setUp(self)
        self.processor = new_processor()
        for p in (patch("services.rag.processor.processor", self.processor),
                  patch.object(supabase_storage, "is_enabled", return_value=False)):
            p.start()
            self.addCleanup(p.stop)

    def link(self, link_id):
        return self.sql("SELECT * FROM public_links WHERE link_id = %s", (link_id,))[0]

    def run_jobs(self):
        jobs, self.jobs[:] = list(self.jobs), []
        for name, args in jobs:
            getattr(link_service, name)(*args)

    def test_first_build_makes_the_link_ready_and_active(self):
        who = user()
        link_id = self.create(who, url=FOLDER_URL).json()["link_id"]
        self.run_jobs()
        row = self.link(link_id)
        self.assertEqual((row["index_status"], row["status"], row["index_progress"]), ("ready", "active", 100))
        self.assertTrue(row["index_version"] and row["fingerprint"] and row["last_synced_at"])
        # Sources now lists the files found in the folder and its sub-folder.
        names = sorted(r["name"] for r in self.sql("SELECT name FROM public_link_items WHERE link_id = %s", (link_id,)))
        self.assertEqual(names, ["notes.txt", "report.pdf"])

    def test_a_rebuild_keeps_the_user_off_switch_and_the_old_index_until_done(self):
        who = user()
        link_id = self.create(who, url=FOLDER_URL).json()["link_id"]
        self.run_jobs()
        first = self.link(link_id)["index_version"]
        self.sql("UPDATE public_links SET status = 'inactive', fingerprint = 'changed' WHERE link_id = %s", (link_id,))

        seen = []
        real_build = self.processor.build_public_link_index

        def build_and_look(*args, **kwargs):
            seen.append(self.link(link_id)["index_version"])
            return real_build(*args, **kwargs)

        with patch.object(self.processor, "build_public_link_index", build_and_look):
            resp = self.client(who).post(f"/api/v1/public-links/{link_id}/sync")
            self.assertEqual(resp.json()["result"], "indexing")
            self.run_jobs()
        self.assertEqual(seen, [first])
        row = self.link(link_id)
        self.assertNotEqual(row["index_version"], first)
        self.assertEqual((row["index_status"], row["status"]), ("ready", "inactive"))

    def test_nothing_readable_fails_with_a_reason_and_lists_the_files(self):
        who = user()
        link_id = self.create(who, url="https://example.com/", item_urls=["https://example.com/gone.pdf"]).json()["link_id"]
        self.run_jobs()
        row = self.link(link_id)
        self.assertEqual((row["index_status"], row["index_error"]), ("failed", "No file in this link could be read."))
        self.assertEqual([i["name"] for i in row["skipped_items"]], ["gone.pdf"])
        listed = self.client(who).get("/api/v1/public-links").json()["links"][0]
        self.assertEqual((listed["index_status"], listed["index_ready"]), ("failed", False))
        self.assertEqual(listed["skipped_items"][0]["reason"], SKIP_DOWNLOAD)

    def test_an_unreachable_folder_fails_instead_of_emptying_the_index(self):
        who = user()
        link_id = self.create(who, url=FOLDER_URL).json()["link_id"]
        with patch("httpx.AsyncClient.get", drive_pages({})):
            self.run_jobs()
        row = self.link(link_id)
        self.assertEqual(row["index_status"], "failed")
        self.assertIn("still shared publicly", row["index_error"])

    def test_a_link_deleted_mid_build_leaves_no_index(self):
        who = user()
        link_id = self.create(who, url=FOLDER_URL).json()["link_id"]
        real_build = self.processor.build_public_link_index

        def build_then_delete(*args, **kwargs):
            result = real_build(*args, **kwargs)
            self.sql("DELETE FROM public_links WHERE link_id = %s", (link_id,))
            return result

        with patch.object(self.processor, "build_public_link_index", build_then_delete), \
                patch.object(supabase_storage, "delete_public_link_index") as delete:
            self.run_jobs()
        delete.assert_called_once_with(link_id)
        self.assertNotIn(public_link_cache_key(link_id), self.processor.vector_store_cache)

    def test_one_build_at_a_time_and_a_cut_off_one_is_taken_over(self):
        who = user()
        link_id = self.create(who, url=FOLDER_URL).json()["link_id"]
        self.assertFalse(link_service.start_indexing(link_id))  # queued
        self.sql("UPDATE public_links SET index_status = 'indexing', index_heartbeat_at = now() WHERE link_id = %s",
                 (link_id,))
        self.assertFalse(link_service.start_indexing(link_id))  # running, still reporting
        self.sql("UPDATE public_links SET index_heartbeat_at = now() - interval '2 hours' WHERE link_id = %s", (link_id,))
        self.assertTrue(link_service.start_indexing(link_id))  # stopped reporting

    def test_a_long_queue_is_not_mistaken_for_cut_off_builds(self):
        """Waiting in the queue is not a stopped build, however long it waits."""
        who = user()
        link_id = self.create(who, url=FOLDER_URL).json()["link_id"]
        self.sql("UPDATE public_links SET updated_at = now() - interval '2 hours' WHERE link_id = %s", (link_id,))
        self.assertFalse(link_service.start_indexing(link_id))
        listed = self.client(who).get("/api/v1/public-links").json()["links"][0]
        self.assertEqual((listed["index_status"], listed["index_stale"]), ("queued", False))

    def test_startup_queues_what_the_last_process_left_behind(self):
        who = user()
        ids = {state: self.create(who, url=f"https://example.com/{state}.pdf").json()["link_id"]
               for state in ("pending", "queued", "indexing", "failed", "ready")}
        for state, link_id in ids.items():
            self.sql("UPDATE public_links SET index_status = %s, index_heartbeat_at = now() WHERE link_id = %s",
                     (state, link_id))
        self.jobs.clear()
        link_service.start_pending_indexing()
        queued = {args[0] for _, args in self.jobs}  # other tests' links may be queued too
        self.assertTrue({ids["pending"], ids["queued"], ids["indexing"]} <= queued)
        self.assertFalse({ids["failed"], ids["ready"]} & queued)

    def test_a_build_that_raises_is_marked_failed(self):
        who = user()
        link_id = self.create(who, url=FOLDER_URL).json()["link_id"]
        with patch.object(self.processor, "build_public_link_index", side_effect=MemoryError()):
            link_service._run_job(link_service._build_index, link_id, None)
        row = self.link(link_id)
        self.assertEqual((row["index_status"], row["index_error"]), ("failed", link_service.UNEXPECTED_FAILURE))

    def test_shutdown_stops_the_build_and_leaves_it_for_the_next_startup(self):
        who = user()
        link_id = self.create(who, url=FOLDER_URL).json()["link_id"]
        link_service._stopping.set()
        self.addCleanup(link_service._stopping.clear)
        link_service._run_job(link_service._build_index, link_id, None)
        self.assertEqual(self.link(link_id)["index_status"], "queued")
        link_service._stopping.clear()
        self.jobs.clear()
        link_service.start_pending_indexing()
        self.assertIn(("_build_index", (link_id, None)), self.jobs)

    def test_a_failed_rebuild_keeps_the_file_list_of_the_index_in_use(self):
        who = user()
        link_id = self.create(who, url=FOLDER_URL).json()["link_id"]
        self.run_jobs()
        before = sorted(r["name"] for r in self.sql("SELECT name FROM public_link_items WHERE link_id = %s", (link_id,)))
        pages = {"ROOTFOLDER": '<a href="https://drive.google.com/file/d/FILE9/view">new.pdf</a>'}
        with patch("httpx.AsyncClient.get", drive_pages(pages)):
            self.client(who).post(f"/api/v1/public-links/{link_id}/sync")
            self.run_jobs()  # FILE9 can't be downloaded, so the rebuild fails
        self.assertEqual(self.link(link_id)["index_status"], "failed")
        after = sorted(r["name"] for r in self.sql("SELECT name FROM public_link_items WHERE link_id = %s", (link_id,)))
        self.assertEqual(after, before)

    def test_the_old_local_version_stays_until_the_database_moves_on(self):
        who = user()
        link_id = self.create(who, url=FOLDER_URL).json()["link_id"]
        self.run_jobs()
        first = self.link(link_id)["index_version"]
        link_dir = os.path.join(config.public_link_index_folder, link_id)
        seen = []

        def upload_and_look(*args):
            seen.append(sorted(os.listdir(link_dir)))
            return True

        self.sql("UPDATE public_links SET fingerprint = 'changed' WHERE link_id = %s", (link_id,))
        with patch.object(supabase_storage, "is_enabled", return_value=True), \
                patch.object(supabase_storage, "upload_public_link_index", upload_and_look), \
                patch.object(supabase_storage, "delete_public_link_index", return_value=True):
            self.client(who).post(f"/api/v1/public-links/{link_id}/sync")
            self.run_jobs()
        second = self.link(link_id)["index_version"]
        self.assertEqual(seen, [sorted([first, second])])
        self.assertEqual(os.listdir(link_dir), [second])

    def test_activating_needs_an_index(self):
        who = user()
        link_id = self.create(who, url=FOLDER_URL).json()["link_id"]
        resp = self.client(who).post("/api/v1/public-links/activate", json={"link_id": link_id, "active": True})
        self.assertEqual(resp.status_code, 409)


class SyncTest(BuildJobTest):
    def indexed_link(self, who):
        link_id = self.create(who, url=FOLDER_URL).json()["link_id"]
        self.run_jobs()
        return link_id

    def sync(self, who, link_id):
        return self.client(who).post(f"/api/v1/public-links/{link_id}/sync")

    def test_nothing_changed_says_up_to_date(self):
        who = user()
        link_id = self.indexed_link(who)
        version = self.link(link_id)["index_version"]
        resp = self.sync(who, link_id)
        self.assertEqual(resp.json()["result"], "up_to_date")
        self.assertEqual(self.jobs, [])
        self.assertEqual(self.link(link_id)["index_version"], version)

    def test_a_new_file_rebuilds(self):
        who = user()
        link_id = self.indexed_link(who)
        pages = {"ROOTFOLDER": '<a href="https://drive.google.com/file/d/FILE1/view">report.pdf</a>'
                                 '<a href="https://drive.google.com/file/d/FILE3/view">new.pdf</a>'}
        with patch("httpx.AsyncClient.get", drive_pages(pages)):
            body = self.sync(who, link_id).json()
        self.assertEqual((body["result"], body["link"]["index_status"]), ("indexing", "queued"))
        self.assertEqual([name for name, _ in self.jobs], ["_build_index"])

    def test_a_single_file_link_always_rebuilds(self):
        """Its file list can't change, so only a rebuild can pick up an edited file."""
        who = user()
        link_id = self.create(who, url="https://drive.google.com/file/d/FILE1/view").json()["link_id"]
        self.run_jobs()
        self.assertEqual(self.link(link_id)["index_status"], "ready")
        self.assertEqual(self.sync(who, link_id).json()["result"], "indexing")

    def test_a_cut_off_build_can_be_restarted_by_sync(self):
        who = user()
        link_id = self.indexed_link(who)
        self.sql("UPDATE public_links SET index_status = 'indexing', index_heartbeat_at = now() - interval '2 hours' "
                 "WHERE link_id = %s", (link_id,))
        listed = self.client(who).get("/api/v1/public-links").json()["links"][0]
        self.assertTrue(listed["index_stale"])
        self.assertEqual(self.sync(who, link_id).json()["result"], "indexing")
        self.assertEqual([name for name, _ in self.jobs], ["_build_index"])

    def test_unreachable_link_and_strangers(self):
        who = user()
        link_id = self.indexed_link(who)
        with patch("httpx.AsyncClient.get", drive_pages({})):
            self.assertEqual(self.sync(who, link_id).status_code, 502)
        self.assertEqual(self.sync(user(), link_id).status_code, 404)


class DeleteIndexTest(unittest.TestCase):
    def test_storage_failure_is_logged_not_raised(self):
        processor = MagicMock()
        with patch("services.rag.processor.processor", processor), \
                patch.object(supabase_storage, "is_enabled", return_value=True), \
                patch.object(supabase_storage, "delete_public_link_index", return_value=False), \
                self.assertLogs("services.public_links", level="ERROR"):
            link_service._delete_index("link-z")
        processor.drop_public_link_index.assert_called_once_with("link-z")


if __name__ == "__main__":
    unittest.main()
