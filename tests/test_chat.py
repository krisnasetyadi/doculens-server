"""MS-657: behaviour of chat collections (router/ + services/chat.py) against real SQL,
recorded before they are moved: uploading a WhatsApp export, listing,
previewing, paging messages, activating, moving to a folder and deleting.
Upload size limits and progress streaming have their own tests
(test_storage_limits.py, test_upload_progress.py). Embeddings are faked and
Supabase Storage is off. Runs in a throwaway schema (testing_db); skipped
when no database is reachable."""

import json
import os
import tempfile
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from services.storage import supabase as storage
from config import config
from models import UserRecord
from services.rag.processor import processor
from router import chat
from services.auth import get_current_user
from tests.testing_db import IsolatedSchemaTestCase

EXPORT = (
    "[2025-12-18 08:30:15] Sarah: Hello there, how are you doing today\n"
    "[2025-12-18 08:31:00] Budi: Fine, the project is on track\n"
    "[2025-12-18 08:32:30] Sarah: Great, send the report on Friday\n"
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.0]


class ChatTestCase(IsolatedSchemaTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(processor, "embeddings", FakeEmbeddings()))
        self.stack.enter_context(patch.object(storage, "is_enabled", lambda: False))
        self.stack.enter_context(patch.object(storage, "_s3_client", lambda: None))
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for name in ("chat_upload_folder", "chat_index_folder"):
            (directory / name).mkdir()
            self.stack.enter_context(patch.object(config, name, str(directory / name)))
        self.upload_dir = directory / "chat_upload_folder"
        self.index_dir = directory / "chat_index_folder"
        self.sql("DELETE FROM collections WHERE kind = 'chat'")
        self.role = "admin"
        self.user_id = "admin-1"

    @property
    def http(self):
        app = FastAPI()
        app.include_router(chat.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id=self.user_id, email=f"{self.user_id}@example.com", role=self.role, is_active=True)
        return TestClient(app)

    def upload(self, content=EXPORT, name="chat.txt", **form):
        return self.http.post("/api/v1/chat-collections/upload", data=form,
                              files={"file": (name, content.encode(), "text/plain")})

    def registered(self, collection_id=None, owner_id="admin-1"):
        """A chat collection row in the database without uploading a file."""
        cid = collection_id or str(uuid.uuid4())
        self.assertTrue(storage.register_chat_collection(cid, "chat.txt", "whatsapp", 3, ["Budi", "Sarah"], None,
                                                         owner_id=owner_id))
        return cid


class UploadTest(ChatTestCase):
    def test_a_whatsapp_export_becomes_a_chat_collection(self):
        resp = self.upload()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual((body["file_name"], body["platform"], body["message_count"], body["status"]),
                         ("chat.txt", "whatsapp", 3, "success"))
        self.assertEqual(sorted(body["participants"]), ["Budi", "Sarah"])
        self.assertTrue(body["date_range"]["start"].startswith("2025-12-18T08:30:15"))
        cid = body["collection_id"]
        self.assertEqual((self.upload_dir / cid / "chat.txt").read_text(encoding="utf-8"), EXPORT)
        self.assertTrue((self.index_dir / cid / "metadata.json").exists())

    def test_uploads_are_registered_in_the_database_without_supabase_storage(self):
        cid = self.upload().json()["collection_id"]
        row = storage.get_chat_collection(cid, "admin-1")
        self.assertEqual((row["message_count"], sorted(row["participants"]), row["file_name"]),
                         (3, ["Budi", "Sarah"], "chat.txt"))

    def test_bad_uploads(self):
        self.assertEqual(self.upload(platform="signal").status_code, 400)
        self.assertEqual(self.upload(name="chat.csv").status_code, 400)
        self.role = "user"
        self.assertEqual(self.upload().status_code, 403)


class ListTest(ChatTestCase):
    def test_only_the_users_own_chats_never_the_ones_on_disk(self):
        """Local disk doesn't say who added a chat (MS-913), so it is never listed."""
        (self.index_dir / "old").mkdir()
        (self.index_dir / "old" / "metadata.json").write_text(
            json.dumps({"collection_id": "old", "message_count": 9}), encoding="utf-8")
        self.assertEqual(self.http.get("/api/v1/chat-collections").json(), {"collections": [], "count": 0})

        cid = self.upload().json()["collection_id"]
        self.registered(owner_id="admin-2")
        self.registered(owner_id=None)
        body = self.http.get("/api/v1/chat-collections").json()
        self.assertEqual(([c["collection_id"] for c in body["collections"]], body["count"]), ([cid], 1))


class ReadTest(ChatTestCase):
    def test_preview_and_messages(self):
        cid = self.upload().json()["collection_id"]
        preview = self.http.get(f"/api/v1/chat-collections/{cid}/preview", params={"max_chars": 500}).json()
        self.assertEqual((preview["file_name"], preview["content_preview"], preview["truncated"]), ("chat.txt", EXPORT, False))
        long_export = EXPORT * 20
        long_id = self.upload(long_export).json()["collection_id"]
        cut = self.http.get(f"/api/v1/chat-collections/{long_id}/preview", params={"max_chars": 500}).json()
        self.assertEqual((cut["content_preview"], cut["truncated"]), (long_export[:500], True))

        page = self.http.get(f"/api/v1/chat-collections/{cid}/messages", params={"offset": 1, "limit": 1}).json()
        self.assertEqual((page["subtype"], page["total"], page["has_more"], page["offset"]), ("whatsapp", 3, True, 1))
        self.assertEqual([(m["sender"], m["content"]) for m in page["messages"]], [("Budi", "Fine, the project is on track")])

    def test_missing_files(self):
        self.assertEqual(self.http.get("/api/v1/chat-collections/nope/preview").status_code, 404)
        self.assertEqual(self.http.get("/api/v1/chat-collections/nope/messages").status_code, 404)
        empty = self.registered()
        (self.upload_dir / empty).mkdir()
        self.assertEqual(self.http.get(f"/api/v1/chat-collections/{empty}/preview").status_code, 404)
        self.assertEqual(self.http.get("/api/v1/chat-collections/x/preview", params={"max_chars": 10}).status_code, 422)


class ManageTest(ChatTestCase):
    def test_activate_and_move_to_folder(self):
        cid = self.registered()
        resp = self.http.post("/api/v1/chat-collections/activate", json={"collection_id": cid, "active": False})
        self.assertEqual(resp.json(), {"status": "success", "collection_id": cid, "active": False})
        self.assertEqual(storage.get_chat_collection(cid, "admin-1")["status"], "inactive")
        self.assertEqual(self.http.post("/api/v1/chat-collections/activate",
                                        json={"collection_id": "nope", "active": True}).status_code, 404)

        folder = storage.create_folder(str(uuid.uuid4()), "Chats", "admin-1")["folder_id"]
        theirs = storage.create_folder(str(uuid.uuid4()), "Theirs", "someone-else")["folder_id"]
        move = lambda target, folder_id: self.http.post("/api/v1/chat-collections/move-to-folder",
                                                       json={"collection_id": target, "folder_id": folder_id})
        self.assertEqual(move(cid, folder).json(), {"status": "success", "collection_id": cid, "folder_id": folder})
        self.assertEqual(self.sql("SELECT folder_id FROM collections WHERE collection_id = %s", (cid,))[0]["folder_id"], folder)
        self.assertEqual(move(cid, "no-such-folder").status_code, 404)
        self.assertEqual(move(cid, theirs).status_code, 404)
        self.assertEqual(move("nope", None).status_code, 404)
        self.assertEqual(move(cid, None).json()["folder_id"], None)

    def test_delete_removes_files_row_and_cache(self):
        cid = self.upload().json()["collection_id"]
        processor.vector_store_cache[f"chat_{cid}"] = "index"
        self.assertEqual(self.http.delete(f"/api/v1/chat-collections/{cid}").json(), {"status": "deleted", "collection_id": cid})
        self.assertFalse((self.upload_dir / cid).exists() or (self.index_dir / cid).exists())
        self.assertNotIn(f"chat_{cid}", processor.vector_store_cache)
        self.assertIsNone(storage.get_chat_collection(cid, "admin-1"))
        self.assertEqual(self.http.delete(f"/api/v1/chat-collections/{cid}").status_code, 404)
        self.assertEqual(self.http.delete("/api/v1/chat-collections/never-existed").status_code, 404)
        self.assertEqual(self.http.delete(f"/api/v1/chat-collections/{self.registered()}").status_code, 200)
        self.role = "user"
        self.assertEqual(self.http.delete(f"/api/v1/chat-collections/{cid}").status_code, 403)


class PrivacyTest(ChatTestCase):
    """MS-913: a chat is private to whoever added it. Nobody else reaches it,
    another admin included, and someone else's chat reads as not found."""

    def setUp(self):
        super().setUp()
        self.own = self.upload().json()["collection_id"]  # admin-1's
        self.folder = storage.create_folder(str(uuid.uuid4()), "Shared files", "admin-1")["folder_id"]
        self.assertTrue(storage.set_collection_folder(self.own, self.folder))

    def tries(self):
        cid = self.own
        return {
            "list": self.http.get("/api/v1/chat-collections"),
            "preview": self.http.get(f"/api/v1/chat-collections/{cid}/preview"),
            "messages": self.http.get(f"/api/v1/chat-collections/{cid}/messages"),
            "activate": self.http.post("/api/v1/chat-collections/activate", json={"collection_id": cid, "active": False}),
            "move": self.http.post("/api/v1/chat-collections/move-to-folder", json={"collection_id": cid, "folder_id": None}),
            "delete": self.http.delete(f"/api/v1/chat-collections/{cid}"),
        }

    def assert_untouched(self):
        row = storage.get_chat_collection(self.own, "admin-1")
        self.assertEqual((row["status"], row["folder_id"]), ("active", self.folder))
        self.assertTrue((self.upload_dir / self.own).exists() and (self.index_dir / self.own).exists())

    def test_another_admin_finds_nothing_and_changes_nothing(self):
        for who in ("admin-2", "admin-of-another-workspace"):
            self.user_id = who
            resp = self.tries()
            self.assertEqual(resp.pop("list").json(), {"collections": [], "count": 0})
            self.assertEqual({name: r.status_code for name, r in resp.items()}, dict.fromkeys(resp, 404))
            self.assert_untouched()

    def test_a_member_of_the_workspace_is_refused(self):
        self.user_id, self.role = "member-of-admin-1", "user"
        resp = self.tries()
        self.assertEqual({name: r.status_code for name, r in resp.items()}, dict.fromkeys(resp, 403))
        self.assert_untouched()

    def test_only_the_owners_active_chats_are_searched(self):
        theirs = self.registered(owner_id="admin-2")
        nobodys = self.registered(owner_id=None)
        off = self.registered()
        storage.set_chat_collection_status(off, False, "admin-1")
        self.assertEqual(processor.get_user_chat_collections("admin-1"), [self.own])
        self.assertEqual(processor.get_user_chat_collections("admin-2"), [theirs])
        # A member the file folder above is shared with gets no chat from it.
        self.assertEqual(processor.get_user_chat_collections("member-of-admin-1"), [])
        self.assertNotIn(nobodys, processor.get_user_chat_collections("admin-1") + processor.get_user_chat_collections("admin-2"))

    def test_without_the_database_no_chat_is_searched(self):
        with patch.object(storage, "has_database", lambda: False):
            self.assertEqual(processor.get_user_chat_collections("admin-1"), [])
            self.assertEqual(self.http.get(f"/api/v1/chat-collections/{self.own}/preview").status_code, 503)

    def test_storage_files_of_someone_elses_chat_are_never_deleted(self):
        deleted = []

        class S3:
            def list_objects_v2(self, Bucket, Prefix):
                return {"Contents": [{"Key": f"{Prefix}index.faiss"}]}

            def delete_object(self, Bucket, Key):
                deleted.append(Key)

        with patch.object(storage, "_s3_client", lambda: S3()):
            self.assertFalse(storage.delete_chat_collection_from_db(self.own, "admin-2"))
            self.assertEqual(deleted, [])
            self.assertTrue(storage.delete_chat_collection_from_db(self.own, "admin-1"))
        self.assertEqual(len(deleted), 2)  # uploads and indices


if __name__ == "__main__":
    import unittest
    unittest.main()
