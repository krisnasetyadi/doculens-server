"""MS-657: behaviour of chat collections (router/chat.py) against real SQL,
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

import storage
from config import config
from models import UserRecord
from processor import processor
from router import chat
from services.auth import get_current_user
from testing_db import IsolatedSchemaTestCase

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

    @property
    def http(self):
        app = FastAPI()
        app.include_router(chat.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="admin-1", email="admin@example.com", role=self.role, is_active=True)
        return TestClient(app)

    def upload(self, content=EXPORT, name="chat.txt", **form):
        return self.http.post("/api/v1/chat-collections/upload", data=form,
                              files={"file": (name, content.encode(), "text/plain")})

    def registered(self, collection_id=None):
        """A chat collection row in the database (uploads only register one when S3 is on)."""
        cid = collection_id or str(uuid.uuid4())
        self.assertTrue(storage.register_chat_collection(cid, "chat.txt", "whatsapp", 3, ["Budi", "Sarah"], None))
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

    def test_bad_uploads(self):
        self.assertEqual(self.upload(platform="signal").status_code, 400)
        self.assertEqual(self.upload(name="chat.csv").status_code, 400)
        self.role = "user"
        self.assertEqual(self.upload().status_code, 403)


class ListTest(ChatTestCase):
    def test_disk_is_used_until_the_database_has_rows(self):
        cid = self.upload().json()["collection_id"]
        (self.index_dir / "half-built").mkdir()
        body = self.http.get("/api/v1/chat-collections").json()
        by_id = {c["collection_id"]: c for c in body["collections"]}
        self.assertEqual(body["count"], 2)
        self.assertEqual((by_id[cid]["message_count"], by_id["half-built"]["status"]), (3, "no_metadata"))

        row = self.registered()
        body = self.http.get("/api/v1/chat-collections").json()
        self.assertEqual(([c["collection_id"] for c in body["collections"]], body["count"]), ([row], 1))


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
        self.assertEqual(storage.get_chat_collection(cid)["status"], "inactive")
        self.assertEqual(self.http.post("/api/v1/chat-collections/activate",
                                        json={"collection_id": "nope", "active": True}).status_code, 404)

        folder = storage.create_folder(str(uuid.uuid4()), "Chats", "someone-else")["folder_id"]
        move = lambda target, folder_id: self.http.post("/api/v1/chat-collections/move-to-folder",
                                                       json={"collection_id": target, "folder_id": folder_id})
        self.assertEqual(move(cid, folder).json(), {"status": "success", "collection_id": cid, "folder_id": folder})
        self.assertEqual(self.sql("SELECT folder_id FROM collections WHERE collection_id = %s", (cid,))[0]["folder_id"], folder)
        self.assertEqual(move(cid, "no-such-folder").status_code, 404)
        self.assertEqual(move("nope", None).status_code, 404)
        self.assertEqual(move(cid, None).json()["folder_id"], None)

    def test_delete_removes_files_row_and_cache(self):
        cid = self.upload().json()["collection_id"]
        self.registered(cid)
        processor.vector_store_cache[f"chat_{cid}"] = "index"
        self.assertEqual(self.http.delete(f"/api/v1/chat-collections/{cid}").json(), {"status": "deleted", "collection_id": cid})
        self.assertFalse((self.upload_dir / cid).exists() or (self.index_dir / cid).exists())
        self.assertNotIn(f"chat_{cid}", processor.vector_store_cache)
        self.assertIsNone(storage.get_chat_collection(cid))
        self.role = "user"
        self.assertEqual(self.http.delete(f"/api/v1/chat-collections/{cid}").status_code, 403)


if __name__ == "__main__":
    import unittest
    unittest.main()
