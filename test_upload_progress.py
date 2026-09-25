"""Tests for the coarse upload-progress loading bar: streamed checkpoints,
survival past a client disconnect, and the reload-restoration GET endpoint."""

import asyncio
import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from threading import Event
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms553-local-test-secret-not-for-production")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from reportlab.pdfgen import canvas

from config import config
from processor import processor
from router.auth import get_current_user, UserRecord
from router import upload, chat
from upload_progress import progress_response, upload_progress_store, _UploadProgressStore


class FakeEmbeddings(Embeddings):
    """Deterministic, network-free stand-in for the real HuggingFace model."""

    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.0]


def pdf_bytes():
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "Hello world, this is a test document with enough text to index.")
    c.save()
    return buf.getvalue()


WHATSAPP_EXPORT = (
    "[2025-12-18 08:30:15] Sarah: Hello there, how are you doing today\n"
    "[2025-12-18 08:31:00] Sarah: I wanted to check on the project status\n"
)


class ProgressStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_progress_arrives_before_ready(self):
        release = Event()

        def work(report):
            report("processing", 35)
            if not release.wait(2):
                raise RuntimeError("test timed out")
            report("saving", 90)
            return {"collection_id": "one", "status": "success"}

        stream = progress_response(work, upload_id="one").body_iterator
        first = json.loads(await anext(stream))
        self.assertEqual(first["stage"], "processing")
        self.assertEqual(first["upload_id"], "one")
        release.set()
        rest = [json.loads(chunk) async for chunk in stream]
        self.assertEqual([e["type"] for e in [first, *rest]], ["progress", "progress", "ready"])

    async def test_disconnect_does_not_stop_processing(self):
        release, completed = Event(), Event()

        def work(report):
            report("processing", 35)
            if not release.wait(2):
                raise RuntimeError("test timed out")
            report("saving", 90)
            completed.set()
            return {"collection_id": "one", "status": "success"}

        stream = progress_response(work, upload_id="one").body_iterator
        await anext(stream)
        await stream.aclose()
        release.set()
        self.assertTrue(await asyncio.to_thread(completed.wait, 2))

    async def test_progress_never_reports_100_before_ready(self):
        def work(report):
            report("saving", 500)
            return {"collection_id": "one", "status": "success"}

        events = [json.loads(chunk) async for chunk in progress_response(work, upload_id="one").body_iterator]
        self.assertLess(events[0]["progress"], 100)
        self.assertEqual(events[-1]["type"], "ready")

    async def test_error_reaches_the_stream_and_the_store(self):
        with patch.object(upload_progress_store, "_records", {}):
            upload_progress_store.start("one", owner_id="alice")

            def work(report):
                raise ValueError("boom")

            events = [json.loads(chunk) async for chunk in progress_response(work, upload_id="one").body_iterator]
            self.assertEqual(events[-1]["type"], "error")
            self.assertEqual(upload_progress_store.get("one", "alice")["status"], "error")


class UploadProgressStoreTests(unittest.TestCase):
    def test_get_is_owner_scoped(self):
        store = _UploadProgressStore()
        store.start("one", owner_id="alice")
        self.assertIsNone(store.get("one", owner_id="bob"))
        self.assertIsNone(store.get("missing", owner_id="alice"))
        snapshot = store.get("one", owner_id="alice")
        self.assertEqual((snapshot["status"], snapshot["stage"]), ("uploading", "reading"))
        self.assertNotIn("owner_id", snapshot)

    def test_update_reflects_progress_then_terminal_state(self):
        store = _UploadProgressStore()
        store.start("one", owner_id="alice")
        store.update("one", {"type": "progress", "stage": "processing", "progress": 40})
        self.assertEqual(store.get("one", "alice")["progress"], 40)
        store.update("one", {"type": "ready", "progress": 100, "result": {"collection_id": "one"}})
        snapshot = store.get("one", "alice")
        self.assertEqual(snapshot["status"], "success")
        self.assertEqual(snapshot["result"]["collection_id"], "one")

    def test_finished_records_expire_but_active_ones_never_do(self):
        store = _UploadProgressStore()
        with patch("upload_progress.monotonic", return_value=1):
            store.start("active", owner_id="alice")
            store.start("done", owner_id="alice")
            store.update("done", {"type": "ready", "progress": 100, "result": {}})
        with patch("upload_progress.monotonic", return_value=10 * 60 + 5):
            self.assertIsNone(store.get("done", "alice"))
            self.assertIsNotNone(store.get("active", "alice"))


class UploadRouteTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(upload_progress_store, "_records", {}))
        self.stack.enter_context(patch.object(processor, "embeddings", FakeEmbeddings()))
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for name in ("upload_folder", "index_folder", "chat_upload_folder", "chat_index_folder"):
            folder = directory / name
            folder.mkdir()
            self.stack.enter_context(patch.object(config, name, str(folder)))

        self.app = FastAPI()
        self.app.include_router(upload.router, prefix="/api/v1")
        self.app.include_router(chat.router, prefix="/api/v1")
        self.app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="test-user", email="test@example.com", role="admin", is_active=True,
        )
        self.client = TestClient(self.app)

    def events(self, response):
        return [json.loads(line) for line in response.text.strip().split("\n")]

    def test_pdf_stream_progress_reports_before_ready_and_restores_via_get(self):
        response = self.client.post(
            "/api/v1/pdf-collections/upload",
            params={"persist_mode": "local", "stream_progress": "true"},
            files={"files": ("report.pdf", pdf_bytes(), "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        events = self.events(response)
        self.assertTrue(any(e["type"] == "progress" for e in events))
        self.assertEqual(events[-1]["type"], "ready")
        collection_id = events[-1]["result"]["collection_id"]

        status = self.client.get(f"/api/v1/pdf-collections/uploads/{collection_id}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "success")
        self.assertEqual(status.headers["cache-control"], "no-store")

    def test_pdf_upload_status_is_owner_scoped(self):
        upload_progress_store.start("someones-upload", owner_id="someone-else")
        self.assertEqual(self.client.get("/api/v1/pdf-collections/uploads/someones-upload").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/pdf-collections/uploads/unknown").status_code, 404)

    def test_pdf_non_streaming_upload_is_unchanged(self):
        response = self.client.post(
            "/api/v1/pdf-collections/upload",
            params={"persist_mode": "local"},
            files={"files": ("report.pdf", pdf_bytes(), "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        self.assertTrue(response.headers["content-type"].startswith("application/json"))

    def test_chat_stream_progress_reports_before_ready_and_restores_via_get(self):
        response = self.client.post(
            "/api/v1/chat-collections/upload",
            params={"stream_progress": "true"},
            files={"file": ("chat.txt", WHATSAPP_EXPORT.encode(), "text/plain")},
        )
        self.assertEqual(response.status_code, 200)
        events = self.events(response)
        self.assertTrue(any(e["type"] == "progress" for e in events))
        self.assertEqual(events[-1]["type"], "ready")
        collection_id = events[-1]["result"]["collection_id"]

        status = self.client.get(f"/api/v1/chat-collections/uploads/{collection_id}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["result"]["message_count"], 2)

    def test_chat_upload_status_requires_admin(self):
        self.app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="test-user", email="test@example.com", role="user", is_active=True,
        )
        self.assertEqual(self.client.get("/api/v1/chat-collections/uploads/whatever").status_code, 403)

    def test_chat_upload_still_rejects_non_whatsapp_text(self):
        response = self.client.post(
            "/api/v1/chat-collections/upload",
            files={"file": ("notes.txt", b"just some plain notes, not a chat export", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("No messages found", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
