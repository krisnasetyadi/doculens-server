"""Tests for MS-504: per-file size, batch size and workspace storage quota."""

import asyncio
import io
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms504-local-test-secret-not-for-production")

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from reportlab.pdfgen import canvas

import storage_limits
from config import config
from processor import processor
from router import chat, payment, upload
from router.auth import get_current_user, UserRecord
from storage_limits import GB, MB, StorageLimits
from upload_progress import upload_progress_store

LIMITS = StorageLimits(plan_name="Test", storage_limit_bytes=100, max_file_bytes=10, max_batch_files=3)


class FakeConn:
    """Stands in for the app database: answers the used-bytes query only."""

    def __init__(self, used: int):
        self.used = used
        self.closed = False

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql, self.params = sql, params

    def fetchone(self):
        return {"used": self.used}

    def close(self):
        self.closed = True


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.0]


def pdf_bytes(padding: int = 0) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "Hello world, this is a test document with enough text to index.")
    c.save()
    return buf.getvalue() + b"\n%" + b"x" * padding


class MessageTests(unittest.TestCase):
    def test_sizes_read_like_the_ticket_copy(self):
        self.assertEqual(storage_limits.format_size(50 * MB), "50 MB")
        self.assertEqual(storage_limits.format_size(5 * GB), "5 GB")
        self.assertEqual(storage_limits.format_size(int(1.5 * GB)), "1.5 GB")
        self.assertEqual(storage_limits.format_size(int(4.97 * GB)), "4.9 GB")

    def test_error_messages_and_statuses(self):
        limits = StorageLimits("Team", 5 * GB, 50 * MB, 10)
        too_large = storage_limits.file_too_large_error(limits)
        self.assertEqual((too_large.status_code, too_large.detail),
                         (413, "File exceeds the 50 MB maximum size limit."))
        batch = storage_limits.batch_too_large_error(limits)
        self.assertEqual((batch.status_code, batch.detail),
                         (400, "You can only upload up to 10 files at a time."))
        quota = storage_limits.quota_exceeded_error()
        self.assertEqual((quota.status_code, quota.detail),
                         (402, "Storage Limit Reached. Please delete older files to free up space."))


class SavingTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.dest = os.path.join(self.dir.name, "file.bin")

    def test_measure_uses_the_real_bytes_and_keeps_the_position(self):
        f = io.BytesIO(b"x" * 7)
        f.seek(2)
        self.assertEqual(storage_limits.measure(f), 7)
        self.assertEqual(f.tell(), 2)

    def test_a_file_exactly_at_the_limit_is_saved(self):
        written = storage_limits.save_capped(io.BytesIO(b"x" * 10), self.dest, 10, LIMITS)
        self.assertEqual(written, 10)
        self.assertEqual(os.path.getsize(self.dest), 10)

    def test_a_file_over_the_limit_is_refused_and_leaves_nothing(self):
        with self.assertRaises(HTTPException) as caught:
            storage_limits.save_capped(io.BytesIO(b"x" * 11), self.dest, 10, LIMITS)
        self.assertEqual(caught.exception.status_code, 413)
        self.assertFalse(os.path.exists(self.dest))

    def test_the_cap_applies_mid_stream_not_only_to_the_total(self):
        with patch.object(storage_limits, "_COPY_CHUNK", 4):
            with self.assertRaises(HTTPException):
                storage_limits.save_capped(io.BytesIO(b"x" * 9), self.dest, 8, LIMITS)
        self.assertFalse(os.path.exists(self.dest))


class ReserveTests(unittest.TestCase):
    def setUp(self):
        storage_limits._reservations.clear()

    def test_fits_exactly(self):
        reservation = storage_limits.reserve(FakeConn(used=60), "ws", LIMITS, 40)
        self.assertEqual(reservation.size, 40)

    def test_one_byte_over_is_refused_with_the_quota_error(self):
        with self.assertRaises(HTTPException) as caught:
            storage_limits.reserve(FakeConn(used=60), "ws", LIMITS, 41)
        self.assertEqual(caught.exception.status_code, 402)

    def test_ts03_a_nearly_full_workspace_refuses_a_bigger_upload(self):
        limits = StorageLimits("Team", 5 * GB, 50 * MB, 10)
        used = int(4.95 * GB)
        with self.assertRaises(HTTPException) as caught:
            storage_limits.reserve(FakeConn(used=used), "ws", limits, 100 * MB)
        self.assertEqual(caught.exception.status_code, 402)

    def test_two_uploads_cannot_both_take_the_last_space(self):
        conn = FakeConn(used=50)
        first = storage_limits.reserve(conn, "ws", LIMITS, 30)
        with self.assertRaises(HTTPException):
            storage_limits.reserve(conn, "ws", LIMITS, 30)
        first.release()
        storage_limits.reserve(conn, "ws", LIMITS, 30)

    def test_release_is_idempotent(self):
        conn = FakeConn(used=0)
        reservation = storage_limits.reserve(conn, "ws", LIMITS, 100)
        reservation.release()
        reservation.release()
        storage_limits.reserve(conn, "ws", LIMITS, 100)

    def test_workspaces_do_not_share_reservations(self):
        conn = FakeConn(used=0)
        storage_limits.reserve(conn, "a", LIMITS, 100)
        storage_limits.reserve(conn, "b", LIMITS, 100)

    def test_an_abandoned_reservation_expires(self):
        conn = FakeConn(used=0)
        with patch("storage_limits.time.monotonic", return_value=0):
            storage_limits.reserve(conn, "ws", LIMITS, 100)
        with patch("storage_limits.time.monotonic",
                   return_value=storage_limits._RESERVATION_TTL_SECONDS + 1):
            storage_limits.reserve(conn, "ws", LIMITS, 100)

    def test_without_a_database_the_quota_is_skipped_not_fatal(self):
        reservation = storage_limits.reserve(None, "ws", LIMITS, 10 * GB)
        reservation.release()

    def test_used_bytes_sums_the_admin_and_their_members(self):
        conn = FakeConn(used=123)
        self.assertEqual(storage_limits.get_used_bytes(conn, "admin-1"), 123)
        self.assertEqual(conn.params, ("admin-1", "admin-1"))
        self.assertIn("created_by", conn.sql)


class RouteTests(unittest.TestCase):
    """The upload endpoints with tiny limits, so no 50 MB file is needed."""

    def setUp(self):
        storage_limits._reservations.clear()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(upload_progress_store, "_records", {}))
        self.stack.enter_context(patch.object(processor, "embeddings", FakeEmbeddings()))
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for name in ("upload_folder", "index_folder", "chat_upload_folder", "chat_index_folder"):
            folder = directory / name
            folder.mkdir()
            self.stack.enter_context(patch.object(config, name, str(folder)))
        self.upload_dir = directory / "upload_folder"

        self.limits = StorageLimits("Test", storage_limit_bytes=1_000_000, max_file_bytes=20_000, max_batch_files=2)
        self.used = 0
        for module in (upload, chat):
            self.stack.enter_context(patch.object(
                module, "resolve_storage_limits", lambda user: (self.limits, "workspace-1")))
        self.stack.enter_context(patch.object(
            storage_limits.app_db, "get_app_conn", lambda source="": FakeConn(self.used)))

        self.app = FastAPI()
        self.app.include_router(upload.router, prefix="/api/v1")
        self.app.include_router(chat.router, prefix="/api/v1")
        self.app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="u1", email="u@example.com", role="admin", is_active=True)
        self.client = TestClient(self.app)

    def post(self, *files, **params):
        return self.client.post(
            "/api/v1/pdf-collections/upload",
            params={"persist_mode": "local", **params},
            files=[("files", (name, data, "application/pdf")) for name, data in files],
        )

    def test_ts01_file_over_the_limit_is_blocked_and_nothing_is_written(self):
        response = self.post(("big.pdf", pdf_bytes(padding=30_000)))
        self.assertEqual(response.status_code, 413)
        self.assertTrue(response.json()["detail"].startswith("File exceeds the "))
        self.assertTrue(response.json()["detail"].endswith(" maximum size limit."))
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_a_file_within_the_limit_uploads(self):
        response = self.post(("ok.pdf", pdf_bytes()))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")

    def test_ts02_too_many_files_is_rejected_whole(self):
        response = self.post(("a.pdf", pdf_bytes()), ("b.pdf", pdf_bytes()), ("c.pdf", pdf_bytes()))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "You can only upload up to 2 files at a time.")
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_exactly_the_batch_limit_is_allowed(self):
        response = self.post(("a.pdf", pdf_bytes()), ("b.pdf", pdf_bytes()))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["file_count"], 2)

    def test_ts03_full_workspace_blocks_the_upload_entirely(self):
        self.used = self.limits.storage_limit_bytes - 10
        response = self.post(("ok.pdf", pdf_bytes()))
        self.assertEqual(response.status_code, 402)
        self.assertEqual(response.json()["detail"],
                         "Storage Limit Reached. Please delete older files to free up space.")
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_the_reservation_is_released_after_a_successful_upload(self):
        self.assertEqual(self.post(("ok.pdf", pdf_bytes())).status_code, 200)
        self.assertEqual(storage_limits._reservations.get("workspace-1", {}), {})

    def test_the_reservation_is_released_after_a_failed_upload(self):
        with patch.object(upload, "process_pdfs", side_effect=RuntimeError("boom")):
            self.assertEqual(self.post(("ok.pdf", pdf_bytes())).status_code, 500)
        self.assertEqual(storage_limits._reservations.get("workspace-1", {}), {})

    def test_files_with_an_unsupported_extension_do_not_count_towards_the_size_limit(self):
        response = self.post(("ok.pdf", pdf_bytes()), ("huge.exe", b"x" * 50_000))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["file_count"], 1)

    def test_the_registered_size_is_the_bytes_that_were_written(self):
        data = pdf_bytes()
        with patch.object(upload, "_register_uploaded_collection") as register:
            self.assertEqual(self.post(("ok.pdf", data)).status_code, 200)
        self.assertEqual(register.call_args.kwargs["size_bytes"], len(data))

    def test_chat_upload_over_the_limit_is_blocked(self):
        export = ("[2025-12-18 08:30:15] Sarah: " + "hello " * 5000 + "\n").encode()
        response = self.client.post(
            "/api/v1/chat-collections/upload",
            files={"file": ("chat.txt", export, "text/plain")},
        )
        self.assertEqual(response.status_code, 413)


class RemoteDownloadTests(unittest.TestCase):
    """upload-from-url(s) download the file themselves, so the cap has to hold
    while streaming (a server may omit or lie about Content-Length)."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.dest = os.path.join(self.dir.name, "remote.pdf")
        self.real_client = httpx.AsyncClient

    def download(self, body: bytes, headers: dict | None = None):
        def handler(request):
            return httpx.Response(200, content=body, headers={"content-type": "application/pdf", **(headers or {})})

        def make_client(*args, **kwargs):
            return self.real_client(transport=httpx.MockTransport(handler), follow_redirects=True)

        with patch.object(upload, "assert_public_url_safe"), \
             patch.object(upload.httpx, "AsyncClient", make_client):
            return asyncio.run(upload._download_remote_pdf("https://example.com/a.pdf", self.dest, LIMITS))

    def test_a_file_within_the_limit_downloads(self):
        self.download(b"x" * 10)
        self.assertEqual(os.path.getsize(self.dest), 10)

    def test_an_oversized_body_is_cut_off_and_removed(self):
        with self.assertRaises(HTTPException) as caught:
            self.download(b"x" * 11)
        self.assertEqual(caught.exception.status_code, 413)
        self.assertFalse(os.path.exists(self.dest))

    def test_a_declared_length_over_the_limit_is_refused_before_writing(self):
        with self.assertRaises(HTTPException) as caught:
            self.download(b"x" * 5, headers={"content-length": "9999"})
        self.assertEqual(caught.exception.status_code, 413)
        self.assertFalse(os.path.exists(self.dest))


class UsageEndpointTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(payment.router, prefix="/api/v1")
        self.app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="u1", email="u@example.com", role="user", is_active=True)
        self.client = TestClient(self.app)

    def usage(self, used: int, limits: StorageLimits = StorageLimits("Team", 5 * GB, 50 * MB, 10)):
        with patch.object(payment, "resolve_storage_limits", lambda user: (limits, "admin-1")), \
             patch.object(payment, "_get_app_conn", lambda: FakeConn(used)):
            return self.client.get("/api/v1/payments/storage/usage")

    def test_ts04_reports_used_against_the_limit(self):
        body = self.usage(int(4.2 * GB)).json()
        self.assertEqual(body["plan_name"], "Team")
        self.assertEqual(body["limit_bytes"], 5 * GB)
        self.assertEqual(body["used_bytes"], int(4.2 * GB))
        self.assertEqual(body["remaining_bytes"], 5 * GB - int(4.2 * GB))
        self.assertEqual(body["usage_percent"], 84.0)
        self.assertEqual((body["max_file_bytes"], body["max_batch_files"]), (50 * MB, 10))
        self.assertFalse(body["blocked"])

    def test_full_workspace_is_blocked_and_never_shows_negative_space(self):
        body = self.usage(6 * GB).json()
        self.assertEqual(body["remaining_bytes"], 0)
        self.assertEqual(body["usage_percent"], 100.0)
        self.assertTrue(body["blocked"])

    def test_empty_workspace(self):
        body = self.usage(0).json()
        self.assertEqual((body["used_bytes"], body["usage_percent"], body["blocked"]), (0, 0.0, False))

    def test_every_plan_carries_the_three_limits(self):
        for plan in payment.PLAN_QUOTAS.values():
            self.assertGreater(plan["storage_limit_bytes"], 0)
            self.assertGreater(plan["max_file_bytes"], 0)
            self.assertGreater(plan["max_batch_files"], 0)


if __name__ == "__main__":
    unittest.main()
