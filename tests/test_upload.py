"""MS-657: behaviour of document uploads (router/ + services/upload.py) against real SQL,
recorded before it is moved. Size limits and progress streaming have their own
tests (test_storage_limits.py, test_upload_progress.py); this covers persist
modes, remote imports and Drive folder listing. Remote servers, DNS, embeddings
and Supabase Storage are faked, so nothing leaves the machine. Runs in a
throwaway schema (testing_db); skipped when no database is reachable."""

import io
import os
import socket
import tempfile
import unittest
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from reportlab.pdfgen import canvas

from services.storage import supabase as storage
from services.storage import limits as storage_limits
from config import config
from models import UserRecord
from services.rag.processor import processor
from router import upload
from services import upload as upload_service
from services.auth import get_current_user
from tests.testing_db import IsolatedSchemaTestCase

_real_getaddrinfo = socket.getaddrinfo
_RealAsyncClient = httpx.AsyncClient


def fake_getaddrinfo(host, *args, **kwargs):
    """Hostnames resolve to a public address without DNS; literal IPs resolve as themselves."""
    if host.replace(".", "").isdigit():
        return _real_getaddrinfo(host, *args, **kwargs)
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("142.250.4.100", 0))]


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.0]


def pdf_bytes(text="Hello world, this is a test document with enough text to index.") -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    return buf.getvalue()


def drive_folder_html(files=(), folders=()):
    links = [f'<a href="https://drive.google.com/file/d/{fid}/view">{name}</a>' for fid, name in files]
    links += [f'<a href="/drive/folders/{fid}">{name}</a>' for fid, name in folders]
    return "\n".join(links)


class UploadTestCase(IsolatedSchemaTestCase):
    def setUp(self):
        storage_limits._reservations.clear()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(processor, "embeddings", FakeEmbeddings()))
        self.stack.enter_context(patch.object(storage, "is_enabled", lambda: False))
        self.stack.enter_context(patch("socket.getaddrinfo", fake_getaddrinfo))
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for name in ("upload_folder", "index_folder"):
            (directory / name).mkdir()
            self.stack.enter_context(patch.object(config, name, str(directory / name)))
        self.upload_dir = directory / "upload_folder"
        self.index_dir = directory / "index_folder"

        # url -> httpx.Response (or an exception to raise); every request is recorded
        self.remote: dict = {}
        self.requested: list[str] = []

        def handler(request):
            url = str(request.url)
            self.requested.append(url)
            for prefix, answer in self.remote.items():
                if url.startswith(prefix):
                    if isinstance(answer, Exception):
                        raise answer
                    return answer
            return httpx.Response(404)

        self.stack.enter_context(patch(
            "httpx.AsyncClient",
            lambda *a, **kw: _RealAsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)))

        self.user_id = f"admin-{uuid.uuid4().hex[:8]}"
        app = FastAPI()
        app.include_router(upload.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id=self.user_id, email=f"{self.user_id}@example.com", role="admin", is_active=True)
        self.http = TestClient(app)

    def serve_pdf(self, url, data=None, filename=None, content_type="application/pdf"):
        headers = {"content-type": content_type}
        if filename:
            headers["content-disposition"] = f'attachment; filename="{filename}"'
        self.remote[url] = httpx.Response(200, content=data if data is not None else pdf_bytes(), headers=headers)

    def collection_row(self, collection_id):
        rows = self.sql("SELECT * FROM collections WHERE collection_id = %s", (collection_id,))
        return rows[0] if rows else None

    def upload_files(self, *files, **params):
        return self.http.post("/api/v1/pdf-collections/upload", params=params,
                              files=[("files", (name, data, "application/octet-stream")) for name, data in files])


class FileUploadTest(UploadTestCase):
    def test_auto_mode_registers_the_collection_and_keeps_local_files(self):
        data = pdf_bytes()
        resp = self.upload_files(("report.pdf", data), ("notes.txt", b"plain text notes " * 10))
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual((body["status"], body["file_count"], body["file_names"]),
                         ("success", 2, ["report.pdf", "notes.txt"]))
        row = self.collection_row(body["collection_id"])
        self.assertEqual((row["owner_id"], row["file_names"], row["storage_paths"]), (self.user_id, ["report.pdf", "notes.txt"], []))
        self.assertEqual(row["size_bytes"], len(data) + len(b"plain text notes " * 10))
        self.assertTrue((self.upload_dir / body["collection_id"] / "report.pdf").exists())

    def test_local_mode_writes_nothing_to_the_database(self):
        body = self.upload_files(("report.pdf", pdf_bytes()), persist_mode="local").json()
        self.assertEqual(body["status"], "success")
        self.assertIsNone(self.collection_row(body["collection_id"]))

    def test_database_mode_needs_supabase_storage_and_leaves_nothing(self):
        resp = self.upload_files(("report.pdf", pdf_bytes()), persist_mode="database")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(self.sql("SELECT * FROM collections WHERE owner_id = %s", (self.user_id,)), [])
        self.assertEqual(list(self.upload_dir.iterdir()), [])
        self.assertEqual(list(self.index_dir.iterdir()), [])

    def test_unsupported_files_only_are_refused(self):
        resp = self.upload_files(("tool.exe", b"MZ"), ("image.png", b"\x89PNG"))
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_a_file_without_text_creates_no_collection(self):
        blank = io.BytesIO()
        canvas.Canvas(blank).save()
        resp = self.upload_files(("blank.pdf", blank.getvalue()))
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No readable text", resp.json()["detail"])
        self.assertEqual(list(self.upload_dir.iterdir()), [])


class UploadFromUrlTest(UploadTestCase):
    def post(self, url, title=None, **params):
        return self.http.post("/api/v1/pdf-collections/upload-from-url", params=params,
                              json={"url": url, **({"title": title} if title is not None else {})})

    def test_a_public_pdf_becomes_a_titled_collection(self):
        self.serve_pdf("https://files.example.com/download", filename="Annual Report.pdf")
        resp = self.post("  https://files.example.com/download  ", title="  Annual\n  report  ")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual((body["file_names"], body["title"]), (["Annual Report.pdf"], "Annual report"))
        row = self.collection_row(body["collection_id"])
        self.assertEqual((row["title"], row["file_names"], row["owner_id"]),
                         ("Annual report", ["Annual Report.pdf"], self.user_id))

    def test_drive_file_links_are_fetched_through_the_download_url(self):
        self.serve_pdf("https://drive.google.com/uc?export=download&id=DRIVEFILE")
        resp = self.post("https://drive.google.com/file/d/DRIVEFILE/view?usp=sharing")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(self.requested, ["https://drive.google.com/uc?export=download&id=DRIVEFILE"])
        # Without a Content-Disposition name the URL path names the file (real Drive sends one).
        self.assertEqual(resp.json()["file_names"], ["uc.pdf"])

    def test_a_page_that_is_not_a_pdf_is_refused(self):
        self.remote["https://example.com/page"] = httpx.Response(200, text="<html></html>", headers={"content-type": "text/html"})
        resp = self.post("https://example.com/page")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("did not return a PDF", resp.json()["detail"])
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_unreachable_and_private_urls_are_refused(self):
        self.remote["https://example.com/gone.pdf"] = httpx.ConnectError("down")
        unreachable = self.post("https://example.com/gone.pdf")
        self.assertEqual(unreachable.status_code, 400)
        self.assertIn("Unable to download", unreachable.json()["detail"])
        self.assertEqual(self.post("http://127.0.0.1/a.pdf").status_code, 400)
        self.assertEqual(self.post("ftp://example.com/a.pdf").status_code, 400)
        self.assertEqual(self.post("   ").status_code, 400)
        self.assertEqual(list(self.upload_dir.iterdir()), [])
        self.assertNotIn("http://127.0.0.1/a.pdf", self.requested)


class UploadFromUrlsTest(UploadTestCase):
    def post(self, urls, title=None):
        return self.http.post("/api/v1/pdf-collections/upload-from-urls", json={"urls": urls, "title": title})

    def test_good_files_are_kept_when_others_fail(self):
        self.serve_pdf("https://example.com/a.pdf")
        resp = self.post(["https://example.com/a.pdf", "https://example.com/missing.pdf", "  "], title="Mixed")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual((body["file_count"], body["file_names"], body["title"]), (1, ["a.pdf"], "Mixed"))
        self.assertEqual(self.collection_row(body["collection_id"])["file_names"], ["a.pdf"])

    def test_all_failing_reports_the_first_failure(self):
        self.remote["https://example.com/page"] = httpx.Response(200, text="hi", headers={"content-type": "text/html"})
        resp = self.post(["https://example.com/page", "https://example.com/missing.pdf"])
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(resp.json()["detail"].startswith("No valid PDF files imported: https://example.com/page ("))
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_empty_and_oversized_batches_are_refused(self):
        self.assertEqual(self.post(["", "  "]).status_code, 400)
        too_many = [f"https://example.com/{i}.pdf" for i in range(100)]
        resp = self.post(too_many)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("files at a time", resp.json()["detail"])
        self.assertEqual(self.requested, [])


class DriveFolderItemsTest(UploadTestCase):
    def list_items(self, url, **extra):
        return self.http.post("/api/v1/pdf-collections/drive/folder-items", json={"url": url, **extra})

    def serve_folder(self, folder_id, **content):
        self.remote[f"https://drive.google.com/embeddedfolderview?id={folder_id}"] = httpx.Response(
            200, text=drive_folder_html(**content))

    def test_subfolders_are_walked_and_duplicates_dropped(self):
        self.serve_folder("ROOT", files=[("F1", "a.pdf"), ("F1", "a copy")], folders=[("SUB", "Sub")])
        self.serve_folder("SUB", files=[("F2", "b.pdf")], folders=[("ROOT", "back to root")])
        body = self.list_items("https://drive.google.com/drive/folders/ROOT?usp=sharing").json()
        self.assertEqual(body["folder_id"], "ROOT")
        self.assertEqual([(f["id"], f["name"]) for f in body["files"]], [("F1", "a.pdf"), ("F2", "b.pdf")])
        self.assertEqual([f["id"] for f in body["folders"]], ["SUB", "ROOT"])
        self.assertEqual(body["count"], 2)

    def test_depth_limits_and_non_recursive_listing(self):
        self.serve_folder("ROOT", files=[("F1", "a.pdf")], folders=[("SUB", "Sub")])
        self.serve_folder("SUB", files=[("F2", "b.pdf")])
        for extra in ({"recursive": False}, {"max_depth": 0}):
            body = self.list_items("https://drive.google.com/drive/folders/ROOT", **extra).json()
            self.assertEqual([f["id"] for f in body["files"]], ["F1"])
            self.assertEqual([f["id"] for f in body["folders"]], ["SUB"])

    def test_invalid_and_unreachable_folders(self):
        self.assertEqual(self.list_items("https://example.com/drive/folders/X").status_code, 400)
        self.assertEqual(self.list_items("   ").status_code, 400)
        resp = self.list_items("https://drive.google.com/drive/folders/PRIVATE")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Ensure the folder is public", resp.json()["detail"])


class RegistrationFailureTest(UploadTestCase):
    """With a database configured the collection list is read from it, so an
    upload whose row could not be saved must not be reported as a success."""

    def setUp(self):
        super().setUp()
        failing = patch.object(storage, "register_collection", return_value=False)
        failing.start()
        self.addCleanup(failing.stop)

    def assert_refused_and_nothing_left(self, resp):
        self.assertEqual(resp.status_code, 500, resp.text)
        self.assertEqual(resp.json()["detail"], "Failed to save the collection")
        self.assertEqual(list(self.upload_dir.iterdir()), [])
        self.assertEqual(list(self.index_dir.iterdir()), [])
        self.assertEqual(storage_limits._reservations.get(self.user_id, {}), {})

    def test_file_upload(self):
        self.assert_refused_and_nothing_left(self.upload_files(("report.pdf", pdf_bytes())))

    def test_upload_from_url(self):
        self.serve_pdf("https://example.com/a.pdf")
        self.assert_refused_and_nothing_left(self.http.post(
            "/api/v1/pdf-collections/upload-from-url", json={"url": "https://example.com/a.pdf"}))

    def test_upload_from_urls(self):
        self.serve_pdf("https://example.com/a.pdf")
        self.assert_refused_and_nothing_left(self.http.post(
            "/api/v1/pdf-collections/upload-from-urls", json={"urls": ["https://example.com/a.pdf"]}))


class RegisterCollectionTest(unittest.TestCase):
    def test_the_connection_is_returned_when_the_insert_fails(self):
        class BrokenConn:
            closed = False

            def cursor(self):
                raise RuntimeError("insert failed")

            def close(self):
                self.closed = True

        conn = BrokenConn()
        with patch.object(storage.db, "get_conn", lambda source="": conn):
            self.assertFalse(storage.register_collection("c1", ["a.pdf"], 1))
        self.assertTrue(conn.closed)


class FilenameTest(unittest.TestCase):
    """Pure helpers; no database needed."""

    def test_pdf_names(self):
        self.assertEqual(upload_service._safe_pdf_filename("../../etc/passwd", "fb"), "passwd.pdf")
        self.assertEqual(upload_service._safe_pdf_filename("Report (final).PDF", "fb"), "Report _final_.PDF")
        self.assertEqual(upload_service._safe_pdf_filename("", "fb"), "fb.pdf")

    def test_upload_names_keep_their_extension(self):
        self.assertEqual(upload_service._safe_upload_filename("会社概要.docx", "fb"), "fb.docx")
        self.assertEqual(upload_service._safe_upload_filename("data.csv", "fb"), "data.csv")
        self.assertEqual(upload_service._safe_upload_filename("noext", "fb"), "noext.pdf")

    def test_titles_and_headers(self):
        self.assertIsNone(upload_service._normalize_title("   "))
        self.assertEqual(len(upload_service._normalize_title("x" * 200)), 150)
        self.assertEqual(upload_service._extract_filename_from_headers(
            httpx.Headers({"content-disposition": "attachment; filename*=UTF-8''r%C3%A9sum%C3%A9.pdf"})), "résumé.pdf")
        self.assertIsNone(upload_service._extract_filename_from_headers(httpx.Headers({})))


if __name__ == "__main__":
    unittest.main()
