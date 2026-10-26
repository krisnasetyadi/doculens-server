"""MS-657: behaviour of PDF collection management (router/ + services/collections.py)
against real SQL, recorded before it is moved: listing, activating, moving to
a folder, deleting, serving files, the text viewer and the file list.
Supabase Storage is off, so files come from a temporary local folder. Runs in
a throwaway schema (testing_db); skipped when no database is reachable."""

import os
import tempfile
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.storage import supabase as storage
from config import config
from models import UserRecord
from services.rag.processor import processor
from router import collections
from services.auth import get_current_user
from tests.testing_db import IsolatedSchemaTestCase


def user(role="user") -> UserRecord:
    uid = f"{role}-{uuid.uuid4().hex[:8]}"
    return UserRecord(user_id=uid, email=f"{uid}@example.com", role=role, is_active=True)


class CollectionsTestCase(IsolatedSchemaTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(storage, "is_enabled", lambda: False))
        # Some paths (S3 scan, delete cleanup) use the client directly, not is_enabled().
        self.stack.enter_context(patch.object(storage, "_s3_client", lambda: None))
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for name in ("upload_folder", "index_folder"):
            (directory / name).mkdir()
            self.stack.enter_context(patch.object(config, name, str(directory / name)))
        self.upload_dir = directory / "upload_folder"
        self.index_dir = directory / "index_folder"

    def client(self, who):
        app = FastAPI()
        app.include_router(collections.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: who
        return TestClient(app)

    def collection(self, owner, files=("a.pdf",), title="Docs", on_disk=True, register=True):
        """A collection owned by `owner`, registered in the DB and/or present on disk."""
        cid = str(uuid.uuid4())
        if register:
            self.assertTrue(storage.register_collection(cid, list(files), 3, title=title, owner_id=owner.user_id))
        if on_disk:
            (self.upload_dir / cid).mkdir()
            for name in files:
                (self.upload_dir / cid / name).write_bytes(b"%PDF-1.4 test" if name.endswith(".pdf") else b"line\n")
            (self.index_dir / cid).mkdir()
            (self.index_dir / cid / "index.faiss").write_bytes(b"x")
        return cid


class EmptyDatabaseListTest(CollectionsTestCase):
    """Own class, so the collections table starts empty."""

    def test_admins_fall_back_to_local_disk_and_users_see_nothing(self):
        cid = self.collection(user(), files=("a.pdf", "notes.txt", "image.png"), register=False)
        self.assertEqual(self.client(user()).get("/api/v1/pdf-collections").json(), [])
        listed = self.client(user("admin")).get("/api/v1/pdf-collections").json()
        self.assertEqual([c["collection_id"] for c in listed], [cid])
        self.assertEqual((sorted(listed[0]["file_names"]), listed[0]["document_count"], listed[0]["title"]),
                         (["a.pdf", "notes.txt"], 2, None))


class ListTest(CollectionsTestCase):
    def test_users_see_their_own_and_admins_see_all(self):
        alice, bob = user(), user()
        a = self.collection(alice, files=("a.pdf", "b.pdf"), title="Alice docs")
        b = self.collection(bob)
        alice_view = self.client(alice).get("/api/v1/pdf-collections").json()
        self.assertEqual([c["collection_id"] for c in alice_view], [a])
        self.assertEqual((alice_view[0]["document_count"], alice_view[0]["title"], alice_view[0]["status"],
                          alice_view[0]["owner_id"], alice_view[0]["folder_id"]),
                         (2, "Alice docs", "active", alice.user_id, None))
        admin_ids = {c["collection_id"] for c in self.client(user("admin")).get("/api/v1/pdf-collections").json()}
        self.assertTrue({a, b} <= admin_ids)


class ActivateAndMoveTest(CollectionsTestCase):
    def test_activate(self):
        owner = user()
        cid = self.collection(owner)
        resp = self.client(owner).post("/api/v1/pdf-collections/activate", json={"collection_id": cid, "active": False})
        self.assertEqual(resp.json(), {"status": "success", "collection_id": cid, "active": False})
        self.assertEqual(storage.get_collection(cid)["status"], "inactive")
        self.assertEqual(self.client(user()).post("/api/v1/pdf-collections/activate",
                                                  json={"collection_id": cid, "active": True}).status_code, 403)
        self.assertEqual(self.client(owner).post("/api/v1/pdf-collections/activate",
                                                 json={"collection_id": "nope", "active": True}).status_code, 404)

    def test_deactivated_collections_are_not_chat_sources(self):
        """MS-665: chat searches and names only the collections left active."""
        owner = user()
        kept = self.collection(owner, title="Report")
        dropped = self.collection(owner, title="Attendance Mei")
        self.client(owner).post("/api/v1/pdf-collections/activate", json={"collection_id": dropped, "active": False})
        for is_admin in (False, True):
            ids = [cid for cid, _ in storage.list_active_collection_titles_for_user(owner.user_id, is_admin)]
            self.assertIn(kept, ids)
            self.assertNotIn(dropped, ids)

    def test_move_to_folder(self):
        owner, stranger = user(), user()
        cid = self.collection(owner)
        mine = storage.create_folder(str(uuid.uuid4()), "Mine", owner.user_id)["folder_id"]
        theirs = storage.create_folder(str(uuid.uuid4()), "Theirs", stranger.user_id)["folder_id"]
        move = lambda who, folder, target=cid: self.client(who).post(
            "/api/v1/pdf-collections/move-to-folder", json={"collection_id": target, "folder_id": folder})

        self.assertEqual(move(owner, mine).json(), {"status": "success", "collection_id": cid, "folder_id": mine})
        self.assertEqual(storage.get_collection(cid)["folder_id"], mine)
        self.assertEqual(move(owner, theirs).status_code, 404)
        self.assertEqual(move(owner, "no-such-folder").status_code, 404)
        self.assertEqual(move(stranger, theirs).status_code, 403)
        self.assertEqual(move(owner, None).json()["folder_id"], None)
        self.assertIsNone(storage.get_collection(cid)["folder_id"])
        self.assertEqual(move(owner, None, "nope").status_code, 404)
        self.assertEqual(move(user("admin"), theirs).status_code, 200)  # admins may use any folder

    def test_flag_as_reference_framework(self):
        """MS-848: the flag is saved, listed, and only the owner (or an admin) can set it."""
        owner = user()
        cid = self.collection(owner)
        flag = lambda who, value, target=cid: self.client(who).post(
            "/api/v1/pdf-collections/reference-framework",
            json={"collection_id": target, "is_reference_framework": value})
        listed = lambda: next(c for c in self.client(owner).get("/api/v1/pdf-collections").json()
                              if c["collection_id"] == cid)

        self.assertFalse(listed()["is_reference_framework"])
        self.assertEqual(flag(owner, True).json(),
                         {"status": "success", "collection_id": cid, "is_reference_framework": True})
        self.assertTrue(listed()["is_reference_framework"])
        self.assertEqual(flag(user(), False).status_code, 403)
        self.assertEqual(flag(owner, True, "nope").status_code, 404)
        self.assertEqual(flag(user("admin"), False).status_code, 200)
        self.assertFalse(listed()["is_reference_framework"])
        with patch.object(storage, "has_database", lambda: False):
            self.assertEqual(flag(owner, True).status_code, 503)


class DeleteTest(CollectionsTestCase):
    def test_owner_deletes_row_files_and_cache(self):
        owner = user()
        cid = self.collection(owner)
        with patch.object(processor, "invalidate_cache") as invalidate:
            self.assertEqual(self.client(user()).delete(f"/api/v1/pdf-collections/{cid}").status_code, 403)
            resp = self.client(owner).delete(f"/api/v1/pdf-collections/{cid}")
        self.assertEqual(resp.json(), {"status": "success", "message": "Collection deleted"})
        invalidate.assert_called_once_with(cid)
        self.assertIsNone(storage.get_collection(cid))
        self.assertFalse((self.upload_dir / cid).exists() or (self.index_dir / cid).exists())

    def test_unattributed_collections_are_admin_only(self):
        cid = self.collection(user(), register=False)
        self.assertEqual(self.client(user()).delete(f"/api/v1/pdf-collections/{cid}").status_code, 403)
        self.assertEqual(self.client(user("admin")).delete(f"/api/v1/pdf-collections/{cid}").status_code, 200)
        self.assertFalse((self.upload_dir / cid).exists())

    def test_deleting_what_is_already_gone_is_a_404(self):
        admin = self.client(user("admin"))
        cid = self.collection(user(), on_disk=False)
        self.assertEqual(admin.delete(f"/api/v1/pdf-collections/{cid}").status_code, 200)
        self.assertEqual(admin.delete(f"/api/v1/pdf-collections/{cid}").status_code, 404)
        self.assertEqual(admin.delete("/api/v1/pdf-collections/never-existed").status_code, 404)


class ServeFileTest(CollectionsTestCase):
    def test_owner_gets_the_file_with_its_real_type(self):
        owner = user()
        cid = self.collection(owner, files=("a.pdf", "Laporan é.docx"))
        pdf = self.client(owner).get(f"/api/v1/files/{cid}/a.pdf")
        self.assertEqual((pdf.status_code, pdf.content), (200, b"%PDF-1.4 test"))
        self.assertEqual(pdf.headers["content-type"], "application/pdf")
        self.assertTrue(pdf.headers["content-disposition"].startswith("inline"))
        docx = self.client(owner).get(f"/api/v1/files/{cid}/Laporan%20%C3%A9.docx")
        self.assertEqual(docx.status_code, 200)
        self.assertTrue(docx.headers["content-disposition"].startswith("attachment"))
        self.assertIn("filename*=utf-8''Laporan%20%C3%A9.docx", docx.headers["content-disposition"])

    def test_refusals(self):
        owner = user()
        cid = self.collection(owner, files=("a.pdf",))
        get = lambda who, name, target=cid: self.client(who).get(f"/api/v1/files/{target}/{name}")
        self.assertEqual(get(user(), "a.pdf").status_code, 403)
        self.assertEqual(get(owner, "..%2F..%2Fsecret.pdf").status_code, 400)
        self.assertEqual(get(owner, "a.exe").status_code, 400)
        missing = get(owner, "b.pdf")
        self.assertEqual((missing.status_code, missing.json()["detail"]), (404, "File 'b.pdf' not found. Available: a.pdf"))
        self.assertEqual(get(user("admin"), "a.pdf", "no-such-collection").status_code, 404)


class TextContentTest(CollectionsTestCase):
    def test_lines_are_paged(self):
        owner = user()
        cid = self.collection(owner, files=("notes.txt", "a.pdf"))
        (self.upload_dir / cid / "notes.txt").write_text("\n".join(f"line {i}" for i in range(1, 8)), encoding="utf-8")
        url = f"/api/v1/pdf-collections/{cid}/text-content"
        body = self.client(owner).get(url, params={"file_name": "notes.txt", "offset": 5, "limit": 5}).json()
        self.assertEqual((body["total_lines"], body["offset"], body["has_more"]), (7, 5, False))
        self.assertEqual(body["lines"], [{"line_number": 6, "content": "line 6"}, {"line_number": 7, "content": "line 7"}])
        first = self.client(owner).get(url, params={"file_name": "notes.txt", "limit": 2}).json()
        self.assertEqual((len(first["lines"]), first["has_more"]), (2, True))

        self.assertEqual(self.client(owner).get(url, params={"file_name": "a.pdf"}).status_code, 400)
        self.assertEqual(self.client(owner).get(url, params={"file_name": "../x.txt"}).status_code, 400)
        self.assertEqual(self.client(owner).get(url, params={"file_name": "gone.txt"}).status_code, 404)
        self.assertEqual(self.client(user()).get(url, params={"file_name": "notes.txt"}).status_code, 403)
        self.assertEqual(self.client(owner).get(url, params={"file_name": "notes.txt", "limit": 501}).status_code, 422)


class FileListTest(CollectionsTestCase):
    def test_registered_collections_list_their_stored_names(self):
        owner = user()
        cid = self.collection(owner, files=("a b.pdf", "c.txt"), on_disk=False)
        body = self.client(owner).get(f"/api/v1/collection/{cid}/files").json()
        self.assertEqual((body["collection_id"], body["file_count"]), (cid, 2))
        self.assertEqual([(f["file_name"], f["url"]) for f in body["files"]],
                         [("a b.pdf", f"/api/v1/files/{cid}/a%20b.pdf"), ("c.txt", f"/api/v1/files/{cid}/c.txt")])
        self.assertEqual(self.client(user()).get(f"/api/v1/collection/{cid}/files").status_code, 403)

    def test_unregistered_collections_are_read_from_disk(self):
        cid = self.collection(user(), files=("a.pdf", "image.png"), register=False)
        admin = self.client(user("admin"))
        body = admin.get(f"/api/v1/collection/{cid}/files").json()
        self.assertEqual([(f["file_name"], f["size_bytes"]) for f in body["files"]], [("a.pdf", len(b"%PDF-1.4 test"))])
        self.assertEqual(admin.get("/api/v1/collection/nope/files").status_code, 404)


if __name__ == "__main__":
    import unittest
    unittest.main()
