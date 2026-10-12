"""MS-657: behaviour of source folders (router/ + services/folders.py) against real SQL,
recorded before it is moved: creating (top level and nested), listing per
owner, renaming and moving, and deleting (children move up a level). The
storage-level hierarchy rules have their own tests in test_nested_folders.py.
Runs in a throwaway schema (testing_db); skipped when no database is reachable."""

import os
import uuid

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import storage
from models import UserRecord
from router import folders
from services.auth import get_current_user
from testing_db import IsolatedSchemaTestCase


def user(role="user") -> UserRecord:
    uid = f"{role}-{uuid.uuid4().hex[:8]}"
    return UserRecord(user_id=uid, email=f"{uid}@example.com", role=role, is_active=True)


class FoldersTest(IsolatedSchemaTestCase):
    def client(self, who):
        app = FastAPI()
        app.include_router(folders.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: who
        return TestClient(app)

    def create(self, who, name, parent=None):
        return self.client(who).post("/api/v1/source-folders", json={"name": name, "parent_folder_id": parent})

    def test_create_nested_and_list_per_owner(self):
        owner, other, admin = user(), user(), user("admin")
        top = self.create(owner, "  Contracts  ")
        self.assertEqual(top.status_code, 201, top.text)
        self.assertEqual((top.json()["name"], top.json()["owner_id"], top.json()["parent_folder_id"]),
                         ("Contracts", owner.user_id, None))
        child = self.create(owner, "2026", top.json()["folder_id"]).json()
        self.assertEqual(child["parent_folder_id"], top.json()["folder_id"])
        theirs = self.create(other, "Theirs").json()["folder_id"]

        self.assertEqual(self.create(owner, "   ").status_code, 400)
        self.assertEqual(self.create(owner, "x", theirs).status_code, 404)        # not their parent
        self.assertEqual(self.create(owner, "x", "no-such-folder").status_code, 404)

        own_ids = {f["folder_id"] for f in self.client(owner).get("/api/v1/source-folders").json()}
        self.assertEqual(own_ids, {top.json()["folder_id"], child["folder_id"]})
        admin_ids = {f["folder_id"] for f in self.client(admin).get("/api/v1/source-folders").json()}
        self.assertTrue(own_ids | {theirs} <= admin_ids)

    def test_rename_and_move(self):
        owner, other, admin = user(), user(), user("admin")
        a = self.create(owner, "A").json()["folder_id"]
        b = self.create(owner, "B").json()["folder_id"]
        child = self.create(owner, "Child", a).json()["folder_id"]
        put = lambda who, folder, body: self.client(who).put(f"/api/v1/source-folders/{folder}", json=body)

        renamed = put(owner, child, {"name": " Kid "}).json()
        self.assertEqual((renamed["name"], renamed["parent_folder_id"]), ("Kid", a))  # parent kept when not sent
        self.assertEqual(put(owner, child, {"name": "Kid", "parent_folder_id": b}).json()["parent_folder_id"], b)
        self.assertIsNone(put(owner, child, {"name": "Kid", "parent_folder_id": None}).json()["parent_folder_id"])
        self.assertEqual(put(owner, a, {"name": "A", "parent_folder_id": a}).status_code, 400)  # into itself

        self.assertEqual(put(owner, a, {"name": "  "}).status_code, 400)
        self.assertEqual(put(other, a, {"name": "x"}).status_code, 404)
        self.assertEqual(put(admin, a, {"name": "x"}).status_code, 403)  # admins see it but do not own it
        self.assertEqual(put(owner, "nope", {"name": "x"}).status_code, 404)

    def test_delete_moves_children_up(self):
        owner, admin = user(), user("admin")
        top = self.create(owner, "Top").json()["folder_id"]
        middle = self.create(owner, "Middle", top).json()["folder_id"]
        child = self.create(owner, "Child", middle).json()["folder_id"]
        cid = str(uuid.uuid4())
        storage.register_collection(cid, ["a.pdf"], 1, owner_id=owner.user_id)
        storage.set_collection_folder(cid, middle)

        self.assertEqual(self.client(admin).delete(f"/api/v1/source-folders/{middle}").status_code, 403)
        self.assertEqual(self.client(user()).delete(f"/api/v1/source-folders/{middle}").status_code, 404)
        self.assertEqual(self.client(owner).delete(f"/api/v1/source-folders/{middle}").json(),
                         {"folder_id": middle, "deleted": True})
        self.assertEqual(storage.get_folder(child)["parent_folder_id"], top)
        self.assertEqual(storage.get_collection(cid)["folder_id"], top)
        self.assertEqual(self.client(owner).delete(f"/api/v1/source-folders/{middle}").status_code, 404)


if __name__ == "__main__":
    import unittest
    unittest.main()
