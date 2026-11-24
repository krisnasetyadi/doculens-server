"""MS-1180: the Lens Guide scope. Path and count logic runs on fakes; who sees
what runs against real SQL in a throwaway schema (testing_db), skipped when no
database is reachable. Never point DATABASE_URL at a shared database for it."""

import unittest
import uuid
from unittest.mock import patch
from urllib.parse import urlparse

from models import UserRecord
from services.lens_guide import scope as scope_module
from services.lens_guide.match import match_folders, resolve_folder_label
from services.lens_guide.scope import load_scope
from services.storage import supabase as storage
from tests.testing_db import IsolatedSchemaTestCase, _base_url


def user(role="user") -> UserRecord:
    uid = f"{role}-{uuid.uuid4().hex[:8]}"
    return UserRecord(user_id=uid, email=f"{uid}@example.com", role=role, is_active=True)


def fake_scope(folders, collections):
    """load_scope over fixed storage rows (who sees them is tested below)."""
    with patch.object(scope_module.supabase_storage, "list_folders_for_user", return_value=folders), \
            patch.object(scope_module.supabase_storage, "list_active_collections_for_user", return_value=collections):
        return load_scope(user())


class ScopePathTests(unittest.TestCase):
    def test_three_levels_and_root(self):
        result = fake_scope(
            [
                {"folder_id": "d", "name": "Delivery", "parent_folder_id": None},
                {"folder_id": "y", "name": "Proyek Klien Y", "parent_folder_id": "d"},
                {"folder_id": "b", "name": "Backend", "parent_folder_id": "y"},
            ],
            [
                {"collection_id": "c1", "title": "Arsitektur", "file_names": ["a.pdf", "b.pdf"], "folder_id": "b"},
                {"collection_id": "c2", "title": "Brief", "file_names": ["brief.pdf"], "folder_id": "y"},
                {"collection_id": "c3", "title": None, "file_names": ["cuti.pdf"], "folder_id": None},
            ],
        )
        folders = {f.folder_id: f for f in result.folders}
        self.assertEqual(folders["b"].path, ["Delivery", "Proyek Klien Y", "Backend"])
        self.assertEqual(folders["d"].path, ["Delivery"])
        # A folder counts the files of every folder below it.
        self.assertEqual((folders["b"].file_count, folders["y"].file_count, folders["d"].file_count), (2, 3, 3))

        collections = {c.collection_id: c for c in result.collections}
        self.assertEqual(collections["c1"].folder_path, ["Delivery", "Proyek Klien Y", "Backend"])
        self.assertEqual((collections["c3"].folder_id, collections["c3"].folder_path), (None, []))
        self.assertEqual(collections["c3"].title, "cuti.pdf")

    def test_folders_outside_the_scope_are_never_named(self):
        # "secret" is not in the user's folders: its child starts the path, and a
        # collection filed in it counts as being at the root.
        result = fake_scope(
            [{"folder_id": "child", "name": "Mine", "parent_folder_id": "secret"}],
            [
                {"collection_id": "c1", "title": "Doc", "file_names": ["x.pdf"], "folder_id": "secret"},
                {"collection_id": "c2", "title": "Doc 2", "file_names": ["y.pdf"], "folder_id": "child"},
            ],
        )
        self.assertEqual(result.folders[0].path, ["Mine"])
        self.assertIsNone(result.folders[0].parent_folder_id)
        self.assertEqual(result.folders[0].file_count, 1)
        self.assertEqual((result.collections[0].folder_id, result.collections[0].folder_path), (None, []))

    def test_same_named_folders_get_a_number_so_a_pick_names_one(self):
        # MS-1182: an admin sees their "A" and another user's "A" at the root.
        # Picking one sends its path as text; equal paths looped on "which one?".
        result = fake_scope(
            [
                {"folder_id": "new", "name": "A", "parent_folder_id": None, "owner_id": "u2", "created_at": "2026-11-02"},
                {"folder_id": "old", "name": "A ", "parent_folder_id": None, "owner_id": "u1", "created_at": "2026-11-01"},
                {"folder_id": "sub", "name": "Docs", "parent_folder_id": "new"},
                {"folder_id": "b", "name": "B", "parent_folder_id": None},
            ],
            [],
        )
        folders = {f.folder_id: f for f in result.folders}
        self.assertEqual(
            {k: f.path for k, f in folders.items()},
            {"old": ["A (1)"], "new": ["A (2)"], "sub": ["A (2)", "Docs"], "b": ["B"]},
        )
        self.assertEqual(folders["new"].name, "A")  # the stored name is untouched

        self.assertEqual({f.folder_id for f in match_folders("A", result)}, {"old", "new"})
        self.assertEqual([f.folder_id for f in match_folders("folder A (2)", result)], ["new"])
        self.assertEqual([f.folder_id for f in resolve_folder_label("A (2)", result)], ["new"])

    def test_a_cycle_does_not_hang(self):
        result = fake_scope(
            [
                {"folder_id": "a", "name": "A", "parent_folder_id": "b"},
                {"folder_id": "b", "name": "B", "parent_folder_id": "a"},
            ],
            [{"collection_id": "c1", "title": "Doc", "file_names": ["x.pdf"], "folder_id": "a"}],
        )
        self.assertEqual({f.folder_id: f.path for f in result.folders}, {"a": ["B", "A"], "b": ["A", "B"]})

    def test_a_folder_failure_is_not_an_empty_scope(self):
        # MS-1181: "no folder found" must not be said when the folders could not be read.
        doc = {"collection_id": "c1", "title": "Doc", "file_names": ["x.pdf"], "folder_id": None}
        with patch.object(scope_module.supabase_storage, "list_folders_for_user", side_effect=RuntimeError("db down")), \
                patch.object(scope_module.supabase_storage, "list_active_collections_for_user", return_value=[doc]):
            result = load_scope(user())
        self.assertFalse(result.folders_loaded)
        self.assertEqual((result.folders, [c.collection_id for c in result.collections]), ([], ["c1"]))
        self.assertTrue(fake_scope([], []).folders_loaded)

    def test_each_folder_keeps_its_owner(self):
        admin = user("admin")
        result = fake_scope(
            [
                {"folder_id": "mine", "name": "Proyek Klien Y", "parent_folder_id": None, "owner_id": admin.user_id},
                {"folder_id": "budi", "name": "Proyek Klien Y", "parent_folder_id": None, "owner_id": "u-budi"},
            ],
            [],
        )
        self.assertEqual({f.folder_id: f.owner_id for f in result.folders}, {"mine": admin.user_id, "budi": "u-budi"})


# A pooled database (the Supabase pooler in .env) ignores the search_path the
# throwaway schema relies on, so the rows land in the real public tables.
LOCAL_DB = urlparse(_base_url() or "").hostname in ("127.0.0.1", "localhost")


@unittest.skipUnless(LOCAL_DB, "needs DATABASE_URL on a local Postgres, never a shared one")
class ScopeAccessTests(IsolatedSchemaTestCase):
    def collection(self, owner, title, folder_id=None):
        cid = str(uuid.uuid4())
        self.assertTrue(storage.register_collection(cid, [f"{title}.pdf"], 1, title=title, owner_id=owner.user_id))
        if folder_id:
            self.assertTrue(storage.set_collection_folder(cid, folder_id))
        return cid

    def folder(self, owner, name, parent=None):
        return storage.create_folder(str(uuid.uuid4()), name, owner.user_id, parent)["folder_id"]

    def test_each_user_gets_only_their_own_and_admin_gets_all(self):
        alice, bob, admin = user(), user(), user("admin")
        self.sql("INSERT INTO users (user_id, email, password_hash, role, name) VALUES (%s, %s, 'x', 'user', 'Alice')",
                 (alice.user_id, alice.email))
        self.sql("INSERT INTO users (user_id, email, password_hash, role) VALUES (%s, %s, 'x', 'user')",
                 (bob.user_id, bob.email))
        delivery = self.folder(alice, "Delivery")
        project = self.folder(alice, "Proyek Klien Y", delivery)
        backend = self.folder(alice, "Backend", project)
        alice_doc = self.collection(alice, "Arsitektur", backend)
        alice_root = self.collection(alice, "Kebijakan Cuti")
        bob_folder = self.folder(bob, "Rahasia Bob")
        bob_doc = self.collection(bob, "Gaji", bob_folder)

        dropped = self.collection(alice, "Lama", project)
        self.assertTrue(storage.set_collection_status(dropped, False))
        chat = str(uuid.uuid4())
        self.assertTrue(storage.register_chat_collection(chat, "grup.txt", "whatsapp", 3, [], None, owner_id=alice.user_id))

        seen = {}
        for who in (alice, bob, admin):
            result = load_scope(who)
            seen[who.role if who is admin else who.user_id] = (
                {f.folder_id for f in result.folders},
                {c.collection_id for c in result.collections},
                result,
            )

        a_folders, a_docs, a_result = seen[alice.user_id]
        self.assertEqual(a_folders, {delivery, project, backend})
        self.assertEqual(a_docs, {alice_doc, alice_root})  # not inactive, not chat, not Bob's
        doc = next(c for c in a_result.collections if c.collection_id == alice_doc)
        self.assertEqual(doc.folder_path, ["Delivery", "Proyek Klien Y", "Backend"])
        self.assertEqual(next(c for c in a_result.collections if c.collection_id == alice_root).folder_path, [])
        self.assertEqual(next(f for f in a_result.folders if f.folder_id == delivery).file_count, 1)

        b_folders, b_docs, _ = seen[bob.user_id]
        self.assertEqual((b_folders, b_docs), ({bob_folder}, {bob_doc}))

        admin_folders, admin_docs, admin_result = seen["admin"]
        self.assertTrue({delivery, project, backend, bob_folder} <= admin_folders)
        self.assertTrue({alice_doc, alice_root, bob_doc} <= admin_docs)
        self.assertFalse({dropped, chat} & admin_docs)

        self.assertTrue(a_result.folders_loaded)
        owner = {f.folder_id: f.owner_id for f in admin_result.folders}
        self.assertEqual((owner[delivery], owner[bob_folder]), (alice.user_id, bob.user_id))
        # MS-1181: an owner's name for the folder cards, else their email.
        names = storage.get_user_names([alice.user_id, bob.user_id, "nobody"])
        self.assertEqual(names, {alice.user_id: "Alice", bob.user_id: bob.email})


if __name__ == "__main__":
    unittest.main()
