"""MS-1181: finding and confirming the folder a user names, end to end through
POST /agnostic/query with a fake model. The scope rows stand in for what the
access rule returns (tested in test_lens_guide_scope)."""

import json
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("JWT_SECRET", "conversation-local-test-secret-not-for-production")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastapi import HTTPException

from models import DriveFolderItem, DriveFolderItemsResponse, UserRecord
from router import agnostic
from services import agnostic as agnostic_service
from services.auth import get_current_user
from services.billing import inflight
from services.lens_guide import drive_import
from services.lens_guide.drive_import import find_drive_link
from services.lens_guide.reply import requested_name
from services.rag.processor import processor
from tests.fakes import fake_llm

FOLDERS = [
    {"folder_id": "d", "name": "Delivery", "parent_folder_id": None, "owner_id": "u1"},
    {"folder_id": "y", "name": "Proyek Klien Y", "parent_folder_id": "d", "owner_id": "u1"},
    {"folder_id": "yb", "name": "Backend", "parent_folder_id": "y", "owner_id": "u1"},
    {"folder_id": "e", "name": "Kosong", "parent_folder_id": "d", "owner_id": "u1"},
    {"folder_id": "by", "name": "Proyek Klien Y", "parent_folder_id": None, "owner_id": "u-budi"},
]
COLLECTIONS = [
    {"collection_id": "c1", "title": "Arsitektur", "file_names": ["a.pdf", "b.pdf"], "folder_id": "yb"},
    {"collection_id": "c2", "title": "Brief", "file_names": ["brief.pdf"], "folder_id": "y"},
    {"collection_id": "c3", "title": "Cuti", "file_names": ["cuti.pdf"], "folder_id": None},
    {"collection_id": "c4", "title": "Catatan Budi", "file_names": ["budi.pdf"], "folder_id": "by"},
]


class LensGuideEndpoint(unittest.TestCase):
    """POST /agnostic/query as admin u1 over FOLDERS and COLLECTIONS. No tests of its own."""

    def setUp(self):
        inflight._in_flight_workspace_tokens.clear()
        inflight._in_flight_user_tokens.clear()
        app = FastAPI()
        app.include_router(agnostic.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="u1", email="u1@example.com", role="admin", is_active=True
        )
        self.client = TestClient(app)
        storage = agnostic_service.supabase_storage
        self.folders = patch.object(storage, "list_folders_for_user", return_value=FOLDERS).start()
        for target, attr, kwargs in [
            (storage, "list_active_collections_for_user", {"return_value": COLLECTIONS}),
            (storage, "get_user_names", {"return_value": {"u-budi": "Budi"}}),
            (agnostic_service, "resolve_workspace_id", {"return_value": "w1"}),
            (agnostic_service, "enforce_rate_limit", {}),
            (agnostic_service, "enforce_plan_limit", {"return_value": None}),
            (agnostic_service, "enforce_member_allocation", {"return_value": None}),
            (agnostic_service, "get_usage_snapshot", {"return_value": None}),
            (agnostic_service, "log_token_usage", {}),
        ]:
            patch.object(target, attr, **kwargs).start()
        self.hybrid_search = patch.object(processor, "hybrid_search", return_value={}).start()
        self.addCleanup(patch.stopall)

    def ask(self, question, llm, **body):
        with patch.object(processor, "get_llm", return_value=(llm, "gemini/gemini-test")), \
                patch.object(processor, "generate_hybrid_answer", return_value=("Jawaban", "gemini/x", {})):
            response = self.client.post("/api/v1/agnostic/query", json={"question": question, **body})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def find(self, name):
        return fake_llm(tool_name="find_folder", tool_args={"name": name})


class FindFolderTests(LensGuideEndpoint):
    def test_one_match_is_confirmed_with_location_and_counts(self):
        data = self.ask("aku baru masuk Delivery Klien Y", self.find("Delivery Klien Y"))
        self.assertEqual(data["guide"]["folder"], {
            "path": ["Delivery", "Proyek Klien Y"], "file_count": 3, "subfolder_count": 1, "owner": None,
        })
        # The path stays in the text (a typed "ya" still carries the scope); the counts are on the card only.
        self.assertEqual(
            data["answer"], "Saya menemukan folder **Delivery › Proyek Klien Y**. Benar ini folder yang kamu maksud?",
        )
        self.hybrid_search.assert_not_called()

    def test_the_streamed_reply_carries_the_guide_in_done(self):
        with patch.object(processor, "get_llm", return_value=(self.find("Delivery Klien Y"), "gemini/gemini-test")):
            response = self.client.post("/api/v1/agnostic/query?stream=true", json={"question": "aku baru masuk Klien Y"})
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([e["type"] for e in events], ["done"])
        self.assertEqual(events[0]["result"]["guide"]["folder"]["path"], ["Delivery", "Proyek Klien Y"])

    def test_several_matches_ask_which_with_full_location_and_owner(self):
        data = self.ask("aku baru masuk Proyek Klien Y", self.find("Proyek Klien Y"))
        folders = data["guide"]["folders"]
        self.assertEqual([(f["path"], f["owner"]) for f in folders], [
            (["Delivery", "Proyek Klien Y"], None), (["Proyek Klien Y"], "Budi"),
        ])
        self.assertIsNone(data["guide"]["folder"])
        # The choices are on the cards only; the one clicked sends its path as the reply.
        self.assertEqual(
            data["answer"], 'Ada 2 folder yang cocok dengan "Proyek Klien Y" dan bisa kamu buka. Yang mana yang kamu maksud?',
        )

    def test_no_match_asks_where_the_documents_are(self):
        data = self.ask("aku baru masuk Proyek Klien Z", self.find("Proyek Klien Z"))
        self.assertTrue(data["guide"]["missing_source"])
        # Only the user's own folders to pick from, even for an admin.
        self.assertEqual([f["owner"] for f in data["guide"]["folders"]], [None] * 4)
        self.assertTrue(data["answer"].startswith('Saya belum menemukan folder "Proyek Klien Z"'))
        self.assertIn("link folder Google Drive publiknya di sini", data["answer"])
        # The text names no folder: only the card lists them.
        self.assertNotIn("Delivery", data["answer"])

    def test_a_confirmed_folder_limits_the_search_to_it_and_its_subfolders(self):
        llm = fake_llm(tool_args={"query": "dokumen onboarding", "folder": "Delivery › Proyek Klien Y", "task": "answer"})
        data = self.ask("ya, apa yang harus kubaca dulu?", llm)
        self.assertEqual(data["answer"], "Jawaban")
        self.assertEqual(sorted(self.hybrid_search.call_args[0][1]), ["c1", "c2"])

    def test_a_confirmed_folder_without_documents_is_not_widened(self):
        llm = fake_llm(tool_args={"query": "dokumen onboarding", "folder": "Delivery › Kosong", "task": "answer"})
        data = self.ask("ya", llm)
        self.assertIn("belum berisi dokumen aktif", data["answer"])
        self.hybrid_search.assert_not_called()

    def test_naming_a_folder_goes_to_the_router_even_when_the_rules_would_search(self):
        llm = self.find("Proyek Klien Y")
        data = self.ask("ringkas dokumen Proyek Klien Y", llm)
        llm.invoke.assert_called_once()
        self.assertIsNotNone(data["guide"]["folders"])

        llm = fake_llm("tidak dipakai")
        self.ask("apa kata dokumen Cuti soal lembur?", llm)  # no folder named: the rules search as before
        llm.invoke.assert_not_called()
        self.hybrid_search.assert_called()

    def test_folders_that_could_not_be_read_turn_folder_finding_off(self):
        self.folders.side_effect = RuntimeError("db down")
        llm = fake_llm("Halo! Ada yang bisa kubantu?")
        data = self.ask("aku baru masuk Proyek Klien Y", llm)
        self.assertEqual([t["name"] for t in llm.bind_tools.call_args[0][0]], ["search_documents"])
        self.assertIsNone(data["guide"])



def listing(*names):
    files = [DriveFolderItem(id=f"f{i}", name=n, url=f"https://drive.google.com/file/d/f{i}/view", item_type="file")
             for i, n in enumerate(names)]
    return DriveFolderItemsResponse(folder_id="1kZ", files=files, folders=[], count=len(files))


class DriveImportOfferTests(LensGuideEndpoint):
    """A Drive link pasted in the chat (mockup "Belum ada sumber")."""

    NOT_FOUND = [
        {"role": "user", "content": "aku baru masuk Proyek Klien Z"},
        {"role": "assistant", "content": 'Saya belum menemukan folder "Proyek Klien Z" di dokumen yang bisa kamu buka. ...'},
    ]

    def offer(self, question, listed=None, error=None, memory=None):
        llm = fake_llm("tidak dipakai")
        mock = AsyncMock(return_value=listed, side_effect=error)
        with patch.object(drive_import, "list_drive_folder_items", mock):
            data = self.ask(question, llm, memory=memory or [])
        llm.invoke.assert_not_called()  # no model call
        self.hybrid_search.assert_not_called()
        return data, mock

    def test_a_folder_link_after_no_match_is_offered_under_that_project_name(self):
        data, listed = self.offer(
            "drive.google.com/drive/folders/1kZabc (Proyek Klien Z)", listing("a.pdf", "b.pdf"), memory=self.NOT_FOUND,
        )
        self.assertEqual(listed.call_args[0][0].url, "https://drive.google.com/drive/folders/1kZabc")
        self.assertEqual(data["guide"]["import_link"], {
            "url": "https://drive.google.com/drive/folders/1kZabc", "title": "Proyek Klien Z", "file_count": 2,
            "urls": ["https://drive.google.com/file/d/f0/view", "https://drive.google.com/file/d/f1/view"],
        })
        self.assertIn("folder Google Drive berisi 2 PDF", data["answer"])

    def test_a_file_link_needs_no_listing_and_has_no_title_out_of_context(self):
        data, listed = self.offer("ini filenya https://drive.google.com/file/d/xyz/view?usp=sharing.")
        listed.assert_not_called()
        link = data["guide"]["import_link"]
        self.assertEqual((link["urls"], link["file_count"], link["title"]),
                         (["https://drive.google.com/file/d/xyz/view?usp=sharing"], 1, None))

    def test_a_private_or_empty_folder_gets_no_card(self):
        # A bare link has no words to tell the language by: the conversation's is used.
        link, memory = "https://drive.google.com/drive/folders/1kZ", self.NOT_FOUND
        data, _ = self.offer(link, error=HTTPException(status_code=400), memory=memory)
        self.assertIn("Saya tidak bisa membuka link itu", data["answer"])
        self.assertIsNone(data["guide"])
        data, _ = self.offer(link, listing(), memory=memory)
        self.assertIn("tidak berisi PDF", data["answer"])
        self.assertIsNone(data["guide"])

    def test_the_offer_comes_even_with_no_source_switched_on(self):
        llm = fake_llm("tidak dipakai")
        with patch.object(drive_import, "list_drive_folder_items", AsyncMock(return_value=listing("a.pdf"))):
            data = self.ask("dokumennya di drive.google.com/drive/folders/1kZ", llm, include_pdf_results=False)
        self.assertEqual(data["guide"]["import_link"]["file_count"], 1)



class DriveLinkAndNameTests(unittest.TestCase):
    def test_drive_file_and_folder_links_only(self):
        self.assertEqual(find_drive_link("lihat drive.google.com/drive/u/0/folders/1kZ_x-9) ya"),
                         "https://drive.google.com/drive/u/0/folders/1kZ_x-9")
        self.assertIsNone(find_drive_link("https://docs.google.com/document/d/abc/edit"))
        self.assertIsNone(find_drive_link("https://drive.google.com/"))
        self.assertIsNone(find_drive_link("dokumen proyek klien Z"))

    def test_requested_name_reads_back_only_the_latest_no_match_reply(self):
        en = [{"role": "assistant", "content": 'I couldn\'t find a folder "Client Z" among the documents...'}]
        self.assertEqual(requested_name(en), "Client Z")
        later = DriveImportOfferTests.NOT_FOUND + [{"role": "assistant", "content": "Saya menemukan folder ..."}]
        self.assertIsNone(requested_name(later))
        self.assertIsNone(requested_name(None))


if __name__ == "__main__":
    unittest.main()
