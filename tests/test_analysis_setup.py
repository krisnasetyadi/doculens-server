"""MS-1114: starting an analysis from one sentence (services/analysis_setup.py).
No database and no real model: storage and the LLM are faked."""

import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms1114-local-test-secret-not-for-production")

from fastapi import HTTPException

from models import SetupDraftRequest, TableMapping, UserRecord
from services import analysis_setup
from services.lens_guide.scope import Scope, ScopeCollection, ScopeFolder
from tests.fakes import fake_llm
from tests.test_requirement_table import soa_workbook

USER = UserRecord(user_id="u1", email="u1@example.com", role="user", is_active=True)

SCOPE = Scope(
    folders=[
        ScopeFolder("f-iso", "ISO 27001", None, ["ISO 27001"]),
        ScopeFolder("f-docs", "Dokumen", "f-iso", ["ISO 27001", "Dokumen"]),
        ScopeFolder("f-rules", "Rules", None, ["Rules"]),
        ScopeFolder("f-empty", "Empty", None, ["Empty"]),
    ],
    collections=[
        ScopeCollection("c-policy", "Backup policy", 1, "f-docs", ["ISO 27001", "Dokumen"]),
        ScopeCollection("c-notes", "Meeting notes", 1, "f-iso", ["ISO 27001"]),
        ScopeCollection("c-soa", "SoA ISO27001", 1, "f-rules", ["Rules"]),
        ScopeCollection("c-brief", "Product brief", 1, None, []),
    ],
)
FLAGS = {
    "c-policy": {"file_names": ["Backup policy.docx"], "is_reference_framework": False},
    "c-notes": {"file_names": ["notes.png"], "is_reference_framework": False},
    "c-soa": {"file_names": ["SoA ISO27001.xlsx"], "is_reference_framework": True},
    "c-brief": {"file_names": ["brief.pdf"], "is_reference_framework": False},
}


class SuggestTest(unittest.TestCase):
    def setUp(self):
        for target, value in (("load_scope", SCOPE),):
            p = patch.object(analysis_setup, target, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(analysis_setup.supabase_storage, "collection_reference_flags", return_value=FLAGS)
        p.start()
        self.addCleanup(p.stop)

    def suggest(self, reply=None, error=None):
        llm = fake_llm(reply, error=error)
        with patch.object(analysis_setup.processor, "get_llm", return_value=(llm, "m/x")):
            return analysis_setup.suggest("Are our documents ready for the ISO 27001 audit?", USER)

    def test_the_ai_picks_the_rule_file_and_folders(self):
        suggestion, tokens = self.suggest(
            '```json\n{"rule_file": "R1", "rule_file_why": "It lists ISO controls.", "folders": ["F1", "F9"], '
            '"folders_why": "It holds the ISO documents."}\n```')
        self.assertEqual((suggestion.rule_file.collection_id, suggestion.suggested_by, tokens), ("c-soa", "ai", 42))
        self.assertEqual(suggestion.rule_file_why, "It lists ISO controls.")
        self.assertEqual([f.folder_id for f in suggestion.folders if f.suggested], ["f-iso"])  # F9 was never shown
        # Marked rule files first; an image can't be a rule file.
        self.assertEqual([f.collection_id for f in suggestion.rule_files], ["c-soa", "c-policy", "c-brief"])

    def test_folders_count_their_subfolders_and_loose_documents_get_their_own_entry(self):
        suggestion, _ = self.suggest('{"rule_file": null, "folders": []}')
        by_name = {f.name: f for f in suggestion.folders}
        self.assertEqual(sorted(by_name["ISO 27001"].collection_ids), ["c-notes", "c-policy"])
        self.assertEqual((by_name["Not in a folder"].folder_id, by_name["Not in a folder"].collection_ids), (None, ["c-brief"]))
        self.assertNotIn("Empty", by_name)  # a folder without documents is not offered
        self.assertIsNone(suggestion.rule_file)

    def test_without_the_ai_names_decide(self):
        suggestion, tokens = self.suggest(error=RuntimeError("429 quota"))
        self.assertEqual((suggestion.suggested_by, tokens), ("guess", 0))
        self.assertEqual(suggestion.rule_file.collection_id, "c-soa")  # "iso" and "27001" in its name
        self.assertEqual([f.folder_id for f in suggestion.folders if f.suggested], ["f-iso", "f-docs"])


class SwitchedOffDocumentsTest(unittest.TestCase):
    def test_a_document_kept_out_of_chat_can_still_be_the_rule_file(self):
        from services.lens_guide import scope as scope_module
        rows = [{"collection_id": "c-soa", "title": "SoA", "file_names": ["SoA.xlsx"], "folder_id": None}]
        with patch.object(scope_module.supabase_storage, "list_folders_for_user", return_value=[]), \
                patch.object(scope_module.supabase_storage, "list_pdf_collections_for_user", return_value=rows) as every, \
                patch.object(scope_module.supabase_storage, "list_active_collections_for_user", return_value=[]) as active, \
                patch.object(analysis_setup.supabase_storage, "collection_reference_flags",
                             return_value={"c-soa": {"file_names": ["SoA.xlsx"], "is_reference_framework": True}}), \
                patch.object(analysis_setup.processor, "get_llm", return_value=(fake_llm(error=RuntimeError("x")), "m/x")):
            suggestion, _ = analysis_setup.suggest("ISO audit", USER)
        self.assertEqual(suggestion.rule_file.collection_id, "c-soa")
        every.assert_called_once()
        active.assert_not_called()


class MeteringTest(unittest.TestCase):
    """The setup's AI calls are held to the same limits as the check itself."""

    def run_suggest(self, **limits):
        import asyncio
        work = patch.object(analysis_setup, "suggest", return_value=("suggestion", 120))
        patches = {
            "plan": patch.object(analysis_setup, "enforce_gap_check_plan"),
            "workspace": patch.object(analysis_setup, "resolve_workspace_id", return_value="w-1"),
            "rate": patch.object(analysis_setup, "enforce_rate_limit", **limits.get("rate", {})),
            "quota": patch.object(analysis_setup, "enforce_plan_limit", return_value=None),
            "member": patch.object(analysis_setup, "enforce_member_allocation", return_value=None),
            "log": patch.object(analysis_setup, "log_token_usage"),
        }
        mocks = {name: p.start() for name, p in patches.items()}
        self.addCleanup(patch.stopall)
        suggest = work.start()
        return mocks, suggest, lambda: asyncio.run(analysis_setup.suggest_for(" ISO audit ", USER))

    def test_limits_are_checked_first_and_tokens_logged_after(self):
        mocks, suggest, call = self.run_suggest()
        self.assertEqual(call(), "suggestion")
        mocks["quota"].assert_called_once_with(USER, 0, analysis_setup.SUGGEST_TOKEN_RESERVE)
        suggest.assert_called_once_with("ISO audit", USER)
        mocks["log"].assert_called_once_with("u1", 120)

    def test_a_refused_request_never_reaches_the_ai(self):
        mocks, suggest, call = self.run_suggest(rate={"side_effect": HTTPException(status_code=429, detail="slow down")})
        with self.assertRaises(HTTPException):
            call()
        suggest.assert_not_called()


class DraftTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "SoA.xlsx")
        soa_workbook(self.path)
        patches = [
            patch.object(analysis_setup.supabase_storage, "list_collection_ids_for_user", return_value=["c-soa"]),
            patch.object(analysis_setup.supabase_storage, "get_collection",
                         return_value={"collection_id": "c-soa", "title": "SoA", "file_names": ["SoA.xlsx"]}),
            patch.object(analysis_setup, "_read_document_blocks", side_effect=lambda cid, name, extract: extract(self.path)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def draft(self, reply=None, **body):
        llm = fake_llm(reply, error=None if reply else RuntimeError("no model"))
        with patch.object(analysis_setup.processor, "get_llm", return_value=(llm, "m/x")):
            return analysis_setup.draft(SetupDraftRequest(collection_id="c-soa", **body), USER)

    def test_the_ai_maps_the_columns_and_every_row_is_read_as_written(self):
        draft, _ = self.draft('{"sheet": "SoA", "header_row": 3, "code": "A", "name": "B", "applicable": "C"}')
        self.assertEqual((draft.kind, draft.mapped_by, len(draft.requirements)), ("table", "ai", 93))
        self.assertEqual(draft.requirements[0].source["label"], "SoA · SoA, row 4")
        self.assertEqual([s.name for s in draft.sheets], ["Cover", "SoA"])

    def test_a_mapping_the_ai_got_wrong_falls_back_to_the_headers(self):
        draft, _ = self.draft('{"sheet": "SoA", "header_row": 3, "code": "A", "name": "Q"}')  # no column Q
        self.assertEqual((draft.mapped_by, len(draft.requirements)), ("guess", 93))

    def test_the_users_mapping_is_used_and_checked(self):
        draft, tokens = self.draft(mapping=TableMapping(sheet="SoA", header_row=3, code=None, name="B"))
        self.assertEqual((draft.mapped_by, tokens, draft.requirements[0].code), ("user", 0, None))
        with self.assertRaises(HTTPException) as refused:
            self.draft(mapping=TableMapping(sheet="SoA", header_row=3, name="Q"))
        self.assertEqual(refused.exception.status_code, 400)

    def test_only_a_document_the_user_can_open(self):
        with self.assertRaises(HTTPException) as refused:
            analysis_setup.draft(SetupDraftRequest(collection_id="someone-elses"), USER)
        self.assertEqual(refused.exception.status_code, 403)
        with self.assertRaises(HTTPException) as missing:
            self.draft(file_name="other.xlsx")
        self.assertEqual(missing.exception.status_code, 404)


class DocumentDraftTest(unittest.TestCase):
    def test_a_pdf_is_read_by_the_ai_and_each_item_points_at_its_page(self):
        with patch.object(analysis_setup.supabase_storage, "list_collection_ids_for_user", return_value=["c-iso"]), \
                patch.object(analysis_setup.supabase_storage, "get_collection",
                             return_value={"collection_id": "c-iso", "title": "", "file_names": ["ISO 27001.pdf"]}), \
                patch.object(analysis_setup.processor, "extract_framework_items", return_value=(
                    [{"key": "A.5.1", "code": "A.5.1", "label": "Policies", "expected_document": "policy"},
                     {"key": "backups", "code": None, "label": "Backups", "expected_document": None}], 120)), \
                patch.object(analysis_setup, "_indexed_pages", return_value=[(3, "... a.5.1 policies ..."), (7, "x")]):
            draft, tokens = analysis_setup.draft(SetupDraftRequest(collection_id="c-iso"), USER)
        self.assertEqual((draft.kind, draft.framework_name, tokens), ("document", "ISO 27001", 120))
        self.assertEqual([r.source for r in draft.requirements],
                         [{"label": "ISO 27001 · p. 3", "page": 3}, {"label": "ISO 27001"}])


class HelpersTest(unittest.TestCase):
    def test_a_json_object_inside_a_reply(self):
        self.assertEqual(analysis_setup.parse_json_object('Sure!\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertIsNone(analysis_setup.parse_json_object("no json here"))
        self.assertIsNone(analysis_setup.parse_json_object("{broken"))

    def test_a_page_by_code_then_by_name(self):
        pages = [(1, "intro"), (4, "a.8.13 information backup"), (9, "information backup again")]
        self.assertEqual(analysis_setup.find_page(pages, "A.8.13", "Information backup"), 4)
        self.assertEqual(analysis_setup.find_page(pages, None, "Information backup"), 4)
        self.assertIsNone(analysis_setup.find_page(pages, None, "Nowhere"))


if __name__ == "__main__":
    unittest.main()
