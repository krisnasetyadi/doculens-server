"""MS-1112: stable requirement keys, structured location pointers, trends,
the rules every checked item keeps (services/rag/gap_locations.py), and the
Word / Excel extractors that make DOCX and XLSX pointers exact."""

import os
import tempfile
import unittest

from services.ingest.documents import extract_docx_blocks, extract_xlsx_blocks
from services.rag.gap_locations import (
    build_requirements,
    finalize_item,
    find_block,
    make_pointer,
    normalize_status,
    split_code,
    trend,
)

SEARCHED = {"folders": ["02 IT Infra", "03 Dev"], "document_count": 5}


def no_blocks(collection_id, file_name):
    return []


class RequirementKeyTest(unittest.TestCase):
    def test_codes_become_keys_and_names_become_slugs(self):
        reqs = build_requirements([
            {"label": "A.5.1 Policies for information security", "expected_document": "security policy"},
            "A.8.13 Information backup",
            "Management review",
            "Management review",  # duplicate, dropped
            {"label": "Management Review!"},  # same slug, new key
            "",
        ])
        self.assertEqual([r["key"] for r in reqs], ["A.5.1", "A.8.13", "management-review", "management-review-2"])
        self.assertEqual((reqs[0]["code"], reqs[0]["label"], reqs[0]["expected_document"]),
                         ("A.5.1", "Policies for information security", "security policy"))
        self.assertIsNone(reqs[2]["code"])

    def test_the_same_list_gets_the_same_keys(self):
        raw = ["A.5.1 Policies", "CC6.1 Logical access", "4.2.1 Risk treatment", "Backup"]
        self.assertEqual(build_requirements(raw), build_requirements(list(raw)))
        self.assertEqual(split_code("CC6.1 Logical access"), ("CC6.1", "Logical access"))
        self.assertEqual(split_code("4.2.1 Risk treatment"), ("4.2.1", "Risk treatment"))
        self.assertEqual(split_code("Information security policy"), (None, "Information security policy"))


class StatusAndTrendTest(unittest.TestCase):
    def test_old_statuses_read_as_new_ones(self):
        self.assertEqual([normalize_status(s) for s in ("met", "partial", "not_met", "unknown", "Not Applicable", "??")],
                         ["done", "incomplete", "missing", "needs_decision", "not_applicable", "needs_decision"])

    def test_trend(self):
        self.assertEqual(trend("incomplete", "done"), "improved")
        self.assertEqual(trend("met", "missing"), "worse")
        self.assertEqual(trend("done", "done"), "unchanged")
        self.assertIsNone(trend(None, "done"))
        self.assertIsNone(trend("not_checked", "done"))


class PointerTest(unittest.TestCase):
    def test_pdf_and_slide_pointers_take_the_indexed_page(self):
        pdf = make_pointer("Policy.pdf", "c1", page="9")
        self.assertEqual((pdf["kind"], pdf["page"], pdf["label"]), ("pdf", 9, "Policy.pdf · p. 9"))
        self.assertEqual(make_pointer("Deck.pptx", "c1", page="4")["label"], "Deck.pptx · slide 4")

    def test_docx_and_xlsx_pointers_match_the_quote_to_a_block(self):
        docx_blocks = [
            {"section": "2. Schedule", "paragraph": 1, "text": "Backups run every night at 01:00."},
            {"section": "4. Restore", "paragraph": 2, "text": "A restore test is run every three months by IT."},
        ]
        p = make_pointer("Backup.docx", "c1", blocks=docx_blocks, quote="restore test is run every three months")
        self.assertEqual((p["section"], p["paragraph"], p["label"]),
                         ("4. Restore", 2, "Backup.docx · Section 4. Restore, paragraph 2"))

        rows = [{"sheet": "Access", "row": 6, "text": "User: andi, Granted: 2026-08-01, Revoked: None"},
                {"sheet": "Access", "row": 7, "text": "User: budi, Granted: 2026-08-02"}]
        p = make_pointer("Access.xlsx", "c1", blocks=rows, quote="budi, Granted: 2026-08-02")
        self.assertEqual(p["label"], "Access.xlsx · Access, row 7")

    def test_a_short_quote_is_placed_by_the_passage_the_ai_read(self):
        blocks = [
            {"section": "1. Scope", "paragraph": 1, "text": "This policy covers backup of all company systems."},
            {"section": "5. Restore", "paragraph": 1, "text": "Every backup is restored and tested each quarter."},
        ]
        passage = "5. Restore Every backup is restored and tested each quarter. 6. Records"
        self.assertEqual(find_block(blocks, "backup", passage)["section"], "5. Restore")
        self.assertIsNone(find_block(blocks, "backup", ""))  # too short to place on its own

    def test_a_quote_in_two_paragraphs_is_placed_in_the_one_the_ai_read(self):
        blocks = [
            {"section": "2. Schedule", "paragraph": 4, "text": "Restore tests run each quarter for the ERP database."},
            {"section": "5. Restore", "paragraph": 2, "text": "Restore tests run each quarter for every system."},
        ]
        passage = "5. Restore Restore tests run each quarter for every system."
        self.assertEqual(find_block(blocks, "restore tests run each quarter", passage)["section"], "5. Restore")

    def test_an_unmatched_quote_falls_back_to_the_passage_then_the_file(self):
        blocks = [{"section": None, "paragraph": 3, "text": "Logs are kept twelve months and reviewed monthly."}]
        self.assertEqual(find_block(blocks, "nothing like it", "logs are kept twelve months")["paragraph"], 3)
        p = make_pointer("Logging.docx", "c1", blocks=blocks, quote="unrelated words", passage="other text")
        self.assertEqual((p["paragraph"], p["label"]), (None, "Logging.docx"))


class FinalizeItemTest(unittest.TestCase):
    REQ = {"key": "A.8.13", "code": "A.8.13", "label": "Information backup", "expected_document": "backup procedure"}
    SOURCES = [
        {"id": "S1", "file_name": "Backup.pdf", "collection_id": "c1", "page": "2", "text": "Backups run nightly."},
        {"id": "S2", "file_name": "Backup.docx", "collection_id": "c2", "page": "1", "text": "Restore is tested yearly."},
    ]

    def test_findings_get_structured_pointers(self):
        loaded = []

        def blocks(cid, name):
            loaded.append((cid, name))
            return [{"section": "5. Restore", "paragraph": 1, "text": "Restore is tested yearly."}]

        item = finalize_item(self.REQ, {
            "key": "A.8.13", "status": "incomplete", "reason": "Restore tests are not described. Extra text.",
            "summary": "The procedure exists but does not cover restore tests.",
            "next_step": {"action": "revise_document", "explain": "Add a restore test section."},
            "findings": [
                {"verdict": "met", "text": "Nightly backups", "source": "S1", "quote": "Backups run nightly"},
                {"verdict": "partial", "text": "Restore only yearly", "source": "[S2]", "quote": "tested yearly"},
                {"verdict": "missing", "text": "Quarterly restore test", "source": None, "suggested_place": "after Section 4"},
            ],
        }, self.SOURCES, SEARCHED, blocks)
        self.assertEqual((item["status"], item["requirement_key"], item["reason"]),
                         ("incomplete", "A.8.13", "Restore tests are not described."))
        self.assertEqual([f["location"]["label"] if f["location"] else None for f in item["findings"]],
                         ["Backup.pdf · p. 2", "Backup.docx · Section 5. Restore, paragraph 1", None])
        self.assertEqual(item["findings"][2]["suggested_place"], "after Section 4")
        self.assertEqual(loaded, [("c2", "Backup.docx")])  # only Word/Excel files are re-read
        self.assertEqual(item["next_step"], {"action": "revise_document", "explain": "Add a restore test section."})
        self.assertIsNone(item["searched"])
        self.assertEqual((item["source_citation"], item["evidence"]), ("Backup.pdf · p. 2", "Backups run nightly"))

    def test_a_status_without_any_pointer_gets_the_first_source(self):
        item = finalize_item(self.REQ, {"status": "done", "reason": "Covered.", "findings": [
            {"verdict": "met", "text": "Backups exist", "source": "S9"}]}, self.SOURCES, SEARCHED, no_blocks)
        self.assertEqual(item["findings"][0]["location"]["label"], "Backup.pdf · p. 2")
        self.assertIsNone(item["next_step"])

        bare = finalize_item(self.REQ, {"status": "needs_decision", "findings": []}, self.SOURCES, SEARCHED, no_blocks)
        self.assertEqual(bare["findings"][0]["location"]["file_name"], "Backup.pdf")
        self.assertEqual(bare["next_step"]["action"], "decide_status")

    def test_done_with_nothing_to_point_at_is_missing_and_says_where_it_searched(self):
        item = finalize_item(self.REQ, {"status": "done", "reason": "Fine."}, [], SEARCHED, no_blocks)
        self.assertEqual((item["status"], item["reason"]), ("missing", "No document covers this requirement yet."))
        self.assertEqual(item["searched"], SEARCHED)
        self.assertEqual(item["findings"][-1]["searched"], "Searched 02 IT Infra, 03 Dev (5 documents)")
        self.assertEqual(item["next_step"]["action"], "upload_document")

    def test_missing_findings_say_where_the_ai_looked(self):
        item = finalize_item(self.REQ, {"status": "missing", "reason": "No backup procedure.", "findings": [
            {"verdict": "missing", "text": "Backup procedure"}]}, self.SOURCES, {"folders": [], "document_count": 1}, no_blocks)
        self.assertEqual(item["findings"][0]["searched"], "Searched 1 document")

    def test_not_checked_carries_no_findings_or_next_step(self):
        item = finalize_item(self.REQ, {"status": "not_checked", "reason": "Needs HR records.", "findings": [
            {"verdict": "met", "text": "x", "source": "S1"}]}, self.SOURCES, SEARCHED, no_blocks)
        self.assertEqual((item["status"], item["findings"], item["next_step"], item["searched"]),
                         ("not_checked", [], None, None))

    def test_a_requirement_the_llm_skipped(self):
        self.assertEqual(finalize_item(self.REQ, None, self.SOURCES, SEARCHED, no_blocks)["status"], "needs_decision")
        self.assertEqual(finalize_item(self.REQ, None, [], SEARCHED, no_blocks)["status"], "not_checked")


class ExtractorTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_word_paragraphs_are_numbered_under_their_section(self):
        import docx
        document = docx.Document()
        document.add_paragraph("Backup Procedure MLT-INF-P05")
        document.add_heading("1. Purpose", level=1)
        document.add_paragraph("Protect company data.")
        document.add_paragraph("   ")
        document.add_paragraph("Keep copies offsite.")
        bold = document.add_paragraph()
        bold.add_run("2. Schedule").bold = True  # a numbered bold line, no Heading style
        table = document.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "Daily"
        table.rows[0].cells[1].text = "01:00"
        path = os.path.join(self.dir.name, "backup.docx")
        document.save(path)

        self.assertEqual(extract_docx_blocks(path), [
            {"section": None, "paragraph": 1, "text": "Backup Procedure MLT-INF-P05"},
            {"section": "1. Purpose", "paragraph": 1, "text": "Protect company data."},
            {"section": "1. Purpose", "paragraph": 2, "text": "Keep copies offsite."},
            {"section": "2. Schedule", "paragraph": 1, "text": "Daily | 01:00"},
        ])
        self.assertEqual(extract_docx_blocks(os.path.join(self.dir.name, "missing.docx")), [])

    def test_spreadsheet_rows_keep_their_sheet_and_excel_row_number(self):
        import openpyxl
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Access"
        sheet.append([])
        sheet.append(["User", "Revoked"])
        sheet.append(["andi", "2026-08-10"])
        sheet.append(["budi", None])
        path = os.path.join(self.dir.name, "access.xlsx")
        workbook.save(path)

        self.assertEqual(extract_xlsx_blocks(path), [
            {"sheet": "Access", "row": 3, "text": "User: andi, Revoked: 2026-08-10"},
            {"sheet": "Access", "row": 4, "text": "User: budi"},
        ])




class PointerPreviewTest(unittest.TestCase):
    """A Word or Excel pointer opens the text preview on its paragraph or row."""

    def setUp(self):
        from unittest.mock import patch
        from config import config
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        os.makedirs(os.path.join(self.dir.name, "c1"))
        patcher = patch.object(config, "upload_folder", self.dir.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def content(self, file_name, **kwargs):
        import asyncio
        from unittest.mock import patch
        from models import UserRecord
        from services import collections as collection_service
        admin = UserRecord(user_id="u1", email="a@example.com", role="admin", is_active=True)
        with patch.object(collection_service.supabase_storage, "get_collection", return_value={"owner_id": "u1"}):
            return asyncio.run(collection_service.get_text_content("c1", file_name, offset=0, limit=50, user=admin, **kwargs))

    def test_a_docx_pointer_opens_on_its_paragraph(self):
        import docx
        document = docx.Document()
        document.add_paragraph("Intro")
        for section in ("1. Purpose", "2. Scope", "3. Restore"):
            document.add_heading(section, level=1)
            document.add_paragraph(f"{section} first")
            document.add_paragraph(f"{section} second")
        document.save(os.path.join(self.dir.name, "c1", "proc.docx"))

        page = self.content("proc.docx", section="3. Restore", paragraph=2)
        lines = [line["content"] for line in page["lines"]]
        self.assertEqual(lines[:4], ["2. Scope second", "§ 3. Restore", "3. Restore first", "3. Restore second"])
        self.assertEqual(page["target_line"], page["lines"][3]["line_number"])
        self.assertIsNone(self.content("proc.docx", section="9. Nope", paragraph=1)["target_line"])
        self.assertEqual(self.content("proc.docx", paragraph=1)["target_line"], 1)  # before the first heading

    def test_an_xlsx_pointer_opens_on_its_row(self):
        import openpyxl
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Access"
        sheet.append(["User", "Revoked"])
        sheet.append(["andi", "2026-08-10"])
        workbook.save(os.path.join(self.dir.name, "c1", "access.xlsx"))

        page = self.content("access.xlsx", sheet="Access", row=2)
        self.assertEqual([line["content"] for line in page["lines"]], ["[Sheet: Access]", "Row 2: User: andi, Revoked: 2026-08-10"])
        self.assertEqual(page["target_line"], 2)


if __name__ == "__main__":
    unittest.main()
