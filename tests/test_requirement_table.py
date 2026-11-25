"""MS-1114: reading a requirement list from a spreadsheet
(services/ingest/requirement_table.py). No database, no LLM."""

import os
import tempfile
import unittest

from services.ingest.requirement_table import (
    applicable,
    apply_mapping,
    guess_mapping,
    read_sheets,
    sheet_preview,
    valid_mapping,
)


def soa_workbook(path, controls=93):
    """An SoA as companies keep them: a title above the header, one row per
    control, a Yes/No applicability column, and a cover sheet."""
    import openpyxl
    workbook = openpyxl.Workbook()
    cover = workbook.active
    cover.title = "Cover"
    cover.append(["Statement of Applicability", "Moonlay"])
    soa = workbook.create_sheet("SoA")
    soa.append(["ISO 27001:2022 Statement of Applicability"])
    soa.append([])
    soa.append(["No.", "Control", "Applicable", "Justification"])
    for i in range(1, controls + 1):
        name = "Review" if i in (10, 11) else f"Control number {i}"  # two rows share a name
        soa.append([f"A.{5 + i // 40}.{i}", name, "No" if i == 7 else "Yes", "Business need"])
    workbook.save(path)


class RequirementTableTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "SoA.xlsx")
        soa_workbook(self.path)
        self.sheets = read_sheets(self.path)

    def test_an_soa_imports_every_row_as_written(self):
        mapping = guess_mapping(self.sheets)
        self.assertEqual(mapping, {"sheet": "SoA", "header_row": 3, "code": "A", "name": "B", "applicable": "C"})
        self.assertTrue(valid_mapping(self.sheets, mapping))

        requirements = apply_mapping(self.sheets, mapping, "SoA")
        self.assertEqual(len(requirements), 93)  # no row lost, none invented
        first = requirements[0]
        self.assertEqual((first["code"], first["label"], first["key"], first["applicable"]),
                         ("A.5.1", "Control number 1", "A.5.1", True))
        self.assertEqual(first["source"], {"label": "SoA · SoA, row 4", "sheet": "SoA", "row": 4})
        self.assertFalse(requirements[6]["applicable"])  # "No"
        self.assertEqual(len({r["key"] for r in requirements}), 93)  # keys stay unique

    def test_a_name_column_holding_the_code_is_split(self):
        sheets = [{"name": "Sheet1", "rows": [(1, ["Requirement"]), (2, ["A.5.1 Policies"]), (3, ["Backups"])]}]
        requirements = apply_mapping(sheets, {"sheet": "Sheet1", "header_row": 1, "code": None, "name": "A",
                                              "applicable": None}, "List")
        self.assertEqual([(r["code"], r["label"], r["key"]) for r in requirements],
                         [("A.5.1", "Policies", "A.5.1"), (None, "Backups", "backups")])
        self.assertEqual(requirements[0]["source"]["label"], "List · row 2")

    def test_a_mapping_must_point_at_real_columns(self):
        self.assertFalse(valid_mapping(self.sheets, {"sheet": "SoA", "header_row": 3, "code": "A", "name": "Z"}))
        self.assertFalse(valid_mapping(self.sheets, {"sheet": "Nope", "header_row": 3, "name": "B"}))
        self.assertFalse(valid_mapping(self.sheets, {"sheet": "SoA", "header_row": 2, "name": "B"}))

    def test_yes_and_no_in_several_words(self):
        for value in ("Yes", "ya", "Applicable", "✓", "Included"):
            self.assertTrue(applicable(value), value)
        for value in ("No", "Tidak", "N/A", "Excluded", "Not applicable"):
            self.assertFalse(applicable(value), value)
        self.assertIsNone(applicable("Partly"))
        self.assertIsNone(applicable(""))

    def test_the_preview_shows_column_letters_and_row_numbers(self):
        soa = next(p for p in sheet_preview(self.sheets) if p["name"] == "SoA")
        self.assertEqual(soa["columns"][:3], ["A", "B", "C"])
        self.assertEqual(soa["rows"][1]["row"], 3)  # the empty row 2 is left out
        self.assertEqual(soa["row_count"], 95)


if __name__ == "__main__":
    unittest.main()
