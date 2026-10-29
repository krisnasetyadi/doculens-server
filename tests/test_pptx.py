"""MS-910: text extraction from PowerPoint decks.

Decks are built with python-pptx inside each test, so no fixture files and no
database are needed."""

import os
import tempfile
import unittest
from unittest.mock import patch

from pptx import Presentation
from pptx.util import Inches

from services.ingest import documents
from services.ingest.documents import DOCUMENT_EXTRACTORS, extract_text_from_pptx

LONG = "Kebijakan perjalanan dinas berlaku untuk seluruh karyawan tetap dan kontrak. " * 4


class PptxExtractionTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.deck = Presentation()
        self.blank = self.deck.slide_layouts[6]

    def add_slide(self, text=None):
        slide = self.deck.slides.add_slide(self.blank)
        if text:
            slide.shapes.add_textbox(Inches(1), Inches(1), Inches(6), Inches(1)).text_frame.text = text
        return slide

    def extract(self):
        path = os.path.join(self._dir.name, "deck.pptx")
        self.deck.save(path)
        return extract_text_from_pptx(path)

    def test_text_boxes_tables_groups_and_notes_keep_their_slide_number(self):
        self.add_slide(LONG)
        table_slide = self.add_slide(LONG)
        table = table_slide.shapes.add_table(2, 2, Inches(1), Inches(3), Inches(6), Inches(1)).table
        table.cell(0, 0).text, table.cell(0, 1).text = "Vendor", "Nilai kontrak"
        table.cell(1, 0).text, table.cell(1, 1).text = "PT Sinar Abadi", "Rp 120 juta"
        group_slide = self.add_slide(LONG)
        group_slide.shapes.add_group_shape().shapes.add_textbox(
            Inches(1), Inches(3), Inches(4), Inches(1)).text_frame.text = "SLA 99,5 persen"
        group_slide.notes_slide.notes_text_frame.text = "Kode proyek ZETA-42"

        pages = self.extract()

        self.assertEqual([p["page"] for p in pages], [1, 2, 3])
        self.assertIn("PT Sinar Abadi | Rp 120 juta", pages[1]["text"])
        self.assertIn("SLA 99,5 persen", pages[2]["text"])
        self.assertIn("Kode proyek ZETA-42", pages[2]["text"])

    def test_hidden_slides_are_left_out(self):
        self.add_slide(LONG)
        self.add_slide("RAHASIA " + LONG)._element.set("show", "0")
        self.add_slide(LONG)

        pages = self.extract()

        self.assertEqual([p["page"] for p in pages], [1, 3])
        self.assertNotIn("RAHASIA", " ".join(p["text"] for p in pages))

    def test_short_slides_join_the_next_ones_and_cite_the_first(self):
        self.add_slide("Agenda")
        self.add_slide("Bagian 1")
        self.add_slide(LONG)
        self.add_slide(LONG)

        pages = self.extract()

        self.assertEqual([p["page"] for p in pages], [1, 4])
        self.assertTrue(pages[0]["text"].startswith("Agenda Bagian 1 Kebijakan"))

    def test_a_broken_slide_is_skipped_and_the_rest_kept(self):
        for _ in range(3):
            self.add_slide(LONG)
        real = documents._pptx_slide_text
        calls = []

        def fail_on_second(slide):
            calls.append(slide)
            if len(calls) == 2:
                raise ValueError("bad shape")
            return real(slide)

        with patch.object(documents, "_pptx_slide_text", fail_on_second):
            pages = self.extract()

        self.assertEqual([p["page"] for p in pages], [1, 3])

    def test_deck_without_text_and_non_pptx_files_give_none(self):
        self.add_slide()
        self.assertIsNone(self.extract())
        legacy = os.path.join(self._dir.name, "old.pptx")
        with open(legacy, "wb") as handle:
            handle.write(b"\xd0\xcf\x11\xe0 not a zip package")
        self.assertIsNone(extract_text_from_pptx(legacy))

    def test_pptx_is_an_accepted_upload_type(self):
        self.assertIs(DOCUMENT_EXTRACTORS[".pptx"], extract_text_from_pptx)
        self.assertNotIn(".ppt", DOCUMENT_EXTRACTORS)


if __name__ == "__main__":
    unittest.main()
