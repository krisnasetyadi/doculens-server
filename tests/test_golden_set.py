"""MS-908: the golden question set stays consistent with its source files.

Builds the fixtures (no OCR, no embeddings) and checks that every question
points at a real file and page, and that its expected answer is really on
that page, so a typo in questions.json cannot silently lower the hit rate."""

import json
import os
import tempfile
import unittest

from scripts.golden_set import run
from scripts.golden_set.fixtures import SCANNED_PAGES, build_fixtures


class GoldenSetConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.truth = build_fixtures(cls._dir.name)
        with open(run.QUESTIONS_PATH, encoding="utf-8") as handle:
            cls.questions = json.load(handle)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def test_every_source_file_is_written(self):
        for name in self.truth:
            self.assertTrue(os.path.getsize(os.path.join(self._dir.name, name)) > 0, name)

    def test_every_answer_is_on_the_expected_page(self):
        for question in self.questions:
            with self.subTest(question["id"]):
                pages = self.truth[question["source"]]
                self.assertIn(question["page"], pages)
                self.assertIn(run.normalize(question["answer"]), run.normalize(pages[question["page"]]))

    def test_ids_are_unique_and_every_type_is_covered(self):
        ids = [q["id"] for q in self.questions]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({run.category(q) for q in self.questions},
                         {"pdf digital", "pdf scan", "docx", "xlsx", "csv", "chat"})
        self.assertEqual({q["lang"] for q in self.questions}, {"id", "en"})

    def test_scanned_pages_have_no_embedded_text(self):
        import pdfplumber

        for source, pages in SCANNED_PAGES.items():
            with pdfplumber.open(os.path.join(self._dir.name, source)) as pdf:
                for page in pages:
                    self.assertEqual((pdf.pages[page - 1].extract_text() or "").strip(), "", f"{source} p{page}")

    def test_rank_finds_the_first_matching_file_and_page(self):
        class Doc:
            def __init__(self, source, page):
                self.metadata = {"source": source, "page": page}

        results = [Doc("a.pdf", "1"), Doc("b.pdf", "2"), Doc("b.pdf", "3")]
        self.assertEqual(run.first_rank(results, "b.pdf", 3), 3)
        self.assertEqual(run.first_rank(results, "b.pdf", None), 2)
        self.assertIsNone(run.first_rank(results, "c.pdf", 1))


if __name__ == "__main__":
    unittest.main()
