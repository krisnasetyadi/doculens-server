"""MS-1115: the rules of correcting a status the AI got wrong
(services/compliance_corrections.py). No database: document checks are faked."""

import os
import unittest

os.environ.setdefault("JWT_SECRET", "ms1115-local-test-secret-not-for-production")

from services.compliance_corrections import (
    ai_accuracy,
    content_version,
    carry_corrections,
    present_item,
)

CORRECTION = {"kind": "wrong_status", "status": "done", "ai_status": "missing", "reason": "Reviewed quarterly.",
              "carried_from_run_id": "run-1",
              "corrected_by_name": "Dewi", "basis": [{"collection_id": "c1", "file_name": "a.pdf", "updated_at": "t1"}]}


class CarryTest(unittest.TestCase):
    def items(self):
        return [{"requirement_key": "A.1", "status": "missing"}, {"requirement_key": "A.2", "status": "not_checked"},
                {"requirement_key": "A.3", "status": "missing"}]

    def test_a_correction_is_kept_while_its_documents_are_unchanged(self):
        items = self.items()
        carry_corrections(items, {"A.1": CORRECTION, "A.2": CORRECTION}, unchanged=lambda basis: True)
        self.assertEqual(items[0]["correction"]["carried_from_run_id"], "run-1")
        self.assertNotIn("correction", items[1])  # not checked this time: nothing to correct
        self.assertNotIn("correction", items[2])

    def test_a_new_answer_from_the_ai_lets_it_stand(self):
        items = [{"requirement_key": "A.1", "status": "done"}]
        carry_corrections(items, {"A.1": CORRECTION}, unchanged=lambda basis: True)
        self.assertNotIn("correction", items[0])

    def test_a_changed_document_lets_the_ai_answer_stand(self):
        items = self.items()
        carry_corrections(items, {"A.1": CORRECTION}, unchanged=lambda basis: False)
        self.assertNotIn("correction", items[0])


class ContentVersionTest(unittest.TestCase):
    def test_only_a_change_of_content_counts(self):
        doc = {"file_names": ["a.pdf"], "storage_paths": ["c1/a.pdf"], "chunk_count": 12, "folder_id": "f1",
               "status": "active", "updated_at": "t1"}
        moved = {**doc, "folder_id": "f2", "status": "inactive", "updated_at": "t2"}
        self.assertEqual(content_version(doc), content_version(moved))
        self.assertNotEqual(content_version(doc), content_version({**doc, "chunk_count": 13}))
        self.assertNotEqual(content_version(doc), content_version({**doc, "file_names": ["a.pdf", "b.pdf"]}))


class PresentTest(unittest.TestCase):
    def test_a_corrected_item_reads_as_its_correction(self):
        item = present_item({"status": "missing", "reason": "Nothing found.", "summary": "Nothing found.",
                             "next_step": {"action": "upload_document"}, "findings": [], "correction": CORRECTION})
        self.assertEqual((item["status"], item["ai_status"], item["reason"], item["next_step"]),
                         ("done", "missing", "Reviewed quarterly.", None))

    def test_an_item_without_a_correction_only_gains_ai_status(self):
        self.assertEqual(present_item({"status": "missing"}), {"status": "missing", "ai_status": "missing"})


class AccuracyTest(unittest.TestCase):
    def test_only_statuses_the_ai_claimed_and_a_person_changed_count(self):
        items = [
            {"ai_status": "missing", "correction": {"status": "done"}},  # changed: counts against the AI
            {"ai_status": "done", "correction": {"status": "done"}},  # a missed document, same status
            {"ai_status": "needs_decision", "correction": {"status": "done"}},  # the AI never claimed one
            {"ai_status": "not_checked"},
            {"ai_status": "needs_decision", "answered": False},
            {"ai_status": "incomplete"},
        ]
        self.assertEqual(ai_accuracy(items), {"checked": 3, "corrected": 1})


if __name__ == "__main__":
    unittest.main()
