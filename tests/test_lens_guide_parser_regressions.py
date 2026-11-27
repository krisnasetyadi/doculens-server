"""Review regressions: source-bound paragraphs, version direction and ranges."""
import unittest

from models import MemoryTurn
from services.lens_guide.answer import (
    SourceRef, add_related_results, build_guide_prompt, missing_answer, parse_guide_answer, source_detail,
)
from services.lens_guide.continuation import continuation
from services.lens_guide.scope import Scope, ScopeCollection


def ref(key, name, page=1):
    return SourceRef(key, source_detail(name, "c1", page, "http://testserver"))


class GuideParserReviewRegressions(unittest.TestCase):
    def test_fact_citation_at_paragraph_end_keeps_all_wrapped_lines(self):
        raw = "Core API menyimpan data.\nWeb memanggil Core API.\nWorker menerima tugas asinkron. [S1]"
        answer = parse_guide_answer(raw, "fact", [ref("S1", "architecture.pdf")], [], "id")
        self.assertIn("Core API menyimpan data", answer.text)
        self.assertIn("Web memanggil Core API", answer.text)
        self.assertIn("Worker menerima tugas asinkron", answer.text)
        self.assertEqual(answer.dropped, 0)

    def test_bad_fact_paragraph_is_rejected_as_a_whole(self):
        raw = "Unsupported opening.\nUnsupported ending. [S99]\n\nSupported paragraph. [S1]"
        answer = parse_guide_answer(raw, "fact", [ref("S1", "architecture.pdf")], [], "en")
        self.assertNotIn("Unsupported", answer.text)
        self.assertIn("Supported paragraph", answer.text)
        self.assertEqual(answer.dropped, 1)

    def test_unreferenced_other_paragraph_cannot_borrow_next_paragraph_citation(self):
        answer = parse_guide_answer("No evidence for this.\n\nSupported only here. [S1]", "fact",
                                    [ref("S1", "architecture.pdf")], [], "en")
        self.assertNotIn("No evidence for this", answer.text)
        self.assertIn("Supported only here", answer.text)
        self.assertEqual(answer.dropped, 1)

    def test_fallback_only_displays_one_copy_of_its_notice(self):
        for empty in (True, False):
            answer = missing_answer("id", empty=empty)
            self.assertEqual(answer.lead, "")
            self.assertEqual(answer.text, answer.notices[0].text)

    def test_same_file_two_pages_preserves_both_locations_and_purposes(self):
        raw = "1. Brief | Tujuan | 10 [S1]\n2. Arsitektur | Hubungan modul | 15 [S2]"
        answer = parse_guide_answer(raw, "start", [ref("S1", "guide.pdf", 1), ref("S2", "guide.pdf", 4)], [], "id")
        self.assertEqual(len(answer.steps), 1)
        self.assertEqual([s.page for s in answer.steps[0].sources], [1, 4])
        self.assertEqual(answer.steps[0].where, "hlm 1; hlm 4")
        self.assertIn("Hubungan modul", answer.steps[0].why)
        self.assertIn("25 mnt", answer.lead)
        self.assertEqual(answer.dropped, 0)

    def test_current_version_named_first_is_not_hidden(self):
        raw = ("1. Current | Instruksi berlaku | 10 [S2]\n2. Earlier | Instruksi lama | 10 [S1]\n"
               "VERSI LAMA: dipakai [S2], sedangkan versi lama [S1] diabaikan")
        answer = parse_guide_answer(raw, "start", [ref("S1", "old.pdf"), ref("S2", "current.pdf")], [], "id")
        self.assertEqual([s.file for s in answer.steps], ["current.pdf"])
        self.assertEqual(answer.dropped, 1)

    def test_active_replacement_verb_establishes_direction(self):
        raw = ("1. Current | Instruksi berlaku | 10 [S2]\n2. Earlier | Instruksi lama | 10 [S1]\n"
               "OLD VERSION: [S2] replaces [S1]")
        answer = parse_guide_answer(raw, "start", [ref("S1", "old.pdf"), ref("S2", "current.pdf")], [], "en")
        self.assertEqual([s.file for s in answer.steps], ["current.pdf"])

    def test_ambiguous_version_order_keeps_both_sources(self):
        raw = ("1. Current | Instruksi pertama | 10 [S2]\n2. Earlier | Instruksi kedua | 10 [S1]\n"
               "VERSI LAMA: dokumen [S2] dan dokumen [S1]")
        answer = parse_guide_answer(raw, "start", [ref("S1", "old.pdf"), ref("S2", "current.pdf")], [], "id")
        self.assertEqual({s.file for s in answer.steps}, {"old.pdf", "current.pdf"})
        self.assertIn("Urutan versi belum dapat dipastikan", answer.notices[0].text)
        self.assertEqual(answer.dropped, 0)

    def test_consistent_document_answer_does_not_create_conflict_notice(self):
        raw = "1. Isi checklist | Langkah pertama | [S1]\n2. Minta persetujuan Lead SD | Langkah berikutnya | [S2]"
        answer = parse_guide_answer(raw, "do", [ref("S1", "checklist.pdf"), ref("S2", "approval.pdf")], [], "id")
        self.assertFalse(answer.notices)
        prompt = build_guide_prompt("Checklist and approval describe successive steps.", "Bagaimana rilis?", "do", [], "id")
        self.assertIn("Perbedaan contoh, ruang lingkup, atau urutan langkah bukan bukti konflik", prompt)
        self.assertIn("Jika isi sumber konsisten, jawab tanpa kedua catatan", prompt)
        self.assertNotIn("WAJIB keluarkan", prompt)


class ContinuationReviewRegressions(unittest.TestCase):
    def setUp(self):
        names = [f"chapter-{i}.pdf" for i in range(1, 7)]
        self.scope = Scope(collections=[ScopeCollection("c1", "Guide", len(names), "f1", ["Atlas"], names)])
        raw = "\n".join(f"{i}. Chapter {i} | Purpose {i} | 10 [S{i}]" for i in range(1, 7))
        answer = parse_guide_answer(raw, "start", [ref(f"S{i}", name) for i, name in enumerate(names, 1)], [], "id")
        self.memory = [MemoryTurn(role="assistant", content=answer.text)]

    def test_indonesian_alphabetical_range_completes_middle_steps(self):
        result = continuation("Sudah baca a sampai d", self.memory, self.scope, "http://testserver", "id")
        self.assertEqual(result[1].steps[0].key, "e")

    def test_english_range_and_multiple_ranges(self):
        result = continuation("done with a through c, e to f", self.memory, self.scope, "http://testserver", "en")
        self.assertEqual(result[1].steps[0].key, "d")

    def test_range_does_not_accept_an_unknown_endpoint(self):
        result = continuation("Sudah baca a sampai z", self.memory, self.scope, "http://testserver", "id")
        self.assertEqual(result[1].steps[0].key, "b")

    def test_numeric_range_uses_actual_list_keys(self):
        answer = parse_guide_answer("\n".join(f"{i}. Task {i} | Purpose | [S1]" for i in range(1, 7)), "do",
                                    [ref("S1", "chapter-1.pdf")], [], "id")
        result = continuation("selesai baca 1 hingga 4", [MemoryTurn(role="assistant", content=answer.text)],
                              self.scope, "http://testserver", "id")
        self.assertEqual(result[1].steps[0].key, "5")


class RelatedSourceReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.scope = Scope(collections=[ScopeCollection("outside", "Policy", 1, "f2", ["Policy"], ["policy.pdf"])])

    def row(self, confidence, cid="inside"):
        return {"type": "pdf", "confidence": confidence, "metadata": {
            "collection_id": cid, "source": "policy.pdf", "page": 1}}

    def test_weak_tail_below_prompt_budget_does_not_admit_weak_supplement(self):
        primary = {"merged_results": [self.row(0.8) for _ in range(8)] + [self.row(0.01)]}
        secondary = {"merged_results": [self.row(0.2, "outside")]}
        result = add_related_results(primary, secondary, self.scope, ["inside"])
        self.assertEqual(result, primary)

    def test_floor_prevents_weak_optional_result_even_with_weak_primary(self):
        primary = {"merged_results": [self.row(0.01)]}
        secondary = {"merged_results": [self.row(0.1, "outside")]}
        self.assertEqual(add_related_results(primary, secondary, self.scope, ["inside"]), primary)

    def test_relevant_optional_result_still_joins_the_folder(self):
        primary = {"merged_results": [self.row(0.5)]}
        secondary = {"merged_results": [self.row(0.6, "outside")]}
        result = add_related_results(primary, secondary, self.scope, ["inside"])
        self.assertEqual(len(result["merged_results"]), 2)
        self.assertEqual(result["merged_results"][1]["metadata"]["lens_origin"], "Policy")
