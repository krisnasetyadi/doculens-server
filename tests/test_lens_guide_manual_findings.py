"""MS-1183: findings from the first manual test with a real model (26 Nov 2026).

Every message here is one a user actually typed, or a model actually wrote."""
import unittest

from models import MemoryTurn
from services.lens_guide.answer import NEAREST_WHY, SourceRef, guide_intent, parse_guide_answer, source_detail
from services.agnostic import _clamp_memory
from services.lens_guide.continuation import (
    _carry, carried_answers, carries_list_state, carry_list, continuation, has_step_list, strip_list_state,
)
from services.lens_guide.match import match_folders
from services.lens_guide.scope import Scope, ScopeCollection, ScopeFolder
from services.rag.conversation.language import detect_reply_language

BASE = "http://testserver"
FILES = ["Project Brief.pdf", "Notulen Kickoff.pdf", "Arsitektur Sistem.pdf", "Panduan Setup Lokal v2.pdf"]


def ref(key, name, page=2, cid="c1"):
    return SourceRef(key, source_detail(name, cid, page, BASE))


def reading_list():
    raw = "\n".join(f"{i}. Baca {name} | Alasan {i} | 10 [S{i}]" for i, name in enumerate(FILES, 1))
    return parse_guide_answer(raw, "start", [ref(f"S{i}", n) for i, n in enumerate(FILES, 1)], [], "id", BASE).text


class CasualWordingTests(unittest.TestCase):
    def test_casual_learning_requests_are_learning(self):
        for text in ("aku mau pelajarin modul backup dan pemulihan", "ak mw pelajari modul backup",
                     "pengen belajarin modul backup"):
            self.assertEqual(guide_intent(text), "learn", text)

    def test_casual_indonesian_keeps_indonesian_after_an_english_turn(self):
        memory = [{"role": "user", "content": "How do I connect Telegram?"}]
        self.assertEqual(detect_reply_language("ak mw pelajari modul backup", memory), "id")

    def test_dan_matches_an_ampersand_in_a_folder_name(self):
        folder = ScopeFolder("f1", "Backup & Pemulihan", None, ["Backup & Pemulihan"])
        self.assertEqual(match_folders("backup dan pemulihan", Scope(folders=[folder])), [folder])


class ContinuationWordingTests(unittest.TestCase):
    def setUp(self):
        self.scope = Scope(collections=[ScopeCollection("c1", "Proyek", 4, "y", ["Proyek Klien Y"], FILES)])
        self.memory = [MemoryTurn(role="user", content="Aku baru masuk project Proyek Klien Y"),
                       MemoryTurn(role="assistant", content=reading_list())]

    def next_key(self, question, memory=None):
        found = continuation(question, memory or self.memory, self.scope, BASE, "id")
        self.assertIsNotNone(found, question)
        return found[1].steps[0].key if found[1].steps else None

    def test_ways_people_report_reading(self):
        for text in ("sudah ku baca a dan b", "udah baca a dan b", "aku sudah baca a dan b", "sudah kubaca a dan b",
                     "Sudah baca a & b", "aku udh baca a, b", "I've read a and b"):
            self.assertEqual(self.next_key(text), "c", text)

    def test_ways_people_ask_to_continue(self):
        for text in ("lanjut dong", "ok lanjut", "Lanjutkan", "next please", "lanjut ya"):
            self.assertEqual(self.next_key(text), "a", text)

    def test_a_bare_done_finishes_the_step_just_shown(self):
        shown = continuation("sudah baca a dan b", self.memory, self.scope, BASE, "id")[0]
        memory = self.memory + [MemoryTurn(role="user", content="sudah baca a dan b"),
                                MemoryTurn(role="assistant", content=shown)]
        self.assertEqual(self.next_key("sudah", memory), "d")
        self.assertEqual(self.next_key("udah", memory), "d")

    def test_a_bare_done_after_going_back_finishes_that_step(self):
        # Manual test: "kembali ke a" -> "Langkah a. ...", then "sudah" went to the router.
        shown = continuation("kembali ke a", self.memory, self.scope, BASE, "id", show="a", forced=True)[0]
        self.assertTrue(shown.startswith("Langkah **a."))
        memory = self.memory + [MemoryTurn(role="user", content="kembali ke a"),
                                MemoryTurn(role="assistant", content=shown)]
        self.assertEqual(self.next_key("sudah", memory), "b")

    def test_a_bare_done_right_after_the_list_is_not_guessed(self):
        self.assertIsNone(continuation("sudah", self.memory, self.scope, BASE, "id"))

    def test_a_nearest_documents_reply_is_not_a_list_to_continue(self):
        nearest = ("1. **Arsitektur Sistem.pdf** ([Arsitektur Sistem.pdf](http://testserver/api/v1/files/c1/"
                   "Arsitektur%20Sistem.pdf#page=3), hlm 3): Dokumen terkait yang ditemukan; prosedur yang diminta "
                   "belum terdokumentasi.")
        memory = self.memory + [MemoryTurn(role="user", content="apa langkah core-api?"),
                                MemoryTurn(role="assistant", content=nearest)]
        self.assertEqual(self.next_key("lanjut", memory), "a")


class ListCarriedForwardTests(unittest.TestCase):
    """Manual test, 26 Nov 2026: once the list left the five-question history, "sudah" went to the
    model, which invented a step "e". Every reply of "lanjut" now carries the list and what is done."""

    def setUp(self):
        self.scope = Scope(collections=[ScopeCollection("c1", "Proyek", 4, "y", ["Proyek Klien Y"], FILES)])
        self.memory = [MemoryTurn(role="user", content="Aku baru masuk project Proyek Klien Y"),
                       MemoryTurn(role="assistant", content=reading_list())]

    def say(self, question, **kw):
        found = continuation(question, self.memory, self.scope, BASE, "id", **kw)
        self.assertIsNotNone(found, question)
        text, guide = found
        self.memory += [MemoryTurn(role="user", content=question), MemoryTurn(role="assistant", content=text)]
        return text, guide

    def test_a_long_chain_keeps_the_list_and_what_is_done(self):
        keys = [self.say(question)[1].steps[0].key for question in ("lanjut", "sudah", "sudah", "sudah")]
        self.assertEqual(keys, ["a", "b", "c", "d"])
        self.assertIn("Semua langkah sudah kamu baca", self.say("sudah")[0])
        # Six questions now: the original list is out of the five-question history.
        self.assertEqual(self.say("kembali ke b", show="b", forced=True)[1].steps[0].key, "b")
        self.assertIn("Semua langkah sudah kamu baca", self.say("sudah")[0])

    def test_a_progress_message_without_a_list_says_so_instead_of_guessing(self):
        shown = MemoryTurn(role="assistant", content="Berikutnya: **d. Baca Panduan Setup Lokal v2** (x): y")
        memory = [MemoryTurn(role="user", content="sudah"), shown]
        for question in ("sudah", "lanjut"):
            self.assertIn("tidak ada di riwayat", continuation(question, memory, self.scope, BASE, "id")[0])
        self.assertIn("no longer in", continuation("done", memory, self.scope, BASE, "en")[0])
        # No step was shown: nothing to say, the model answers as before.
        chat = [MemoryTurn(role="user", content="halo"), MemoryTurn(role="assistant", content="Halo!")]
        self.assertIsNone(continuation("sudah", chat, self.scope, BASE, "id"))

    def test_foreign_or_corrupt_state_is_ignored(self):
        line = "a. **X** ([secret.pdf](http://testserver/api/v1/files/c9/secret.pdf#page=1), hlm 1): y"
        for state in (_carry([line], set()), '\n\n[lens-list]: # "bm90IGpzb24"', '\n\n[lens-list]: # "!!!"'):
            memory = [MemoryTurn(role="user", content="lanjut"),
                      MemoryTurn(role="assistant", content="Berikutnya: **a. X** (y): z" + state)]
            found = continuation("lanjut", memory, self.scope, BASE, "id")
            self.assertTrue(found is None or "secret.pdf" not in found[0], state)

    def test_the_state_alone_is_a_list_to_continue(self):
        text = self.say("lanjut")[0]
        self.assertTrue(has_step_list([MemoryTurn(role="assistant", content=text)]))

    def pinned_memory(self, ctx=None):
        """The list reply with its state, then six unrelated exchanges: the list is out of the five-question history."""
        memory = [MemoryTurn(role="user", content="Aku baru masuk project Proyek Klien Y"),
                  MemoryTurn(role="assistant", content=carry_list(reading_list(), ctx))]
        for i in range(6):
            memory += [MemoryTurn(role="user", content=f"pertanyaan {i}"), MemoryTurn(role="assistant", content=f"jawaban {i}")]
        return memory

    def test_a_list_older_than_the_history_is_still_continued_while_side_questions_pile_up(self):
        memory = self.pinned_memory()
        self.assertTrue(has_step_list(memory))
        self.assertEqual(continuation("lanjut", memory, self.scope, BASE, "id")[1].steps[0].key, "a")

    def test_the_pinned_list_reply_is_for_continuation_only_not_for_a_prompt(self):
        memory = [MemoryTurn(role="assistant", content=carry_list(reading_list(), None))]
        for i in range(5):
            memory += [MemoryTurn(role="user", content=f"q{i}"), MemoryTurn(role="assistant", content=f"a{i}")]
        clamped = _clamp_memory(memory)
        self.assertEqual([t["role"] for t in clamped][:2], ["user", "assistant"])
        self.assertFalse(any("Baca" in t["content"] for t in clamped))

    def test_role_and_part_ride_with_the_list(self):
        ctx = {"folder": "Delivery › Proyek Klien Y", "role": "Developer", "part": "core-api"}
        self.memory = self.pinned_memory(ctx)
        self.assertEqual(carried_answers(self.memory, "Delivery › Proyek Klien Y"), {"role": "Developer", "part": "core-api"})
        self.assertEqual(carried_answers(self.memory, "Backup & Pemulihan"), {})
        self.say("lanjut")
        self.assertEqual(carried_answers(self.memory, "Delivery › Proyek Klien Y")["role"], "Developer")

    def test_a_forged_context_is_ignored(self):
        line = "a. **X** ([brief.pdf](http://testserver/api/v1/files/c1/Project%20Brief.pdf#page=1), hlm 1): y"
        for ctx in ({"folder": "F", "role": "x" * 500}, {"folder": "F", "role": ["a"]}, {"folder": 3}):
            memory = [MemoryTurn(role="assistant", content="Berikutnya: **a. X** (y): z" + _carry([line], set(), ctx))]
            self.assertEqual(carried_answers(memory, "F"), {}, ctx)

    def test_a_list_answer_carries_its_state_and_a_nearest_only_one_does_not(self):
        self.assertTrue(carries_list_state(carry_list(reading_list(), None)))
        self.assertFalse(carries_list_state(reading_list()))
        nearest = "1. **x.pdf** ([x.pdf](http://testserver/api/v1/files/c1/x.pdf), hlm 1): " + NEAREST_WHY["id"]
        self.assertEqual(carry_list(nearest, None), nearest)

    def test_the_state_never_reaches_a_model_prompt_or_the_card_text(self):
        text, guide = self.say("lanjut")
        self.assertIn("[lens-list]", text)
        self.assertNotIn("lens-list", strip_list_state(text))
        self.assertNotIn("lens-list", guide.lead)
        self.assertFalse(any("lens-list" in turn["content"] for turn in _clamp_memory(self.memory)))


class FacetPromptTests(unittest.TestCase):
    def test_the_role_question_tells_the_model_a_role_is_never_a_persons_name(self):
        # Manual test, 26 Nov 2026: the chips were Developer, QA, Sari, Rudi, Bayu (names from the kickoff
        # minutes). GPT probe: 2 of 8 runs before this rule, 0 of 32 after.
        from unittest.mock import MagicMock, patch
        from services.rag.processor import processor
        llm = MagicMock()
        llm.bind_tools.return_value.invoke.return_value = MagicMock(tool_calls=[])
        with patch.object(processor, "get_llm", return_value=(llm, "x")):
            processor.extract_facets("FOLDER: Proyek")
        prompt = llm.bind_tools.return_value.invoke.call_args[0][0]
        self.assertIn("never a person's name", prompt)


class ParserFindingTests(unittest.TestCase):
    def test_missing_line_drops_extra_columns(self):
        result = parse_guide_answer("1. Baca brief | Alasan | 10 [S1]\nTIDAK ADA: langkah kerja core-api | 0",
                                    "start", [ref("S1", "Project Brief.pdf")], [], "id", BASE)
        self.assertEqual(result.notices[0].text, "Belum ada dokumen: langkah kerja core-api")
        self.assertNotIn("| 0", result.text)

    def test_minutes_written_without_their_separator_leave_the_reason(self):
        result = parse_guide_answer("1. Baca Notulen Kickoff | Pertanyaan akses ke Bayu 10 [S1]",
                                    "start", [ref("S1", "Notulen Kickoff.pdf")], [], "id", BASE)
        self.assertEqual(result.steps[0].why, "Pertanyaan akses ke Bayu")
        self.assertEqual(result.steps[0].minutes, 10)
        page = parse_guide_answer("1. Baca Arsitektur | Fokus ke halaman 4 [S1]",
                                  "start", [ref("S1", "Arsitektur Sistem.pdf")], [], "id", BASE)
        self.assertEqual((page.steps[0].why, page.steps[0].minutes), ("Fokus ke halaman 4", None))

    def test_only_a_nearest_document_gets_no_follow_these_steps_lead(self):
        # Manual test: "gimana cara minta akses database staging?" read as do.
        refs = [ref("S1", "Notulen Kickoff.pdf"), ref("S2", "Panduan Setup Lokal v1.pdf", 1)]
        docs = [{"collection_id": "c1", "file_name": "Panduan Setup Lokal v1.pdf"},
                {"collection_id": "c1", "file_name": "Panduan Setup Lokal v2.pdf"}]
        result = parse_guide_answer("TIDAK ADA: cara minta akses staging\nTERDEKAT: notulen | Pemegang akses [S1]",
                                    "do", refs, docs, "id", BASE)
        self.assertNotIn("Ikuti langkah", result.text)
        self.assertEqual([n.kind for n in result.notices], ["missing"])

    def test_older_numbered_version_is_never_recommended(self):
        docs = [{"collection_id": "c1", "file_name": "Panduan Setup Lokal v1.pdf"},
                {"collection_id": "c1", "file_name": "Panduan Setup Lokal v2.pdf"}]
        refs = [ref("S1", "Panduan Setup Lokal v2.pdf"), ref("S2", "Panduan Setup Lokal v1.pdf", 1)]
        raw = "1. Baca setup | Siapkan lokal | 15 [S1]\n2. Baca setup lama | Sejarah | 5 [S2]\nNANTI: setup [D1]; setup [D2]"
        result = parse_guide_answer(raw, "start", refs, docs, "id", BASE)
        self.assertEqual([s.file for s in result.steps], ["Panduan Setup Lokal v2.pdf"])
        self.assertNotIn("Panduan Setup Lokal v1.pdf", result.lead)
        self.assertEqual([n.kind for n in result.notices], ["old_version"])
        self.assertIn("Panduan Setup Lokal v1.pdf", result.notices[0].text)

    def test_an_old_numbered_version_against_its_newer_one_is_not_a_conflict(self):
        # Manual test: "gimana cara minta akses database staging?" came back as only a
        # conflict box, setup v1 vs setup v2. v2 replaces v1, so nothing is in conflict.
        refs = [ref("S1", "Panduan Setup Lokal v1.pdf", 1), ref("S2", "Panduan Setup Lokal v2.pdf")]
        raw = ("TIDAK ADA: cara minta akses database staging\n"
               "BERTENTANGAN: setup lama mengunduh dump produksi [S1] vs setup baru tanpa akses staging [S2]")
        for intent in ("fact", "do"):
            result = parse_guide_answer(raw, intent, refs, [], "id", BASE)
            self.assertNotIn("conflict", [n.kind for n in result.notices], intent)
            self.assertNotIn("Panduan Setup Lokal v1.pdf", result.text, intent)

    def test_total_time_over_an_hour_reads_like_the_mockup(self):
        raw = "1. Baca brief | Tujuan | 60 [S1]\n2. Baca notulen | Siapa pegang apa | 40 [S2]"
        result = parse_guide_answer(raw, "start", [ref("S1", "Project Brief.pdf"), ref("S2", "Notulen Kickoff.pdf")],
                                    [], "id", BASE)
        self.assertIn("Perkiraan total: 1 jam 40 mnt.", result.lead)

    def test_a_conflict_notice_starts_with_a_capital(self):
        raw = "BERTENTANGAN: catatan rilis menyebut PM [S1] vs P03 menyebut Lead SD [S2]\n1. Isi F03 | Wajib | [S2]"
        result = parse_guide_answer(raw, "do", [ref("S1", "Catatan Rilis Tim.pdf"), ref("S2", "MLT-DEV P03.pdf")],
                                    [], "id", BASE)
        self.assertTrue(result.notices[0].text.startswith("Catatan rilis"))

    def test_the_explicit_no_conflict_line_is_accepted_and_shows_nothing(self):
        # The do prompt makes the model answer the conflict check on its first line (GPT probe, 26 Nov
        # 2026: conflicts reported 2 of 6 before, 8 of 8 after; 0 of 5 false ones on consistent sources).
        refs = [ref("S1", "MLT-DEV P03.pdf")]
        for first in ("KONFLIK: tidak ada", "NO CONFLICT: none"):
            result = parse_guide_answer(first + "\n1. Isi F03 | Wajib | [S1]", "do", refs, [], "id", BASE)
            self.assertEqual([s.title for s in result.steps], ["Isi F03"], first)
            self.assertFalse(result.notices, first)
            self.assertNotIn("KONFLIK", result.text, first)

    def test_a_conflict_is_a_conflict_not_an_old_version_when_the_files_are_not_numbered_versions(self):
        refs = [ref("S1", "Catatan Rilis Tim.pdf", 1), ref("S2", "MLT-DEV P03.pdf", 1)]
        raw = "BERTENTANGAN: disetujui PM [S1] vs disetujui Lead SD [S2]\n1. Rilis lewat pipeline | Wajib | [S2]"
        result = parse_guide_answer(raw, "do", refs, [], "id", BASE)
        self.assertEqual([n.kind for n in result.notices], ["conflict"])

    def test_document_list_entries_have_no_fake_location(self):
        docs = [{"collection_id": "c1", "file_name": "Notulen Kickoff.pdf"}]
        result = parse_guide_answer("1. Baca brief | Alasan | 10 [S1]\nNANTI: notulen [D1]",
                                    "start", [ref("S1", "Project Brief.pdf")], docs, "id", BASE)
        self.assertIn("Notulen Kickoff.pdf", result.lead)
        self.assertNotIn("Notulen Kickoff.pdf), dokumen", result.lead)
        self.assertNotIn(", dokumen", result.lead)

    def test_nearest_documents_in_a_fact_answer_are_not_written_as_a_reading_list(self):
        result = parse_guide_answer("TIDAK ADA: cara minta akses staging\nTERDEKAT: setup | Data contoh [S1]",
                                    "fact", [ref("S1", "Panduan Setup Lokal v2.pdf")], [], "id", BASE)
        self.assertTrue(result.steps)
        self.assertNotRegex(result.text, r"(?m)^1\. \*\*")
        self.assertIn("Panduan Setup Lokal v2.pdf", result.text)


if __name__ == "__main__":
    unittest.main()
