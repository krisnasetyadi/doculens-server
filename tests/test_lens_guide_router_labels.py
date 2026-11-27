"""MS-1183: the router reads what a message asks for (a preset label), the
server still leads the flow. Messages the word lists miss, from the manual test
(26 Nov 2026). The model is faked: what it would label is given per test."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from models import MemoryTurn
from services import agnostic as agnostic_service
from services.lens_guide.answer import SourceRef, parse_guide_answer, source_detail
from services.lens_guide.continuation import carry_list
from services.rag.processor import processor
from tests.fakes import fake_llm
from tests.test_lens_guide_find import LensGuideEndpoint

ROLES = ["Developer", "QA"]
PARTS = ["web", "core-api"]
BASE = "http://testserver"


def reading_list():
    files = [("c2", "brief.pdf"), ("c1", "a.pdf"), ("c1", "b.pdf")]
    raw = "\n".join(f"{i}. Baca {name} | Alasan {i} | 10 [S{i}]" for i, (_, name) in enumerate(files, 1))
    refs = [SourceRef(f"S{i}", source_detail(name, cid, 1, BASE)) for i, (cid, name) in enumerate(files, 1)]
    return parse_guide_answer(raw, "start", refs, [], "id", BASE).text


class RouterLabelTests(LensGuideEndpoint):
    def setUp(self):
        super().setUp()
        agnostic_service._FACETS.clear()
        self.addCleanup(agnostic_service._FACETS.clear)
        passage = SimpleNamespace(page_content="Peran: Developer, QA. Bagian: web, core-api.", metadata={"collection_id": "c1"})
        patch.object(processor, "search_across_collections", return_value=[passage]).start()
        patch.object(processor, "extract_facets", return_value=(ROLES, PARTS, 30)).start()

    def say(self, question, router, memory=None):
        with patch.object(processor, "get_llm", return_value=(router, "gemini/gemini-test")), \
                patch.object(processor, "generate_hybrid_answer", return_value=("Jawaban", "gemini/x", {})) as answer:
            response = self.client.post("/api/v1/agnostic/query", json={"question": question, "memory": memory or []})
        self.assertEqual(response.status_code, 200, response.text)
        task = answer.call_args[0][8] if answer.called else None
        return response.json(), task

    @staticmethod
    def turn(memory, question, data):
        return memory + [{"role": "user", "content": question}, {"role": "assistant", "content": data["answer"]}]

    def tool_names(self, router):
        return [tool["name"] for tool in router.bind_tools.call_args[0][0]]

    def test_a_start_the_word_list_misses_gets_the_folder_and_the_role_question(self):
        router = fake_llm(tool_name="find_folder", tool_args={"name": "Delivery", "guide": "start"})
        data, _ = self.say("aku masuk project Delivery", router)
        self.assertEqual(data["guide"]["question"]["options"], ROLES)
        self.assertEqual(data["guide"]["folder"]["path"], ["Delivery"])

    def test_a_learning_request_keeps_its_kind_through_confirmation_and_questions(self):
        router = fake_llm(tool_name="find_folder", tool_args={"name": "Delivery Klien Y", "guide": "learn"})
        first, _ = self.say("ajarin aku soal delivery klien y", router)
        self.assertEqual(first["guide"]["folder"]["path"], ["Delivery", "Proyek Klien Y"])
        memory = self.turn([], "ajarin aku soal delivery klien y", first)

        unused = fake_llm("tidak dipakai")
        picked, _ = self.say("Ya, pakai folder Delivery › Proyek Klien Y", unused, memory)
        unused.invoke.assert_not_called()
        self.assertEqual(picked["guide"]["question"]["options"], ROLES)
        memory = self.turn(memory, "Ya, pakai folder Delivery › Proyek Klien Y", picked)

        part, _ = self.say("Developer", unused, memory)
        memory = self.turn(memory, "Developer", part)
        _, task = self.say("core-api", unused, memory)
        self.assertEqual(task, "guide_learn")

    def test_a_task_without_a_folder_becomes_steps_over_every_document(self):
        router = fake_llm(tool_args={"query": "cara deploy ke produksi", "task": "answer", "guide": "do"})
        _, task = self.say("cara deploy ke prod gimana", router)
        self.assertEqual(task, "guide_do")
        self.assertEqual(sorted(self.hybrid_search.call_args[0][1]), ["c1", "c2", "c3", "c4"])

    def test_an_ordinary_question_stays_an_ordinary_answer(self):
        router = fake_llm(tool_args={"query": "tujuan project", "task": "answer", "guide": "none"})
        _, task = self.say("apa tujuan project ini?", router)
        self.assertEqual(task, "answer")

    def test_an_ordinary_folder_lookup_gets_no_guidance_note(self):
        data, _ = self.say("list dokumen delivery klien y",
                           fake_llm(tool_name="find_folder", tool_args={"name": "Delivery Klien Y"}))
        self.assertEqual(data["answer"],
                         "Saya menemukan folder **Delivery › Proyek Klien Y**. Benar ini folder yang kamu maksud?")

    def test_progress_the_word_list_misses_is_read_by_the_router(self):
        memory = [{"role": "user", "content": "aku baru masuk project Delivery"},
                  {"role": "assistant", "content": reading_list()}]
        router = fake_llm(tool_name="continue_list", tool_args={"read": ["a", "b"]})
        data, task = self.say("a sama b kelar", router, memory)
        self.assertIn("continue_list", self.tool_names(router))
        self.assertIsNone(task)
        self.assertEqual(data["guide"]["steps"][0]["key"], "c")

        back = fake_llm(tool_name="continue_list", tool_args={"show": "a"})
        data, _ = self.say("kembali ke a", back, memory)
        self.assertEqual(data["guide"]["steps"][0]["key"], "a")

    def test_a_progress_reply_carries_the_list_for_the_next_turns_but_not_in_its_card(self):
        # Manual test, 26 Nov 2026: after six questions the list left the history and "sudah" went to the model.
        memory = [{"role": "user", "content": "aku baru masuk project Delivery"},
                  {"role": "assistant", "content": reading_list()}]
        router = fake_llm(tool_name="continue_list", tool_args={"read": ["a", "b"]})
        data, _ = self.say("a sama b kelar", router, memory)
        self.assertIn("[lens-list]", data["answer"])
        self.assertNotIn("lens-list", data["guide"]["lead"])
        # Only that reply (step c) is in the history now: "sudah" still knows the list and that a, b are done.
        unused = fake_llm("tidak dipakai")
        again, _ = self.say("sudah", unused, [{"role": "assistant", "content": data["answer"]}])
        unused.invoke.assert_not_called()
        self.assertIn("Semua langkah sudah kamu baca", again["answer"])

    def test_a_task_the_model_mistook_for_a_folder_still_becomes_steps(self):
        # Real router, 26 Nov 2026: "gimana caranya rilis ke produksi?" -> find_folder("rilis ke produksi", do).
        router = fake_llm(tool_name="find_folder", tool_args={"name": "rilis ke produksi", "guide": "do"})
        data, task = self.say("gimana caranya rilis ke produksi?", router)
        self.assertEqual(task, "guide_do")
        self.assertNotIn("belum menemukan folder", data["answer"])

    def test_keys_the_user_did_not_write_are_not_taken_as_read(self):
        # Real router: "abis itu apa?" -> continue_list(read=["a", "b", "c"]).
        memory = [{"role": "user", "content": "aku baru masuk project Delivery"},
                  {"role": "assistant", "content": reading_list()}]
        router = fake_llm(tool_name="continue_list", tool_args={"read": ["a", "b", "c"], "show": "b"})
        data, _ = self.say("abis itu apa?", router, memory)
        self.assertEqual(data["guide"]["steps"][0]["key"], "a")

    def test_a_follow_up_mislabelled_as_start_continues_the_list(self):
        # Manual test, 26 Nov 2026: "abis itu apa?" came back as search(folder, guide=start) and the
        # role question started over. With a list shown and no new folder named, it is progress.
        memory = [{"role": "user", "content": "aku masuk project Delivery"},
                  {"role": "assistant", "content": reading_list()}]
        router = fake_llm(tool_args={"query": "langkah berikutnya", "task": "answer", "folder": "Delivery",
                                     "guide": "start"})
        data, task = self.say("abis itu apa?", router, memory)
        self.assertIsNone(task)
        self.assertIsNone(data["guide"].get("question"))
        self.assertEqual(data["guide"]["steps"][0]["key"], "a")

    def guided_memory(self):
        """A whole guidance: request, role and part answered, reading list shown (Delivery)."""
        memory = []
        start, _ = self.say("aku baru masuk project Delivery", fake_llm("tidak dipakai"))
        memory = self.turn(memory, "aku baru masuk project Delivery", start)
        part, _ = self.say("Developer", fake_llm("tidak dipakai"), memory)
        memory = self.turn(memory, "Developer", part)
        return memory + [{"role": "user", "content": "core-api"}, {"role": "assistant", "content": reading_list()}]

    def test_a_new_task_question_inside_an_active_guide_keeps_role_and_part(self):
        # Manual test, 26 Nov 2026: "gimana cara minta akses database staging?" after the list
        # asked the role again. It is a new question in the same guidance: answered as steps,
        # with the role and part already given, and never as "next step a".
        memory = self.guided_memory()
        router = fake_llm(tool_args={"query": "cara minta akses database staging", "task": "answer",
                                     "folder": "Delivery", "guide": "do"})
        data, task = self.say("gimana cara minta akses database staging?", router, memory)
        self.assertEqual(task, "guide_do")
        self.assertNotIn("question", {k for k, v in (data.get("guide") or {}).items() if v})
        self.assertIn("Peran saya: Developer.", self.hybrid_search.call_args[0][0])

    def test_role_and_part_survive_a_list_older_than_the_history(self):
        # The list and the role and part answers left the five-question history; the carried state has them.
        ctx = {"folder": "Delivery", "role": "Developer", "part": "core-api"}
        memory = [{"role": "assistant", "content": carry_list(reading_list(), ctx)}]
        for i in range(5):
            memory += [{"role": "user", "content": f"pertanyaan {i}"}, {"role": "assistant", "content": f"jawaban {i}"}]
        router = fake_llm(tool_args={"query": "cara minta akses database staging", "task": "answer",
                                     "folder": "Delivery", "guide": "do"})
        data, task = self.say("gimana cara minta akses database staging?", router, memory)
        self.assertEqual(task, "guide_do")
        self.assertNotIn("question", {k for k, v in (data.get("guide") or {}).items() if v})
        searched = self.hybrid_search.call_args[0][0]
        self.assertIn("Peran saya: Developer.", searched)
        self.assertIn("Bagian yang saya pegang: core-api.", searched)

    def test_naming_another_folder_after_a_list_starts_over(self):
        memory = [{"role": "user", "content": "aku masuk project Delivery"},
                  {"role": "assistant", "content": reading_list()}]
        router = fake_llm(tool_name="find_folder", tool_args={"name": "Backend", "guide": "start"})
        data, _ = self.say("sekarang aku pindah ke folder Backend", router, memory)
        self.assertEqual(data["guide"]["question"]["options"], ROLES)

    def test_continue_is_only_offered_when_there_is_a_list(self):
        router = fake_llm("Halo!")
        self.say("halo", router)
        self.assertNotIn("continue_list", self.tool_names(router))


if __name__ == "__main__":
    unittest.main()
