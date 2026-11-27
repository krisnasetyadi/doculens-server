"""MS-1182: the server-led context questions, one test per row of the scenario
table (~/doculens/docs/MS-1182-SCENARIOS.md), end to end through
POST /agnostic/query. The model is faked: what it would read out of the
documents is given per test, and the router must not be called where the
server decides."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services import agnostic as agnostic_service
from services.lens_guide.clarify import (
    Facets, clean_options, guidance_request, is_guidance, known_answers, match_option,
)
from services.rag.processor import processor
from tests.fakes import fake_llm
from tests.test_agnostic_stream import FakeModel
from tests.test_lens_guide_find import LensGuideEndpoint

ROLES = ["Developer", "QA"]
PARTS = ["web", "core-api"]
CARD_Y = [
    {"role": "user", "content": "aku baru masuk Delivery Klien Y"},
    {"role": "assistant", "content": "Saya menemukan folder **Delivery › Proyek Klien Y**. Benar ini folder yang kamu maksud?"},
]
PICKED_Y = CARD_Y + [
    {"role": "user", "content": "Ya, pakai folder Delivery › Proyek Klien Y"},
    {"role": "assistant", "content": "Kamu perannya apa di project Proyek Klien Y?"},
]
ROLE_Y = PICKED_Y + [
    {"role": "user", "content": "Developer"},
    {"role": "assistant", "content": "Kamu pegang bagian yang mana di project Proyek Klien Y?"},
]
GUIDED_Y = ROLE_Y + [
    {"role": "user", "content": "core-api"},
    {"role": "assistant", "content": "Baca Arsitektur hlm 4, lalu Brief hlm 1."},
]


class ScenarioTests(LensGuideEndpoint):
    def setUp(self):
        super().setUp()
        agnostic_service._FACETS.clear()
        self.addCleanup(agnostic_service._FACETS.clear)
        passage = SimpleNamespace(page_content="Peran: Developer, QA. Bagian: web, core-api.", metadata={"collection_id": "c1"})
        self.brief = patch.object(processor, "search_across_collections", return_value=[passage]).start()
        self.facets = patch.object(processor, "extract_facets", return_value=(ROLES, PARTS, 30)).start()
        self.router = fake_llm("tidak dipakai")

    def say(self, question, memory=None, router=None, **body):
        return self.ask(question, router or self.router, memory=memory or [], **body)

    def searched(self):
        return self.hybrid_search.call_args[0][0], sorted(self.hybrid_search.call_args[0][1])

    def no_router(self):
        self.router.invoke.assert_not_called()

    # A: a folder named by its whole name, in a guidance message

    def test_a1_card_and_role_question_in_one_turn(self):
        data = self.say("aku baru masuk project Delivery")
        self.assertEqual(data["answer"], "Saya menemukan folder **Delivery**. Kamu perannya apa di project Delivery?")
        self.assertEqual(data["guide"]["folder"]["path"], ["Delivery"])
        self.assertEqual(data["guide"]["question"]["options"], ROLES)
        self.no_router()
        query, ids, _ = self.brief.call_args[0]
        self.assertEqual(sorted(ids), ["c1", "c2"])
        self.assertIn("Peran: Developer, QA.", self.facets.call_args[0][0])

    def test_a1_the_names_are_read_once_per_folder(self):
        self.say("aku baru masuk project Delivery")
        self.say("aku baru masuk project Delivery")
        self.facets.assert_called_once()

    def test_a2_a_role_in_the_request_skips_to_the_part(self):
        self.facets.return_value = (["Software Developer", "QA Engineer"], PARTS, 30)
        data = self.say("aku developer baru di project Delivery")
        self.assertEqual(data["answer"], "Saya menemukan folder **Delivery**. Kamu pegang bagian yang mana di project Delivery?")
        self.assertEqual(data["guide"]["question"]["options"], PARTS)

    def test_a3_documents_that_tell_nothing_apart_get_the_plain_card(self):
        self.facets.return_value = ([], [], 30)
        data = self.say("aku baru masuk project Delivery")
        self.assertEqual(data["answer"], "Saya menemukan folder **Delivery**. Benar ini folder yang kamu maksud?")
        self.assertIsNone(data["guide"]["question"])
        self.no_router()

    def test_a4_an_empty_folder_gets_the_card_with_zero_files(self):
        data = self.say("aku baru masuk project Kosong")
        self.assertEqual(data["guide"]["folder"]["file_count"], 0)
        self.assertTrue(data["answer"].endswith("Benar ini folder yang kamu maksud?"))
        self.facets.assert_not_called()

    def test_a5_without_a_guidance_wording_the_router_decides_as_before(self):
        router = self.find("Delivery")
        self.say("tolong ringkas project Delivery", router=router)
        router.invoke.assert_called_once()
        self.facets.assert_not_called()

    def test_a_whole_name_only_the_router_spotted_is_confirmed_with_the_question(self):
        data = self.say("aku baru masuk Backend", router=self.find("Backend"))
        self.assertEqual(data["guide"]["folder"]["path"], ["Delivery", "Proyek Klien Y", "Backend"])
        self.assertEqual(data["guide"]["question"]["options"], ROLES)

    # B, C, D: after the card or the list

    def test_b_nothing_to_ask_searches_the_folder_with_the_request(self):
        self.facets.return_value = ([], [], 30)
        data = self.say("Ya, pakai folder Delivery › Proyek Klien Y", CARD_Y)
        self.assertEqual(data["answer"], "Jawaban")
        self.assertEqual(self.searched(), ("aku baru masuk Delivery Klien Y. Dari dokumen di folder ini, apa yang perlu saya baca atau lakukan dulu, berurutan, dan kenapa?", ["c1", "c2"]))
        self.assertEqual(data["guide"]["context"], [{"label": "Project", "value": "Delivery › Proyek Klien Y", "options": []}])
        self.no_router()

    def test_c_the_card_confirmed_then_the_role(self):
        data = self.say("Ya, pakai folder Delivery › Proyek Klien Y", CARD_Y)
        self.assertEqual(data["answer"], "Kamu perannya apa di project Proyek Klien Y?")
        self.assertEqual(data["guide"]["question"]["options"], ROLES)
        self.assertIsNone(data["guide"]["folder"])
        self.no_router()

    def test_d_a_folder_picked_from_several_then_the_role(self):
        memory = [
            {"role": "user", "content": "aku baru masuk Proyek Klien Y"},
            {"role": "assistant", "content": 'Ada 2 folder yang cocok dengan "Proyek Klien Y" dan bisa kamu buka. Yang mana yang kamu maksud?'},
        ]
        data = self.say("Pakai folder Delivery › Proyek Klien Y", memory)
        self.assertEqual(data["answer"], "Kamu perannya apa di project Proyek Klien Y?")

    def test_e_no_folder_matches_as_before(self):
        data = self.say("aku baru masuk project Zeta", router=self.find("Zeta"))
        self.assertTrue(data["guide"]["missing_source"])

    # F, G, H: the answers

    def test_f1_a_role_chip_then_the_part(self):
        data = self.say("Developer", PICKED_Y)
        self.assertEqual(data["answer"], "Kamu pegang bagian yang mana di project Proyek Klien Y?")
        self.assertEqual(data["guide"]["question"]["options"], PARTS)
        self.no_router()

    def test_f2_a_role_in_other_words(self):
        data = self.say("aku developer", PICKED_Y)
        self.assertEqual(data["answer"], "Kamu pegang bagian yang mana di project Proyek Klien Y?")

    def test_f3_an_answer_outside_the_choices_is_kept_as_written(self):
        self.say("Belum tahu", PICKED_Y)
        data = self.say("web", PICKED_Y + [
            {"role": "user", "content": "Belum tahu"},
            {"role": "assistant", "content": "Kamu pegang bagian yang mana di project Proyek Klien Y?"},
        ])
        self.assertEqual([c["value"] for c in data["guide"]["context"]], ["Delivery › Proyek Klien Y", "Belum tahu", "web"])

    def test_g_everything_known_searches_and_shows_the_context(self):
        data = self.say("core-api", ROLE_Y)
        self.assertEqual(data["answer"], "Jawaban")
        self.assertEqual(self.searched(), (
            "aku baru masuk Delivery Klien Y. Dari dokumen di folder ini, apa yang perlu saya baca atau lakukan dulu, berurutan, dan kenapa? Peran saya: Developer. Bagian yang saya pegang: core-api.",
            ["c1", "c2"],
        ))
        self.assertEqual(data["guide"]["context"], [
            {"label": "Project", "value": "Delivery › Proyek Klien Y", "options": []},
            {"label": "Peran", "value": "Developer", "options": ROLES},
            {"label": "Bagian", "value": "core-api", "options": PARTS},
        ])
        self.no_router()

    def test_g_the_streamed_answer_carries_the_context_in_done(self):
        with patch.object(processor, "get_llm", return_value=(FakeModel(), "gemini/x")):
            response = self.client.post("/api/v1/agnostic/query?stream=true", json={"question": "core-api", "memory": ROLE_Y})
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[-1]["type"], "done", events)
        self.assertEqual([c["value"] for c in events[-1]["result"]["guide"]["context"]], ["Delivery › Proyek Klien Y", "Developer", "core-api"])

    def test_h_ubah_searches_again_with_the_new_value(self):
        data = self.say("Ganti Peran jadi QA di folder Delivery › Proyek Klien Y", GUIDED_Y)
        self.assertEqual(self.searched()[0],
                         "aku baru masuk Delivery Klien Y. Dari dokumen di folder ini, apa yang perlu saya baca atau lakukan dulu, berurutan, dan kenapa? Peran saya: QA. Bagian yang saya pegang: core-api.")
        self.assertEqual(data["guide"]["context"][1]["value"], "QA")
        self.no_router()

    def test_h_ubah_still_works_once_the_request_is_out_of_the_clamped_history(self):
        # The server keeps the last 5 questions (MS-237): three follow-ups push the
        # request and the folder card out; the folder in the Ubah message and the
        # part's reply still in view keep the guidance going.
        long_chat = GUIDED_Y + [
            {"role": "user", "content": "berapa lama setup lokal?"}, {"role": "assistant", "content": "Sekitar 30 menit."},
            {"role": "user", "content": "siapa yang review PR?"}, {"role": "assistant", "content": "QA."},
            {"role": "user", "content": "apa itu golden set?"}, {"role": "assistant", "content": "Skrip uji."},
        ]
        data = self.say("Ganti Peran jadi QA di folder Delivery › Proyek Klien Y", long_chat)
        self.assertEqual(self.searched()[0], (
            "Arahkan saya untuk mulai di project Proyek Klien Y. Dari dokumen di folder ini, apa yang perlu saya baca atau lakukan dulu, berurutan, dan kenapa? Peran saya: QA. Bagian yang saya pegang: core-api."
        ))
        self.assertEqual([c["value"] for c in data["guide"]["context"]], ["Delivery › Proyek Klien Y", "QA", "core-api"])
        self.no_router()

    # I, J, L: left to the router

    def test_i_a_fact_question_goes_to_the_router(self):
        router = fake_llm(tool_args={"query": "pemegang jadwal rilis", "folder": "Delivery › Proyek Klien Y", "task": "answer"})
        self.say("siapa pemegang jadwal rilis?", GUIDED_Y, router=router)
        router.invoke.assert_called_once()

    def test_j_turning_the_folder_down_goes_to_the_router(self):
        router = fake_llm("Project atau folder mana yang kamu maksud?")
        self.say("Bukan folder ini", CARD_Y, router=router)
        router.invoke.assert_called_once()
        self.facets.assert_not_called()

    def test_a5_picking_the_card_of_a_plain_request_goes_to_the_router(self):
        memory = [{"role": "user", "content": "tolong ringkas Delivery Klien Y"}] + CARD_Y[1:]
        router = fake_llm(tool_args={"query": "ringkasan", "folder": "Delivery › Proyek Klien Y", "task": "summarize"})
        self.say("Ya, pakai folder Delivery › Proyek Klien Y", memory, router=router)
        router.invoke.assert_called_once()
        self.facets.assert_not_called()

    def test_l_a_guidance_wording_without_a_project_goes_to_the_router(self):
        router = fake_llm("Project apa?")
        self.say("Saya baru di project", router=router)
        router.invoke.assert_called_once()

    # K, M, O, Q

    def test_k_picking_an_empty_folder(self):
        memory = [{"role": "user", "content": "aku baru masuk project Kosong"},
                  {"role": "assistant", "content": "Saya menemukan folder **Delivery › Kosong**. Benar ini folder yang kamu maksud?"}]
        data = self.say("Ya, pakai folder Delivery › Kosong", memory)
        self.assertIn("belum berisi dokumen aktif", data["answer"])
        self.no_router()

    def test_m_a_new_project_starts_over(self):
        data = self.say("aku pindah ke project Delivery", GUIDED_Y)
        self.assertEqual(data["answer"], "Saya menemukan folder **Delivery**. Kamu perannya apa di project Delivery?")

    def test_o_a_failed_read_asks_nothing_and_is_tried_again(self):
        self.facets.return_value = ([], [], 0)
        self.say("aku baru masuk project Delivery")
        data = self.say("aku baru masuk project Delivery")
        self.assertIsNone(data["guide"]["question"])
        self.assertEqual(self.facets.call_count, 2)

    def test_o_a_read_without_a_report_is_not_kept(self):
        self.facets.return_value = (None, None, 30)
        self.say("aku baru masuk project Delivery")
        self.say("aku baru masuk project Delivery")
        self.assertEqual(self.facets.call_count, 2)

    def test_q_english(self):
        data = self.say("I just joined project Delivery")
        self.assertEqual(data["answer"], "I found the folder **Delivery**. What is your role in Delivery?")


class ReadingTests(unittest.TestCase):
    def test_guidance_wordings(self):
        for text in ("Saya baru di project Atlas", "Saya mau mempelajari modul X", "Saya mau mengerjakan rilis",
                     "aku baru masuk project A", "mulai dari mana?", "I just joined project A"):
            self.assertTrue(is_guidance(text), text)
        for text in ("aku mau belajar modul backup", "I'm new to the team", "pelajari modul backup dulu"):
            self.assertTrue(is_guidance(text), text)
        for text in ("ringkas dokumen Atlas", "siapa pemegang jadwal rilis?", "Developer", "Ya, pakai folder A",
                     "what's new in the release notes?", "anything new on the roadmap?",
                     "dokumen onboarding ada di mana?", "jadwal belajar bareng kapan?",
                     "pindah ke halaman 3", "apa yang dipelajari di bab 3?"):
            self.assertFalse(is_guidance(text), text)

    def test_match_option(self):
        options = ["Software Developer", "QA Engineer", "Data Engineer"]
        self.assertEqual(match_option("aku developer", options), "Software Developer")
        self.assertEqual(match_option("QA", options), "QA Engineer")
        self.assertIsNone(match_option("engineer", options))
        self.assertIsNone(match_option("designer", options))
        self.assertEqual(match_option("web", ["web", "core-api"]), "web")
        self.assertIsNone(match_option("website baru", ["web", "core-api"]))

    def test_only_messages_since_the_request_count(self):
        facets = Facets(roles=ROLES, parts=PARTS)
        start = guidance_request("aku pindah ke project Delivery", GUIDED_Y)
        self.assertEqual(start, len(GUIDED_Y))
        self.assertEqual(known_answers(facets, "aku pindah ke project Delivery", GUIDED_Y, start), {})

    def test_a_message_of_its_own_is_not_part_of_the_guidance(self):
        self.assertIsNone(guidance_request("siapa pemegang jadwal rilis?", GUIDED_Y))
        self.assertEqual(guidance_request("Developer", PICKED_Y), 0)

    def test_extract_facets_reads_the_tool_call_and_never_raises(self):
        llm = fake_llm(tool_name="report_facets", tool_args={"roles": ["Developer", "QA"], "parts": "web"})
        with patch.object(processor, "get_llm", return_value=(llm, "gemini/x")):
            self.assertEqual(processor.extract_facets("FOLDER: A"), (["Developer", "QA"], [], 42))
        with patch.object(processor, "get_llm", return_value=(fake_llm(error=RuntimeError("down")), "gemini/x")):
            self.assertEqual(processor.extract_facets("FOLDER: A"), ([], [], 0))
        with patch.object(processor, "get_llm", return_value=(fake_llm("no call"), "gemini/x")):
            self.assertEqual(processor.extract_facets("FOLDER: A"), (None, None, 42))

    def test_clean_options(self):
        self.assertEqual(clean_options(["a", "A", "  b  c ", "", None, "d", "e", "f", "g"]), ["a", "b c", "d", "e", "f"])
        self.assertEqual(clean_options("a, b"), [])
