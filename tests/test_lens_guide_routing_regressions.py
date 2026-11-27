"""MS-1183 routing boundaries: application help, folder picks and follow-ups."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services import agnostic as service
from services.lens_guide.clarify import Facets, guidance_request, has_guidance_context, known_answers
from services.lens_guide.scope import load_scope
from services.rag.processor import processor
from tests.fakes import fake_llm
from tests.test_agnostic_stream import FakeModel
from tests.test_lens_guide_answer import passage
from tests.test_lens_guide_clarify import GUIDED_Y
from tests.test_lens_guide_find import FOLDERS, LensGuideEndpoint


class RoutingRegressionTests(LensGuideEndpoint):
    def setUp(self):
        super().setUp()
        service._FACETS.clear()
        self.addCleanup(service._FACETS.clear)
        self.hybrid_search.return_value = passage()
        self.extract = patch.object(processor, "extract_facets", return_value=(["Developer", "QA"], ["web", "core-api"], 0)).start()
        patch.object(processor, "search_across_collections", return_value=[]).start()

    def search_decision(self, collection=None):
        return SimpleNamespace(action="search", total_tokens=5, search_collection=collection,
                               search_folder="Delivery › Proyek Klien Y", search_task="answer", search_query="deploy")

    def post_search(self, question, raw, memory, stream=False, collection=None):
        with patch.object(processor, "route_message", return_value=self.search_decision(collection)) as route, \
                patch.object(processor, "get_llm", return_value=(FakeModel(pieces=[raw]), "gemini/test")):
            response = self.client.post("/api/v1/agnostic/query" + ("?stream=true" if stream else ""),
                                        json={"question": question, "memory": memory})
        self.assertEqual(response.status_code, 200, response.text)
        route.assert_called_once()
        return response

    def test_application_how_do_i_stays_application_help(self):
        llm = fake_llm("Connect Telegram in Sources, then verify your account.")
        data = self.ask("How do I connect Telegram?", llm)
        self.assertIn("Connect Telegram", data["answer"])
        self.hybrid_search.assert_not_called()
        llm.invoke.assert_called_once()

    def test_application_help_in_guided_folder_does_not_ask_roles(self):
        llm = fake_llm("Connect Telegram in Sources, then verify your account.")
        data = self.ask("How do I connect Telegram?", llm, memory=GUIDED_Y)
        self.assertIn("Connect Telegram", data["answer"])
        self.extract.assert_not_called()
        self.hybrid_search.assert_not_called()

    def test_partial_folder_name_keeps_find_folder_and_confirmation(self):
        self.folders.return_value = FOLDERS + [{"folder_id": "backup", "name": "Backup & Pemulihan",
                                                "owner_id": "u1", "parent_folder_id": None}]
        llm = self.find("Backup")
        data = self.ask("Saya mau mempelajari modul Backup", llm)
        self.assertEqual(data["guide"]["folder"]["path"], ["Backup & Pemulihan"])
        self.assertIn("Benar ini folder", data["answer"])
        self.hybrid_search.assert_not_called()

    def test_procedural_follow_up_is_not_a_new_context_boundary(self):
        self.assertIsNone(guidance_request("How do I deploy?", GUIDED_Y))
        facets = Facets(roles=["Developer", "QA"], parts=["web", "core-api"])
        original = guidance_request("Ganti Peran jadi Developer", GUIDED_Y)
        self.assertEqual(known_answers(facets, "How do I deploy?", GUIDED_Y, original),
                         {"role": "Developer", "part": "core-api"})
        data = self.post_search("How do I deploy?", "1. Deploy | Ikuti prosedur | [S1]", GUIDED_Y).json()
        self.extract.assert_not_called()
        self.assertIsNone(data["guide"]["question"])
        self.assertEqual(data["guide"]["steps"][0]["key"], "1")

    def test_plain_folder_fact_keeps_normal_answer_and_stream(self):
        memory = [{"role": "user", "content": "ringkas dokumen Proyek Klien Y"},
                  {"role": "assistant", "content": "Saya menemukan folder **Delivery › Proyek Klien Y**. Benar ini folder yang kamu maksud?"}]
        raw = "The deployment procedure has one documented approval step."
        response = self.post_search("Who approves deployment?", raw, memory, stream=True)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertIn("delta", [e["type"] for e in events])
        self.assertEqual(events[-1]["result"]["answer"], raw)
        self.assertIsNone(events[-1]["result"]["guide"])
        self.assertEqual(self.hybrid_search.call_count, 1)

    def test_explicit_guidance_fact_stays_honest_without_secondary_search(self):
        response = self.post_search("Bagaimana meminta akses staging?", "TIDAK ADA: prosedur akses staging", GUIDED_Y)
        self.assertEqual(response.json()["guide"]["notices"][0]["kind"], "missing")
        self.assertEqual(self.hybrid_search.call_count, 1)
        self.extract.assert_not_called()

    def test_document_narrowing_does_not_reclassify_same_folder_siblings_as_outside(self):
        self.post_search("How do I deploy?", "1. Deploy | Prosedur | [S1]", GUIDED_Y, collection="Brief")
        self.assertEqual(self.hybrid_search.call_args_list[0].args[1], ["c2"])
        self.assertEqual(set(self.hybrid_search.call_args_list[1].args[1]), {"c3", "c4"})
        self.assertNotIn("c1", self.hybrid_search.call_args_list[1].args[1])

    def test_saved_guide_list_is_an_opt_in_after_the_original_request_is_gone(self):
        scope = load_scope(SimpleNamespace(user_id="u1", role="admin"))
        folder = next(f for f in scope.folders if f.folder_id == "y")
        memory = [{"role": "assistant", "content": "a. **Brief** ([brief.pdf](http://local/api/v1/files/c2/brief.pdf#page=1), hlm 1): Tujuan"}]
        self.assertTrue(has_guidance_context(memory, folder, scope))
        other = next(f for f in scope.folders if f.folder_id == "by")
        self.assertFalse(has_guidance_context(memory, other, scope))

    def test_new_plain_folder_confirmation_does_not_inherit_previous_guidance(self):
        scope = load_scope(SimpleNamespace(user_id="u1", role="admin"))
        other = next(f for f in scope.folders if f.folder_id == "by")
        memory = GUIDED_Y + [{"role": "user", "content": "cari folder milik Budi"},
                             {"role": "assistant", "content": "Saya menemukan folder **Proyek Klien Y**. Benar ini folder yang kamu maksud?"}]
        self.assertFalse(has_guidance_context(memory, other, scope))
