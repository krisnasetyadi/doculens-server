"""MS-1183 T1-T3: folder hints, bounded related search and reading references."""
import unittest
from unittest.mock import patch

from models import MemoryTurn
from services import agnostic as service
from services.lens_guide import continuation as continuation_module
from services.lens_guide.answer import SourceRef, parse_guide_answer, source_detail
from services.lens_guide.continuation import continuation
from services.lens_guide.scope import Scope, ScopeCollection
from services.rag.conversation.intent import RouteDecision
from services.rag.processor import processor
from tests.fakes import fake_llm
from tests.test_agnostic_stream import FakeModel
from tests.test_lens_guide_answer import passage
from tests.test_lens_guide_find import LensGuideEndpoint


class PossibleFolderTests(LensGuideEndpoint):
    def setUp(self):
        super().setUp()
        self.folders.return_value = [
            {"folder_id": f"f{i}", "name": name, "parent_folder_id": None, "owner_id": "u1"}
            for i, name in enumerate(["Catatan Rilis Tim", "Kebijakan Perusahaan", "Dokumen", "Backup & Pemulihan"])
        ]
        self.hybrid_search.return_value = passage()

    def assert_global_guide(self, question):
        # The router reads the request (label "do"); words that look like folder names do not turn it into a folder lookup.
        decision = RouteDecision(action="search", search_query=question, guide="do", model_id="gemini/test")
        with patch.object(processor, "route_message", return_value=decision) as route, \
                patch.object(processor, "get_llm", return_value=(FakeModel(pieces=["1. Rilis | Prosedur tertulis | [S1]"]), "gemini/test")):
            response = self.client.post("/api/v1/agnostic/query", json={"question": question})
        self.assertEqual(response.status_code, 200, response.text)
        route.assert_called_once()
        data = response.json()
        self.assertTrue(data.get("guide") and data["guide"].get("steps"), data)
        self.assertEqual(data["guide"]["steps"][0]["key"], "1")
        self.assertEqual(set(self.hybrid_search.call_args.args[1]), {"c1", "c2", "c3", "c4"})

    def test_release_word_inside_folder_name_is_not_a_folder_hint(self):
        self.assert_global_guide("Saya mau rilis fitur pembayaran ke produksi. Harus apa?")

    def test_common_document_and_policy_words_are_not_folder_hints(self):
        self.assert_global_guide("Saya perlu bikin dokumen kebijakan baru, langkahnya apa?")

    def test_explicit_module_marker_still_confirms_a_partial_name(self):
        llm = self.find("Backup")
        data = self.ask("Saya mau mempelajari modul Backup", llm)
        llm.invoke.assert_called_once()
        self.assertEqual(data["guide"]["folder"]["path"], ["Backup & Pemulihan"])
        self.hybrid_search.assert_not_called()

    def test_capitalized_folder_word_still_goes_to_router(self):
        llm = self.find("Backup")
        data = self.ask("Aku mau belajar Backup", llm)
        llm.invoke.assert_called_once()
        self.assertEqual(data["guide"]["folder"]["path"], ["Backup & Pemulihan"])
        self.hybrid_search.assert_not_called()

    def test_full_folder_name_still_goes_to_folder_confirmation(self):
        llm = self.find("Catatan Rilis Tim")
        data = self.ask("Saya mau mempelajari catatan rilis tim", llm)
        self.assertEqual(data["guide"]["folder"]["path"], ["Catatan Rilis Tim"])
        self.hybrid_search.assert_not_called()

    def test_sentence_initial_capital_is_not_a_proper_folder_hint(self):
        self.folders.return_value += [{"folder_id": "h", "name": "Harus", "parent_folder_id": None, "owner_id": "u1"}]
        self.assert_global_guide("Saya mau rilis fitur pembayaran ke produksi. Harus apa?")

    def test_application_help_still_uses_router(self):
        llm = fake_llm("Connect Telegram in Sources, then verify your account.")
        data = self.ask("How do I connect Telegram?", llm)
        llm.invoke.assert_called_once()
        self.assertIn("Connect Telegram", data["answer"])
        self.hybrid_search.assert_not_called()


class RelatedSearchLimitTests(LensGuideEndpoint):
    def test_sixty_outside_collections_use_the_first_fifty_in_scope_order(self):
        service._FACETS.clear()
        self.addCleanup(service._FACETS.clear)
        outside = [{"collection_id": f"outside-{i:02}", "title": f"Reference {i}",
                    "file_names": [f"reference-{i}.pdf"], "folder_id": None} for i in range(59, -1, -1)]
        rows = [{"collection_id": "c2", "title": "Brief", "file_names": ["brief.pdf"], "folder_id": "y"}] + outside
        with patch.object(service.supabase_storage, "list_active_collections_for_user", return_value=rows), \
                patch.object(processor, "extract_facets", return_value=([], [], 0)), \
                patch.object(processor, "search_across_collections", return_value=[]):
            self.hybrid_search.side_effect = [passage(), {}]
            data = self.ask("Ya, pakai folder Delivery › Proyek Klien Y", fake_llm("not used"), memory=[
                {"role": "user", "content": "Aku baru masuk Delivery Klien Y"},
                {"role": "assistant", "content": "Saya menemukan folder **Delivery › Proyek Klien Y**. Benar ini folder yang kamu maksud?"},
            ])
        self.assertIsNotNone(data["guide"]["context"])
        self.assertEqual(self.hybrid_search.call_count, 2)
        self.assertEqual(self.hybrid_search.call_args_list[1].args[1], [row["collection_id"] for row in outside[:50]])


class ReadingReferenceTests(unittest.TestCase):
    def setUp(self):
        self.scope = Scope(collections=[ScopeCollection("c1", "Manual", 1, "f", ["Atlas"], ["manual.pdf"])])
        self.source = SourceRef("S1", source_detail("manual.pdf", "c1", 4, "http://testserver"))
        text = parse_guide_answer("\n".join(f"{i}. Task {i} | Instruction | [S1]" for i in range(1, 7)),
                                  "do", [self.source], [], "id").text
        self.memory = [MemoryTurn(role="assistant", content=text)]

    def next_key(self, question, memory=None):
        return continuation(question, self.memory if memory is None else memory, self.scope,
                            "http://testserver", "id")[1].steps[0].key

    def test_document_locations_do_not_complete_step_four(self):
        history = self.memory + [MemoryTurn(role="user", content="sudah baca 1 dan 2 dan 3")]
        for label in ("halaman", "hlm", "hal", "page", "p.", "bab", "slide", "sheet", "bagian"):
            with self.subTest(label=label):
                self.assertEqual(self.next_key(f"sudah baca {label} 4", history), "4")

    def test_page_range_does_not_complete_steps(self):
        history = self.memory + [MemoryTurn(role="user", content="sudah baca 1 dan 2 dan 3")]
        self.assertEqual(self.next_key("sudah baca halaman 1 sampai 4", history), "4")

    def test_explicit_step_keys_still_count(self):
        self.assertEqual(self.next_key("sudah baca 1 dan 2"), "3")

    def test_alphabetical_range_still_counts(self):
        sources = [SourceRef(f"S{i}", source_detail(f"chapter-{i}.pdf", "c1", 1, "http://testserver")) for i in range(1, 7)]
        self.scope.collections[0].file_names = [source.source.file_name for source in sources]
        text = parse_guide_answer("\n".join(f"{i}. Chapter {i} | Read | 10 [S{i}]" for i in range(1, 7)),
                                  "start", sources, [], "id").text
        self.assertEqual(self.next_key("sudah baca a sampai d", [MemoryTurn(role="assistant", content=text)]), "e")

    def test_continuation_and_chat_use_the_same_memory_limits(self):
        self.assertEqual(getattr(continuation_module, "MAX_MEMORY_CHATS", None), service.MAX_MEMORY_CHATS)
        self.assertEqual(getattr(continuation_module, "MAX_MEMORY_MESSAGES", None), service.MAX_MEMORY_MESSAGES)
