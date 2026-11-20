"""The processor's language-aware prompts and fallbacks, its conversation
router call, the vector-store load backoff and the related config.

Split out of test_conversation_language.py by MS-657."""

import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET", "conversation-local-test-secret-not-for-production")

from services.rag.conversation.messages import internal_refusal
from services.rag.processor import processor, OFF_TOPIC_REDIRECT_EN, OFF_TOPIC_REDIRECT_ID
from services import agnostic as agnostic_service
from tests.fakes import fake_llm


class PromptLanguageTests(unittest.TestCase):
    def test_indonesian_prompts_unchanged(self):
        general = processor._build_general_prompt("ctx", "apa isi?", [], {}, language="id")
        self.assertIn("- Jawab ringkas dan jelas dalam Bahasa Indonesia", general)
        self.assertIn(OFF_TOPIC_REDIRECT_ID, general)
        self.assertIn("Informasi ini tidak ditemukan dalam dokumen yang diunggah.", general)
        self.assertNotIn("English", general)

        explanation = processor._build_explanation_prompt("ctx", "apa itu x?", [], language="id")
        self.assertIn("- Jawab dengan jelas dalam Bahasa Indonesia", explanation)
        comparison = processor._build_comparison_prompt("ctx", "bandingkan", [], language="id")
        self.assertIn("5. Jawab dalam Bahasa Indonesia", comparison)
        aggregation = processor._build_aggregation_prompt("ctx", "total?", [], language="id")
        self.assertIn("5. Berikan jawaban dalam Bahasa Indonesia", aggregation)
        self.assertIn('"Berdasarkan data dari [sumber], [jawaban numerik]"', aggregation)

    def test_english_prompts_pin_english(self):
        prompts = [
            processor._build_general_prompt("ctx", "what?", [], {}, language="en"),
            processor._build_explanation_prompt("ctx", "what is x?", [], sole_source_type="chat", language="en"),
            processor._build_comparison_prompt("ctx", "compare", [], language="en"),
            processor._build_aggregation_prompt("ctx", "total?", [], language="en"),
        ]
        for prompt in prompts:
            self.assertIn("Write the ENTIRE answer in English", prompt)
            self.assertNotIn("dalam Bahasa Indonesia", prompt)
            self.assertIn(OFF_TOPIC_REDIRECT_EN, prompt)
        self.assertIn("This information was not found in the uploaded documents.", prompts[0])
        self.assertIn("This information was not found in the available chat conversations.", prompts[1])
        self.assertIn("5. IMPORTANT:", prompts[2])

    def test_default_language_is_indonesian_for_direct_callers(self):
        # Builders keep their old behavior when called without `language`.
        self.assertIn("dalam Bahasa Indonesia", processor._build_general_prompt("ctx", "q", [], {}))

    def test_gap_check_prompt_language(self):
        batch = [{"requirement": {"key": "A.5.1", "code": "A.5.1", "label": "Policies", "expected_document": "policy"},
                  "sources": [{"id": "S1", "file_name": "policy.pdf", "page": "2", "text": "Approved yearly."}]}]
        id_prompt = processor._build_gap_check_batch_prompt("ISO 27001", batch, ["policy.pdf"], language="id")
        en_prompt = processor._build_gap_check_batch_prompt("ISO 27001", batch, ["policy.pdf"], language="en")
        self.assertIn("in Indonesian (Bahasa Indonesia)", id_prompt)
        self.assertNotIn("in Indonesian", en_prompt)
        self.assertIn("in English.", en_prompt)
        self.assertEqual(id_prompt, processor._build_gap_check_batch_prompt("ISO 27001", batch, ["policy.pdf"]))
        # MS-1112: the selected documents decide not_checked versus missing; sources are numbered.
        self.assertIn("THE USER SELECTED ONLY THESE DOCUMENTS TO CHECK:\n- policy.pdf", id_prompt)
        self.assertIn("ITEM 1 (key: A.5.1): A.5.1 Policies", id_prompt)
        self.assertIn("[S1] policy.pdf (page 2):\nApproved yearly.", id_prompt)
        self.assertNotIn("DIRECTIONS FROM THE USER", id_prompt)
        self.assertIn("DIRECTIONS FROM THE USER (follow them):\nIgnore drafts",
                      processor._build_gap_check_batch_prompt("ISO 27001", batch, [], "Ignore drafts", "en"))

    def test_canned_english_replies_survive_validation(self):
        # The validator swaps answers matching failure_patterns for a raw
        # chunk — the English canned replies must not trip it.
        results = [{"source": "a.pdf", "content": "chunk", "confidence": 0.5, "type": "pdf"}]
        for text in [OFF_TOPIC_REDIRECT_EN, processor._NOT_FOUND_EN["pdf"] + " Please check the uploaded files."]:
            self.assertEqual(processor._validate_and_clean_answer(text, "q", results, "en"), text)

    def test_fallback_prefixes_follow_language(self):
        result = {"source": "a.pdf", "content": "chunk text", "confidence": 0.5, "type": "pdf"}
        self.assertTrue(processor._extract_direct_answer(result, "what?", "en").startswith("Based on a.pdf:"))
        self.assertTrue(processor._extract_direct_answer(result, "apa?", "id").startswith("Berdasarkan a.pdf:"))
        self.assertTrue(processor._generate_fallback_answer({}, "q", "en").startswith("The system couldn't"))
        self.assertTrue(processor._generate_fallback_answer({}, "q").startswith("Maaf, sistem"))


class RouterTests(unittest.TestCase):
    """processor.route_message: one LLM call that replies or calls search_documents."""

    def route(self, question, llm, language="en", memory=None, titles=("ISO Policy",)):
        with patch.object(processor, "get_llm", return_value=(llm, "gemini/gemini-test")):
            return processor.route_message(question, language, memory, False, list(titles), len(titles))

    def test_reply_path_uses_feature_guide_language_and_tool(self):
        llm = fake_llm("Hi! What can I help you with?")
        decision = self.route("hi", llm)
        self.assertEqual(
            (decision.action, decision.answer, decision.total_tokens),
            ("reply", "Hi! What can I help you with?", 42),
        )
        prompt = llm.invoke.call_args[0][0]
        self.assertIn("Write the ENTIRE reply in English", prompt)
        self.assertIn("PLANS", prompt)
        self.assertIn("do NOT offer a specific document out of nowhere", prompt)
        tool = llm.bind_tools.call_args[0][0][0]
        self.assertEqual(tool["name"], "search_documents")
        self.assertEqual(tool["parameters"]["properties"]["collection"]["enum"], ["ISO Policy"])

    def test_function_call_becomes_a_structured_search(self):
        llm = fake_llm(
            tool_args={"query": "ringkasan ISO Policy", "collection": "ISO Policy", "task": "summarize"},
            total_tokens=9,
        )
        memory = [{"role": "user", "content": "halo"}, {"role": "assistant", "content": "Mau saya ringkas ISO Policy?"}]
        decision = self.route("ok", llm, "id", memory)
        self.assertEqual(
            (decision.action, decision.search_query, decision.search_collection, decision.search_task, decision.total_tokens),
            ("search", "ringkasan ISO Policy", "ISO Policy", "summarize", 9),
        )
        self.assertIn("Mau saya ringkas ISO Policy?", llm.invoke.call_args[0][0])

    def test_bad_function_args_are_normalized(self):
        decision = self.route("what is the leave policy?", fake_llm(tool_args={"query": " ", "task": "weird"}))
        self.assertEqual(
            (decision.search_query, decision.search_task, decision.search_collection),
            ("what is the leave policy?", "answer", None),
        )

    def test_llm_failure_falls_back_safely(self):
        broken = fake_llm(error=RuntimeError("quota"))
        small_talk = self.route("terima kasih", broken, "id")
        self.assertEqual(small_talk.action, "reply")
        self.assertTrue(small_talk.answer.startswith("Sama-sama"))
        data = self.route("what is the leave policy?", broken)
        self.assertEqual((data.action, data.search_query, data.fallback), ("search", "what is the leave policy?", True))

    def route_with_folders(self, question, llm, folder_paths, memory=None):
        with patch.object(processor, "get_llm", return_value=(llm, "gemini/gemini-test")):
            return processor.route_message(
                question, "id", memory, False, ["ISO Policy"], 1, folder_paths=folder_paths,
            )

    def test_find_folder_is_offered_only_when_folders_were_read(self):
        llm = fake_llm("Halo!")
        self.route("hi", llm)
        self.assertEqual([t["name"] for t in llm.bind_tools.call_args[0][0]], ["search_documents"])
        self.assertNotIn("find_folder", llm.invoke.call_args[0][0])

        llm = fake_llm("Halo!")
        self.route_with_folders("halo", llm, [])  # read, but none: a named project still gets asked about
        tools = llm.bind_tools.call_args[0][0]
        self.assertEqual([t["name"] for t in tools], ["search_documents", "find_folder"])
        self.assertNotIn("folder", tools[0]["parameters"]["properties"])
        self.assertIn("CALL find_folder", llm.invoke.call_args[0][0])

    def test_folder_paths_become_the_search_folder_enum(self):
        llm = fake_llm("Halo!")
        paths = ["Delivery", "Delivery › Proyek Klien Y"]
        self.route_with_folders("halo", llm, paths)
        folder = llm.bind_tools.call_args[0][0][0]["parameters"]["properties"]["folder"]
        self.assertEqual(folder["enum"], paths)

        llm = fake_llm("Halo!")
        self.route_with_folders("halo", llm, [f"F{i}" for i in range(51)])
        folder = llm.bind_tools.call_args[0][0][0]["parameters"]["properties"]["folder"]
        self.assertNotIn("enum", folder)  # past the cap: free text, matched back by the server

    def test_find_folder_call_carries_only_the_name(self):
        decision = self.route_with_folders(
            "aku baru masuk Proyek Klien Y",
            fake_llm(tool_name="find_folder", tool_args={"name": " Proyek Klien Y "}, total_tokens=7),
            ["Delivery › Proyek Klien Y"],
        )
        self.assertEqual(
            (decision.action, decision.folder_name, decision.total_tokens), ("find_folder", "Proyek Klien Y", 7),
        )

    def test_find_folder_without_a_name_searches_as_typed(self):
        decision = self.route_with_folders(
            "dokumen onboarding", fake_llm(tool_name="find_folder", tool_args={"name": " "}), [],
        )
        self.assertEqual((decision.action, decision.search_query), ("search", "dokumen onboarding"))

    def test_search_keeps_the_confirmed_folder(self):
        memory = [{"role": "assistant", "content": "Aku menemukan Sources › Delivery › Proyek Klien Y. Benar?"}]
        decision = self.route_with_folders(
            "ya, apa yang harus kubaca dulu?",
            fake_llm(tool_args={"query": "dokumen onboarding", "folder": "Delivery › Proyek Klien Y", "task": "answer"}),
            ["Delivery › Proyek Klien Y"], memory,
        )
        self.assertEqual((decision.action, decision.search_folder), ("search", "Delivery › Proyek Klien Y"))

    def test_a_folder_argument_is_ignored_when_folders_are_off(self):
        decision = self.route("what is in it?", fake_llm(tool_args={"query": "x", "folder": "Secret", "task": "answer"}))
        self.assertIsNone(decision.search_folder)

    def test_reply_leak_is_replaced(self):
        decision = self.route("hi", fake_llm("DocuLens runs on FastAPI with Gemini."))
        self.assertEqual(decision.answer, internal_refusal("en"))
        self.assertTrue(decision.guard_replaced)

    def test_summary_results_spread_over_the_whole_document(self):
        from langchain.schema import Document

        docs = {str(i): Document(page_content=f"chunk {i}", metadata={"source": "a.pdf"}) for i in range(30)}
        store = MagicMock()
        store.docstore._dict = docs
        with patch.object(processor, "get_vector_store", return_value=store):
            results = processor.build_summary_results(["c1"])
        contents = [r["content"] for r in results["merged_results"]]
        self.assertEqual(len(contents), processor.SUMMARY_MAX_CHUNKS)
        self.assertEqual(contents[0], "chunk 0")
        self.assertGreater(int(contents[-1].split()[1]), 25)  # reaches the end, not just the top hits
        self.assertEqual(results["search_analysis"]["source_weights"], {"pdf": 1})

    def test_summary_prompt(self):
        prompt = processor._build_summary_prompt("ctx", "summarize ISO Policy", language="en")
        self.assertIn("RINGKASAN:", prompt)
        self.assertIn("Write the ENTIRE answer in English", prompt)

    def test_long_turns_keep_their_closing_offer(self):
        long_reply = "x" * 2000 + " Mau saya ringkas ISO Policy?"
        clipped = agnostic_service._clip_turn(long_reply)
        self.assertLessEqual(len(clipped), agnostic_service.MAX_MEMORY_CHARS)
        self.assertTrue(clipped.endswith("Mau saya ringkas ISO Policy?"))


class VectorStoreBackoffTests(unittest.TestCase):
    def test_failed_load_backs_off_then_retries(self):
        with patch.object(processor, "_load_vector_store", return_value=None) as load, \
                patch.object(processor, "embeddings", object()), \
                patch("services.rag.indexes.time.monotonic", side_effect=[0, 0, 10, 31, 31, 40]):
            processor._missing_vector_stores.pop("broken", None)
            processor.get_vector_store("broken")   # t=0: fails -> retry after 30s
            processor.get_vector_store("broken")   # t=10: skipped, no load
            processor.get_vector_store("broken")   # t=31: retried, fails -> retry after 120s
            processor.get_vector_store("broken")   # t=40: skipped
        self.assertEqual(load.call_count, 2)
        self.assertEqual(processor._missing_vector_stores["broken"][1], 2)
        processor.invalidate_cache("broken")
        self.assertNotIn("broken", processor._missing_vector_stores)


class ConfigTests(unittest.TestCase):
    def test_empty_thinking_budget_means_default(self):
        from config import Config

        with patch.dict(os.environ, {"GEMINI_THINKING_BUDGET": ""}):
            self.assertIsNone(Config().gemini_thinking_budget)
        with patch.dict(os.environ, {"GEMINI_THINKING_BUDGET": "512"}):
            self.assertEqual(Config().gemini_thinking_budget, 512)



class GapCheckJsonParsingTests(unittest.TestCase):
    """The model's JSON comes back plain, fenced, or wrapped in prose."""

    def test_lists(self):
        parse = processor._parse_json_list
        self.assertEqual(parse('["A.5", "A.6"]'), ["A.5", "A.6"])
        self.assertEqual(parse('```json\n["A.5"]\n```'), ["A.5"])
        self.assertEqual(parse('{"items": ["A.5", 7]}'), ["A.5", "7"])
        self.assertEqual(parse('Here you go: ["A.5", "A.6"] hope it helps'), ["A.5", "A.6"])
        self.assertEqual(parse("no json here"), [])

    def test_objects(self):
        parse = processor._parse_json_objects
        self.assertEqual(parse('[{"label": "A.5"}]'), [{"label": "A.5"}])
        self.assertEqual(parse('Result: [{"label": "A.5"}] done'), [{"label": "A.5"}])
        self.assertEqual(parse("[not json"), [])


class GapCheckSourcesTests(unittest.TestCase):
    """MS-1112: a requirement is judged on passages from every selected document."""

    def test_every_document_gets_its_best_passage_in_before_any_second_best(self):
        from langchain_core.documents import Document

        def search(query, collection_ids, top_k):
            cid = collection_ids[0]
            return [Document(page_content=f"{cid}-{n}", metadata={"source": f"{cid}.pdf", "collection_id": cid, "page": "1"})
                    for n in range(1, top_k + 2)]

        requirement = {"key": "A.8.13", "code": "A.8.13", "label": "Backup", "expected_document": None}
        with patch.object(processor, "search_across_collections", side_effect=search) as searched:
            sources = processor._retrieve_sources(requirement, ["a", "b", "c", "d", "e"])
        self.assertEqual(searched.call_count, 5)  # one search per document
        self.assertEqual([s["text"] for s in sources], ["a-1", "b-1", "c-1", "d-1", "e-1", "a-2", "b-2", "c-2"])
        self.assertEqual((sources[0]["id"], sources[-1]["id"]), ("S1", "S8"))  # capped at 8

if __name__ == "__main__":
    unittest.main()
