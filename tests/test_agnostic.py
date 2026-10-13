"""MS-657: behaviour of POST /agnostic/query (router/agnostic_service.py) recorded
before it is moved, for what test_agnostic_routing.py does not cover:
who may search which source, how retrieved documents become the response,
skills, token metering, the memory window and the error path. Retrieval and
the LLM are faked at the processor; no database is needed."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from config import config
from models import UserRecord
from services.rag.processor import processor
from router import agnostic
from services import agnostic as agnostic_service
from services.auth import get_current_user
from services.billing import inflight

QUESTION = "what does the document say about leave?"  # retrieval by rule, English
COLLECTIONS = [("c1", "ISO Policy"), ("c2", "Handbook")]


def hybrid_args(mock):
    """processor.hybrid_search's positional arguments, by name."""
    names = ["question", "pdf_ids", "search_chat", "search_pdfs", "search_db", "chat_ids",
             "search_links", "link_sources", "search_external_db", "external_db"]
    return dict(zip(names, mock.call_args[0]))


class AgnosticTestCase(unittest.TestCase):
    role = "admin"

    def setUp(self):
        inflight._in_flight_workspace_tokens.clear()
        inflight._in_flight_user_tokens.clear()
        self.user = UserRecord(user_id="u1", email="u1@example.com", role=self.role, is_active=True)
        app = FastAPI()
        app.include_router(agnostic.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: self.user
        self.http = TestClient(app, raise_server_exceptions=False)

        patches = {
            "titles": patch.object(agnostic_service.supabase_storage, "list_collection_titles_for_user",
                                   return_value=COLLECTIONS),
            "skill": patch.object(agnostic_service.supabase_storage, "get_skill_for_user", return_value=None),
            "workspace": patch.object(agnostic_service, "resolve_workspace_id", return_value="w1"),
            "rate": patch.object(agnostic_service, "enforce_rate_limit"),
            "plan": patch.object(agnostic_service, "enforce_plan_limit", return_value=None),
            "member": patch.object(agnostic_service, "enforce_member_allocation", return_value=None),
            "usage": patch.object(agnostic_service, "get_usage_snapshot", return_value=None),
            "log": patch.object(agnostic_service, "log_token_usage"),
            "links": patch.object(agnostic_service, "resolve_active_public_link_sources", return_value=[]),
            "external_db": patch.object(agnostic_service, "resolve_active_database_connections", return_value=[]),
            "all_chats": patch.object(processor, "get_all_chat_collections", return_value=["chat-all"]),
            "search": patch.object(processor, "hybrid_search", return_value={}),
            "summary": patch.object(processor, "build_summary_results", return_value={}),
            "answer": patch.object(processor, "generate_hybrid_answer", return_value=("Answer", "gemini/x", {})),
        }
        self.m = {name: p.start() for name, p in patches.items()}
        self.addCleanup(patch.stopall)

    def ask(self, question=QUESTION, **body):
        return self.http.post("/api/v1/agnostic/query", json={"question": question, **body})

    def answer_args(self):
        """processor.generate_hybrid_answer's positional arguments, by name."""
        names = ["results", "question", "provider", "model", "memory", "skill_instruction",
                 "efficient_mode", "language", "task"]
        return dict(zip(names, self.m["answer"].call_args[0]))


class NonAdminSourcesTest(AgnosticTestCase):
    role = "user"

    def test_only_own_collections_and_never_chat_or_external_databases(self):
        resp = self.ask(pdf_collection_ids=["c1", "someone-elses"], include_chat_results=True,
                        chat_collection_ids=["chat-1"], include_external_db=True)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.m["titles"].assert_called_once_with("u1", False)
        args = hybrid_args(self.m["search"])
        self.assertEqual((args["pdf_ids"], args["search_chat"], args["chat_ids"]), (["c1"], False, []))
        self.assertFalse(args["search_external_db"])
        self.m["external_db"].assert_not_called()
        self.m["all_chats"].assert_not_called()

    def test_no_requested_ids_means_every_own_collection(self):
        self.ask()
        self.assertEqual(sorted(hybrid_args(self.m["search"])["pdf_ids"]), ["c1", "c2"])


class AdminSourcesTest(AgnosticTestCase):
    def test_chat_defaults_to_every_chat_collection(self):
        self.ask(include_chat_results=True)
        args = hybrid_args(self.m["search"])
        self.assertEqual((args["search_chat"], args["chat_ids"]), (True, ["chat-all"]))

    def test_external_databases_and_public_links_are_resolved_and_searched(self):
        connection = {"connection_id": "d1"}
        link = {"link_id": "l1", "items": []}
        self.m["external_db"].return_value = [connection]
        self.m["links"].return_value = [link]
        resp = self.ask(include_external_db=True, external_db_connection_ids=["d1"],
                        include_public_links=True, public_link_ids=["l1"])
        self.m["external_db"].assert_called_once_with(["d1"])
        self.m["links"].assert_called_once_with(["l1"], user_id="u1", is_admin=True)
        args = hybrid_args(self.m["search"])
        self.assertEqual((args["search_links"], args["link_sources"]), (True, [link]))
        self.assertEqual((args["search_external_db"], args["external_db"]), (True, [connection]))
        self.assertEqual(resp.json()["source_type"],
                         "Indexed Collections + Realtime Public Links + Realtime Database Connections")

    def test_sources_that_resolve_to_nothing_are_not_searched(self):
        resp = self.ask(include_public_links=True, include_external_db=True)
        args = hybrid_args(self.m["search"])
        self.assertFalse(args["search_links"] or args["search_external_db"])
        self.assertEqual(resp.json()["source_type"], "Indexed Collections")

    def test_pdf_results_off_searches_no_collection(self):
        self.ask(include_pdf_results=False, include_db_results=True)
        args = hybrid_args(self.m["search"])
        self.assertEqual((args["pdf_ids"], args["search_pdfs"], args["search_db"]), ([], False, True))


class ResponseTest(AgnosticTestCase):
    def test_retrieved_documents_become_sources(self):
        words = " ".join(f"w{i}" for i in range(40))
        self.m["search"].return_value = {
            "pdf_documents": [
                Document(page_content=f"  {words}  ", metadata={"source": "policy.pdf", "page": "3",
                                                                  "collection_id": "c1", "similarity_score": 0.9}),
                Document(page_content="no page", metadata={"source": "notes.txt", "collection_id": "c2"}),
            ],
            "public_link_documents": [Document(page_content="link text", metadata={
                "source": "a.pdf", "public_link_title": "Docs", "item_url": "https://example.com/a.pdf"})],
            "external_db_documents": [Document(page_content="row", metadata={
                "source": "employees", "external_db_label": "HR DB"})],
            "chat_documents": [Document(page_content="c" * 300, metadata={"source": "group", "platform": "whatsapp"})],
            "database_results": {"rows": 1},
            "search_terms": ["leave"],
            "target_tables": ["t1"],
        }
        body = self.ask().json()
        base = "http://testserver/api/v1/files"
        self.assertEqual(body["pdf_sources"], ["policy.pdf (Page 3)", "notes.txt", "Docs / a.pdf", "HR DB / employees"])
        first, second, link, table = body["pdf_sources_detailed"]
        self.assertEqual((first["page"], first["file_url"], first["page_url"]),
                         (3, f"{base}/c1/policy.pdf", f"{base}/c1/policy.pdf#page=3"))
        self.assertEqual(first["search_text"], " ".join(f"w{i}" for i in range(15)))
        self.assertEqual(first["relevance_score"], 0.9)
        self.assertEqual((second["page"], second["page_url"], second["relevance_score"]), (None, f"{base}/c2/notes.txt", 0.0))
        self.assertEqual((link["collection_id"], link["file_url"]), ("public-link", "https://example.com/a.pdf"))
        self.assertEqual((table["collection_id"], table["file_url"]), ("external-db", None))
        self.assertEqual(body["chat_results"], [{"source": "group", "platform": "whatsapp",
                                                 "relevance_score": 0, "content_preview": "c" * 200}])
        self.assertEqual((body["db_results"], body["search_terms"], body["target_tables"]), ({"rows": 1}, ["leave"], ["t1"]))
        self.assertEqual(body["retrieved_count"], 5)

    def test_indonesian_page_label(self):
        self.m["search"].return_value = {"pdf_documents": [
            Document(page_content="x", metadata={"source": "a.pdf", "page": 2, "collection_id": "c1"})]}
        self.assertEqual(self.ask("ringkas dokumen saya").json()["pdf_sources"], ["a.pdf (Halaman 2)"])

    def test_plain_string_answer_uses_the_requested_or_default_model(self):
        self.m["answer"].return_value = "just text"
        self.assertEqual(self.ask(llm_model="m-1").json()["model_used"], "m-1")
        self.assertEqual(self.ask().json()["model_used"], config.default_llm_model)


class SkillAndMeteringTest(AgnosticTestCase):
    def test_a_visible_skill_shapes_the_answer(self):
        self.m["skill"].return_value = {"instruction": "Answer as a table"}
        self.ask("hi", skill_id="s1")
        self.m["skill"].assert_called_once_with("s1", "u1", True)
        self.assertEqual(self.answer_args()["skill_instruction"], "Answer as a table")

    def test_tokens_are_logged_once_with_efficiency_figures(self):
        efficiency = {"enabled": True, "raw_tokens_est": 100, "final_tokens_est": 40}
        self.m["answer"].return_value = ("A", "gemini/x", {"total_tokens": 0, "efficiency": efficiency})
        body = self.ask(efficient_mode=True).json()
        self.m["log"].assert_called_once_with("u1", 0, True, 100, 40)  # logged even at 0 tokens
        self.assertEqual(body["efficiency"], efficiency)
        self.assertTrue(self.answer_args()["efficient_mode"])

    def test_nothing_is_logged_for_a_free_answer(self):
        self.ask()
        self.m["log"].assert_not_called()

    def test_a_failed_usage_log_does_not_break_the_reply(self):
        self.m["answer"].return_value = ("A", "gemini/x", {"total_tokens": 7})
        self.m["log"].side_effect = RuntimeError("db down")
        self.assertEqual(self.ask().json()["answer"], "A")


class MemoryAndErrorsTest(AgnosticTestCase):
    def test_memory_keeps_the_last_five_chats_and_clips_long_turns(self):
        memory = []
        for i in range(1, 8):
            memory += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
        memory[-1]["content"] = "start " + "x" * 2000 + " end?"
        self.ask(memory=memory)
        sent = self.answer_args()["memory"]
        self.assertEqual(sent[0], {"role": "user", "content": "q3"})
        self.assertEqual(len(sent), 10)
        self.assertTrue(sent[-1]["content"].startswith("start ") and sent[-1]["content"].endswith(" end?"))
        self.assertLessEqual(len(sent[-1]["content"]), agnostic_service.MAX_MEMORY_CHARS)

    def test_unknown_command_is_answered_without_the_llm(self):
        with patch.object(processor, "get_llm") as get_llm:
            body = self.ask("/frobnicate").json()
        self.assertEqual(body["model_used"], "system/unknown-command")
        get_llm.assert_not_called()
        self.m["search"].assert_not_called()

    def test_an_unexpected_failure_is_a_500_and_frees_the_reservation(self):
        self.m["search"].side_effect = RuntimeError("index missing")
        resp = self.ask()
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(inflight.in_flight_tokens("w1", "u1"), (0, 0))

    def test_an_empty_question_is_rejected(self):
        self.assertEqual(self.ask("").status_code, 422)


if __name__ == "__main__":
    unittest.main()
