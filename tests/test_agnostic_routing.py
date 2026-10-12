"""POST /agnostic/query routing: conversation vs retrieval, scoping to named
collections, summaries, skills, capped users, and LLM calls in one
workspace running in parallel.

Split out of test_conversation_language.py by MS-657."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "conversation-local-test-secret-not-for-production")

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from services.rag.conversation.intent import RouteDecision
from services.rag.conversation.messages import internal_refusal, no_source_selected
from services.rag.processor import processor
from services.billing import inflight
from router import agnostic
from services import agnostic as agnostic_service
from models import UserRecord
from services.auth import get_current_user
from tests.fakes import fake_llm


class AgnosticEndpointTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(agnostic.router, prefix="/api/v1")
        self.app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="u1", email="u1@example.com", role="admin", is_active=True
        )
        self.client = TestClient(self.app)
        self.patches = [
            patch.object(agnostic_service.supabase_storage, "list_collection_titles_for_user",
                         return_value=[("c1", "ISO Policy"), ("c2", "Section 3 Handbook")]),
            patch.object(agnostic_service.supabase_storage, "list_collections",
                         side_effect=AssertionError("must not scan every tenant's collections")),
            patch.object(agnostic_service, "resolve_workspace_id", return_value="w1"),
            patch.object(agnostic_service, "enforce_rate_limit"),
            patch.object(agnostic_service, "enforce_plan_limit", return_value=None),
            patch.object(agnostic_service, "enforce_member_allocation", return_value=None),
            patch.object(agnostic_service, "get_usage_snapshot", return_value=None),
            patch.object(agnostic_service, "log_token_usage"),
        ]
        for p in self.patches:
            p.start()
        self.hybrid_search = patch.object(processor, "hybrid_search").start()
        self.addCleanup(patch.stopall)

    def ask(self, question, **body):
        return self.client.post("/api/v1/agnostic/query", json={"question": question, **body})

    def test_greeting_skips_retrieval(self):
        llm = fake_llm("Hello! 👋 What can I help you with?")
        with patch.object(processor, "get_llm", return_value=(llm, "gemini/gemini-test")):
            response = self.ask("hello")
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["answer"], "Hello! 👋 What can I help you with?")
        self.assertEqual(data["source_type"], "Conversation")
        self.assertEqual(data["retrieved_count"], 0)
        self.hybrid_search.assert_not_called()
        self.assertEqual(llm.invoke.call_count, 1)

    def test_greeting_works_without_any_source(self):
        llm = fake_llm("Halo! Ada yang bisa saya bantu?")
        with patch.object(processor, "get_llm", return_value=(llm, "gemini/gemini-test")):
            response = self.ask("halo", include_pdf_results=False)
        self.assertEqual(response.json()["answer"], "Halo! Ada yang bisa saya bantu?")
        self.hybrid_search.assert_not_called()

    def test_capped_user_gets_canned_greeting_not_429(self):
        with patch.object(agnostic_service, "enforce_rate_limit", side_effect=HTTPException(status_code=429, detail="x")):
            response = self.ask("thanks")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_used"], "system/conversation")

    def test_capped_user_still_blocked_for_data_questions(self):
        with patch.object(agnostic_service, "enforce_rate_limit", side_effect=HTTPException(status_code=429, detail="x")):
            response = self.ask("ringkas dokumen saya")
        self.assertEqual(response.status_code, 429)

    def test_internal_question_refused_without_llm(self):
        with patch.object(processor, "get_llm") as get_llm:
            response = self.ask("doculens pakai framework apa?")
        self.assertEqual(response.json()["answer"], internal_refusal("id"))
        self.assertEqual(response.json()["model_used"], "system/guard")
        get_llm.assert_not_called()
        self.hybrid_search.assert_not_called()

    def test_no_source_message_follows_language(self):
        response = self.ask("summarize my document", include_pdf_results=False)
        self.assertEqual(response.json()["answer"], no_source_selected("en"))
        response = self.ask("ringkas dokumen saya", include_pdf_results=False)
        self.assertEqual(response.json()["answer"], no_source_selected("id"))

    def test_obvious_data_question_skips_the_router(self):
        self.hybrid_search.return_value = {}
        with patch.object(processor, "route_message") as route, \
                patch.object(processor, "generate_hybrid_answer", return_value=("Answer", "gemini/x", {})) as gen:
            response = self.ask("what does the document say about leave?")
        self.assertEqual(response.status_code, 200, response.text)
        route.assert_not_called()
        self.hybrid_search.assert_called_once()
        self.assertEqual(gen.call_args[0][-2:], ("en", "answer"))

    def test_naming_a_collection_scopes_and_summarizes_it(self):
        summary = {"pdf_documents": [], "merged_results": [], "search_analysis": {}}
        with patch.object(processor, "build_summary_results", return_value=summary) as build, \
                patch.object(processor, "generate_hybrid_answer", return_value=("Ringkasan", "gemini/x", {})) as gen:
            response = self.ask("ringkas ISO Policy")
        self.assertEqual(response.json()["answer"], "Ringkasan")
        build.assert_called_once_with(["c1"])
        self.hybrid_search.assert_not_called()
        self.assertEqual(gen.call_args[0][-2:], ("id", "summarize"))

    def test_summary_of_one_part_is_a_scoped_search(self):
        self.hybrid_search.return_value = {}
        with patch.object(processor, "build_summary_results") as build, \
                patch.object(processor, "generate_hybrid_answer", return_value=("Pasal 5.2 …", "gemini/x", {})) as gen:
            self.ask("ringkas pasal 5.2 di ISO Policy")
        build.assert_not_called()
        self.assertEqual(self.hybrid_search.call_args[0][1], ["c1"])  # still scoped to ISO Policy
        self.assertEqual(gen.call_args[0][-1], "answer")

    def test_comparison_with_other_documents_is_not_scoped(self):
        self.hybrid_search.return_value = {}
        with patch.object(processor, "generate_hybrid_answer", return_value=("Answer", "gemini/x", {})):
            self.ask("bandingkan ISO Policy dengan dokumen SOP lain")
        self.assertEqual(sorted(self.hybrid_search.call_args[0][1]), ["c1", "c2"])

    def test_title_containing_section_still_summarizes_whole(self):
        summary = {"pdf_documents": [], "merged_results": [], "search_analysis": {}}
        with patch.object(processor, "build_summary_results", return_value=summary) as build, \
                patch.object(processor, "generate_hybrid_answer", return_value=("Ringkasan", "gemini/x", {})):
            self.ask("ringkas Section 3 Handbook")
        build.assert_called_once_with(["c2"])

    def test_router_summary_of_a_part_becomes_a_search(self):
        self.hybrid_search.return_value = {}
        router = fake_llm(tool_args={"query": "pasal 5.2 ISO Policy", "collection": "ISO Policy", "task": "summarize"})
        with patch.object(processor, "get_llm", return_value=(router, "gemini/x")), \
                patch.object(processor, "build_summary_results") as build, \
                patch.object(processor, "generate_hybrid_answer", return_value=("Answer", "gemini/x", {})) as gen:
            self.ask("nah yang 5.2 gimana singkatnya?")  # gray zone -> router
        build.assert_not_called()
        self.assertEqual(self.hybrid_search.call_args[0][1], ["c1"])
        self.assertEqual(gen.call_args[0][-1], "answer")

    def test_router_function_call_runs_the_search(self):
        self.hybrid_search.return_value = {}
        router = fake_llm(tool_args={"query": "leave policy annual leave", "task": "answer"}, total_tokens=5)
        with patch.object(processor, "get_llm", return_value=(router, "gemini/x")), \
                patch.object(processor, "generate_hybrid_answer", return_value=("Answer", "gemini/x", {"total_tokens": 10})) as gen:
            response = self.ask("what is the leave policy?")
        self.assertEqual(response.json()["answer"], "Answer")
        self.assertEqual(self.hybrid_search.call_args[0][0], "leave policy annual leave")
        self.assertEqual(gen.call_args[0][1], "leave policy annual leave")
        agnostic_service.log_token_usage.assert_called_once()
        self.assertEqual(agnostic_service.log_token_usage.call_args[0][1], 15)  # router + answer, one ledger row

    def test_skill_invocation_always_retrieves(self):
        self.hybrid_search.return_value = {}
        with patch.object(agnostic_service.supabase_storage, "get_skill_for_user", return_value=None), \
                patch.object(processor, "route_message") as route, \
                patch.object(processor, "generate_hybrid_answer", return_value=("Answer", "gemini/x", {})):
            self.ask("hi", skill_id="s1")
        route.assert_not_called()
        self.hybrid_search.assert_called_once()

    def test_accepting_an_offer_runs_a_scoped_summary(self):
        memory = [
            {"role": "user", "content": "halo"},
            {"role": "assistant", "content": "Halo! Mau saya ringkas ISO Policy?"},
        ]
        router = fake_llm(tool_args={"query": "ringkasan ISO Policy", "collection": "ISO Policy", "task": "summarize"})
        summary = {"pdf_documents": [], "merged_results": [], "search_analysis": {}}
        with patch.object(processor, "get_llm", return_value=(router, "gemini/x")), \
                patch.object(processor, "build_summary_results", return_value=summary) as build, \
                patch.object(processor, "generate_hybrid_answer", return_value=("Ringkasan", "gemini/x", {})) as gen:
            response = self.ask("ok", memory=memory)
        self.assertEqual(response.json()["answer"], "Ringkasan")
        build.assert_called_once_with(["c1"])
        self.hybrid_search.assert_not_called()
        self.assertEqual(gen.call_args[0][1], "ringkasan ISO Policy")
        self.assertEqual(gen.call_args[0][-2:], ("id", "summarize"))

    def test_router_search_without_source_asks_for_one(self):
        router = fake_llm(tool_args={"query": "leave policy", "task": "answer"})
        with patch.object(processor, "get_llm", return_value=(router, "gemini/x")):
            response = self.ask("what is the leave policy?", include_pdf_results=False)
        self.assertEqual(response.json()["answer"], no_source_selected("en"))
        self.assertIn("Sources switched on for this message: none", router.invoke.call_args[0][0])
        self.hybrid_search.assert_not_called()

    def test_capped_user_gray_zone_without_source_gets_free_reply(self):
        with patch.object(agnostic_service, "enforce_rate_limit", side_effect=HTTPException(status_code=429, detail="x")):
            response = self.ask("what is the leave policy?", include_pdf_results=False)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_used"], "system/no-source-selected")

    def test_help_command_is_deterministic(self):
        with patch.object(processor, "get_llm") as get_llm:
            response = self.ask("/help")
        self.assertEqual(response.json()["model_used"], "system/meta-help")
        self.assertIn("What you can do in DocuLens", response.json()["answer"])
        get_llm.assert_not_called()


class ConcurrencyEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = FastAPI()
        self.app.include_router(agnostic.router, prefix="/api/v1")
        self.app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="u1", email="u1@example.com", role="admin", is_active=True
        )
        for target, attr, kwargs in [
            (agnostic_service.supabase_storage, "list_collection_titles_for_user", {"return_value": [("c1", "ISO Policy")]}),
            (agnostic_service, "resolve_workspace_id", {"return_value": "w1"}),
            (agnostic_service, "enforce_rate_limit", {}),
            (agnostic_service, "enforce_plan_limit", {"return_value": None}),
            (agnostic_service, "enforce_member_allocation", {"return_value": None}),
            (agnostic_service, "get_usage_snapshot", {"return_value": None}),
            (agnostic_service, "log_token_usage", {}),
        ]:
            patcher = patch.object(target, attr, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_llm_calls_in_one_workspace_run_in_parallel(self):
        import asyncio
        import time
        import httpx

        def slow_route(*args, **kwargs):
            time.sleep(0.5)
            return RouteDecision(action="reply", answer="Hi!", model_id="gemini/x", total_tokens=10)

        with patch.object(processor, "route_message", side_effect=slow_route):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://t") as client:
                started = time.perf_counter()
                responses = await asyncio.gather(*(
                    client.post("/api/v1/agnostic/query", json={"question": "hi"}) for _ in range(4)
                ))
                elapsed = time.perf_counter() - started
        self.assertTrue(all(r.status_code == 200 for r in responses))
        # Serialized under the lock this took 4 x 0.5s = 2s+.
        self.assertLess(elapsed, 1.5)
        self.assertEqual(inflight.in_flight_tokens("w1", "u1"), (0, 0))

    async def test_reservation_released_when_llm_fails(self):
        import httpx

        with patch.object(processor, "route_message", side_effect=RuntimeError("boom")):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://t") as client:
                response = await client.post("/api/v1/agnostic/query", json={"question": "hi"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(inflight.in_flight_tokens("w1", "u1"), (0, 0))


if __name__ == "__main__":
    unittest.main()
