"""MS-667: the user stops a question while it is being answered (the browser
aborts the request). Before the model is called nothing more runs and no
tokens are counted; a model call already running finishes, its tokens are
counted, and no answer is sent. Either way nothing is held afterwards."""

import asyncio
from unittest.mock import patch

from models import AgnosticQueryRequest
from services import agnostic as agnostic_service
from services.billing import inflight
from services.rag.conversation.intent import RouteDecision
from services.rag.processor import processor
from tests.test_agnostic import QUESTION, AgnosticTestCase


class StopTest(AgnosticTestCase):
    def setUp(self):
        super().setUp()
        self.gone = False

    async def is_disconnected(self):
        return self.gone

    def stop_now(self, *returned):
        """A side effect that stops the question while the step runs."""
        def step(*args, **kwargs):
            self.gone = True
            return returned[0] if len(returned) == 1 else returned
        return step

    def run_query(self, question=QUESTION):
        req = AgnosticQueryRequest(question=question)
        with self.assertNoLogs("services.agnostic", level="ERROR"):
            return asyncio.run(agnostic_service.answer_query(
                req, "http://test", self.user, is_disconnected=self.is_disconnected))

    def assert_nothing_held(self):
        self.assertEqual(inflight.in_flight_tokens("w1", "u1"), (0, 0))

    def test_stopped_while_retrieving_never_calls_the_model(self):
        self.m["search"].side_effect = self.stop_now({})
        response = self.run_query()
        self.assertEqual(response.status_code, agnostic_service.STOPPED_STATUS)
        self.m["answer"].assert_not_called()
        self.m["log"].assert_not_called()
        self.assert_nothing_held()

    def test_stopped_while_the_model_answers_counts_only_its_tokens(self):
        self.m["answer"].side_effect = self.stop_now("Answer", "gemini/x", {"total_tokens": 120})
        response = self.run_query()
        self.assertEqual(response.status_code, agnostic_service.STOPPED_STATUS)
        self.assertEqual(response.body, b"")
        self.m["log"].assert_called_once()
        self.assertEqual(self.m["log"].call_args[0][:2], ("u1", 120))
        self.assert_nothing_held()

    def test_stopped_during_the_router_call_counts_it_and_answers_nothing(self):
        decision = RouteDecision(action="search", search_query="leave policy", model_id="gemini/x", total_tokens=30)
        with patch.object(processor, "route_message", side_effect=self.stop_now(decision)):
            response = self.run_query("hmm, and the other one?")  # not retrieval by rule: goes to the router
        self.assertEqual(response.status_code, agnostic_service.STOPPED_STATUS)
        self.m["search"].assert_not_called()
        self.m["answer"].assert_not_called()
        self.assertEqual(self.m["log"].call_args[0][:2], ("u1", 30))
        self.assert_nothing_held()

    def test_stopped_before_the_router_call_costs_nothing(self):
        self.m["usage"].side_effect = self.stop_now(None)
        with patch.object(processor, "route_message") as route:
            response = self.run_query("hmm, and the other one?")
        self.assertEqual(response.status_code, agnostic_service.STOPPED_STATUS)
        route.assert_not_called()
        self.m["log"].assert_not_called()
        self.assert_nothing_held()

    def test_a_question_that_is_not_stopped_is_answered_as_before(self):
        self.m["answer"].return_value = ("Answer", "gemini/x", {"total_tokens": 50})
        response = self.run_query()
        self.assertEqual(response.answer, "Answer")
        self.assertEqual(self.m["log"].call_args[0][:2], ("u1", 50))
        self.assert_nothing_held()

    def test_the_next_question_can_be_sent_right_after_a_stop(self):
        self.m["search"].side_effect = self.stop_now({})
        self.run_query()
        self.gone = False
        self.m["search"].side_effect = None
        self.m["search"].return_value = {}
        self.assertEqual(self.run_query().answer, "Answer")

    def test_the_route_passes_the_request_disconnect_check(self):
        with patch.object(agnostic_service, "answer_query", wraps=agnostic_service.answer_query) as answer_query:
            self.assertEqual(self.ask().status_code, 200)
        self.assertTrue(callable(answer_query.call_args.kwargs["is_disconnected"]))
