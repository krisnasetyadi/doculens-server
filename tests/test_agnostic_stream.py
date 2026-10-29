"""MS-906: POST /agnostic/query?stream=true answers with NDJSON events
(sources, delta..., done | error) instead of one JSON body. The search and every
check run before the first byte, the finished `done.result` equals the
non-streamed response, usage is logged exactly once however the stream ends,
and the in-flight reservation is always freed. The model is faked."""

import asyncio
import gc
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from langchain_core.messages import AIMessage, AIMessageChunk
from starlette.requests import ClientDisconnect

from models import AgnosticQueryRequest
from services import agnostic as agnostic_service
from services.billing import inflight
from services.rag.processor import processor
from tests.test_agnostic import QUESTION, AgnosticTestCase

STREAM_URL = "/api/v1/agnostic/query?stream=true"
PIECES = ["Leave is ", "twelve days ", "per year, as policy ", "section 4 states."]
ANSWER = "".join(PIECES)


def usage(tokens):
    return {"input_tokens": 0, "output_tokens": tokens, "total_tokens": tokens}


class FakeModel:
    """Writes PIECES, 3 tokens each, then a last empty chunk that reports 0 (as
    Gemini does). `fail_at` raises instead of writing that piece; `gate` holds the
    model back before its second piece."""

    def __init__(self, pieces=PIECES, fail_at=None, gate=None):
        self.pieces = pieces
        self.fail_at = fail_at
        self.gate = gate

    def invoke(self, prompt):
        if self.fail_at is not None:
            raise RuntimeError("model down")
        return AIMessage(content="".join(self.pieces), usage_metadata=usage(3 * len(self.pieces)))

    def stream(self, prompt):
        for i, piece in enumerate(self.pieces):
            if i == 1 and self.gate is not None:
                self.gate.wait(2)
            if i == self.fail_at:
                raise RuntimeError("model down")
            yield AIMessageChunk(content=piece, usage_metadata=usage(3))
        yield AIMessageChunk(content="", usage_metadata=usage(0))


class StreamTestCase(AgnosticTestCase):
    def setUp(self):
        super().setUp()
        # The real answer code runs; only the model behind it is faked.
        real_generate = type(processor).generate_hybrid_answer
        self.m["answer"].side_effect = lambda *args: real_generate(processor, *args)
        self.model = FakeModel()
        patch.object(processor, "get_llm", side_effect=lambda *a: (self.model, "gemini/x")).start()

    def stream(self, question=QUESTION, **body):
        response = self.http.post(STREAM_URL, json={"question": question, **body})
        events = [json.loads(line) for line in response.text.splitlines()]
        return response, events

    def assert_nothing_held(self):
        self.assertEqual(inflight.in_flight_tokens("w1", "u1"), (0, 0))


class StreamEventsTest(StreamTestCase):
    def test_sources_then_deltas_then_done_as_ndjson(self):
        response, events = self.stream()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/x-ndjson"))
        self.assertEqual(response.headers["cache-control"], "no-cache")
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        self.assertEqual([e["type"] for e in events], ["sources"] + ["delta"] * len(PIECES) + ["done"])
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "delta"), ANSWER)
        self.assertEqual(events[0]["source_type"], "Indexed Collections")
        self.assertEqual(events[-1]["result"]["answer"], ANSWER)

    def test_done_is_the_same_response_the_endpoint_returns_without_stream(self):
        _, events = self.stream()
        plain = self.ask().json()
        done = events[-1]["result"]
        for body in (done, plain):
            body.pop("processing_time")
        self.assertEqual(done, plain)

    def test_done_replaces_the_streamed_text_when_the_answer_fails_validation(self):
        self.model = FakeModel(["Sorry, ", "I don't know ", "the answer to that."])
        self.m["search"].return_value = {"merged_results": [
            {"type": "pdf", "source": "policy.pdf", "content": "Leave is twelve days per year.", "confidence": 0.9},
        ]}
        _, events = self.stream()
        streamed = "".join(e["text"] for e in events if e["type"] == "delta")
        self.assertTrue(streamed.startswith("Sorry"))
        self.assertIn("policy.pdf", events[-1]["result"]["answer"])
        self.assertNotEqual(events[-1]["result"]["answer"], streamed)

    def test_a_canned_reply_is_a_lone_done_event(self):
        response, events = self.stream("/help")
        self.assertEqual([e["type"] for e in events], ["done"])
        self.assertEqual(events[0]["result"]["model_used"], "system/meta-help")
        self.assertTrue(response.headers["content-type"].startswith("application/x-ndjson"))
        self.m["search"].assert_not_called()

    def test_a_refusal_before_the_stream_is_an_ordinary_http_error(self):
        self.m["rate"].side_effect = HTTPException(status_code=429, detail="Too many questions")
        response = self.http.post(STREAM_URL, json={"question": QUESTION})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"], "Too many questions")
        self.assertTrue(response.headers["content-type"].startswith("application/json"))
        self.assert_nothing_held()

    def test_an_invalid_request_is_a_422(self):
        self.assertEqual(self.http.post(STREAM_URL, json={"question": ""}).status_code, 422)

    def test_a_failure_while_searching_is_a_500_and_frees_the_reservation(self):
        self.m["search"].side_effect = RuntimeError("index missing")
        response = self.http.post(STREAM_URL, json={"question": QUESTION})
        self.assertEqual(response.status_code, 500)
        self.assert_nothing_held()


class StreamUsageTest(StreamTestCase):
    def test_usage_is_summed_over_the_chunks_and_logged_once_before_done(self):
        logged_before_done = []
        real_event = agnostic_service._event
        self.m["log"].side_effect = lambda *a: None

        def spy(kind, **fields):
            if kind == "done":
                logged_before_done.append(self.m["log"].call_count)
            return real_event(kind, **fields)

        with patch.object(agnostic_service, "_event", spy):
            self.stream()
        self.m["log"].assert_called_once()
        self.assertEqual(self.m["log"].call_args[0][:2], ("u1", 3 * len(PIECES)))
        self.assertEqual(logged_before_done, [1])
        self.assert_nothing_held()

    def test_a_model_that_fails_before_any_text_gives_the_same_fallback_answer_as_without_stream(self):
        self.model = FakeModel(fail_at=0)
        _, events = self.stream()
        self.assertEqual([e["type"] for e in events], ["sources", "done"])
        self.assertEqual(events[-1]["result"]["answer"], self.ask().json()["answer"])
        self.assert_nothing_held()

    def test_a_failure_validating_the_finished_text_gives_the_same_fallback_answer_as_without_stream(self):
        with patch.object(processor, "_validate_and_clean_answer", side_effect=RuntimeError("validator down")):
            _, events = self.stream()
            plain = self.ask().json()
        self.assertEqual([e["type"] for e in events], ["sources"] + ["delta"] * len(PIECES) + ["done"])
        self.assertEqual(events[-1]["result"]["answer"], plain["answer"])
        self.assertNotEqual(events[-1]["result"]["answer"], ANSWER)
        self.assert_nothing_held()

    def test_a_model_that_fails_after_text_went_out_ends_with_error_and_counts_what_was_used(self):
        self.model = FakeModel(fail_at=2)
        _, events = self.stream()
        self.assertEqual([e["type"] for e in events], ["sources", "delta", "delta", "error"])
        self.assertEqual(events[-1]["detail"], "Failed to answer the question")
        self.m["log"].assert_called_once()
        self.assertEqual(self.m["log"].call_args[0][:2], ("u1", 6))
        self.assert_nothing_held()

    def test_efficiency_figures_are_logged_with_the_streamed_answer(self):
        self.stream(efficient_mode=True)
        self.m["log"].assert_called_once()
        self.assertEqual(self.m["log"].call_args[0][2], True)


class StreamLeaveTest(StreamTestCase):
    """The client leaving: the server cancels the response, which closes the
    event generator (or never starts it)."""

    def test_leaving_mid_answer_counts_the_tokens_received_and_frees_the_reservation(self):
        gate = threading.Event()
        self.model = FakeModel(gate=gate)

        pending = self.pending_in_loop()

        async def leave_after_the_first_text():
            events = agnostic_service._AnswerStream(pending).events()
            self.assertEqual(json.loads(await anext(events))["type"], "sources")
            self.assertEqual(json.loads(await anext(events))["text"], PIECES[0])
            threading.Timer(0.05, gate.set).start()  # the next chunk arrives while stopping
            await events.aclose()

        asyncio.run(leave_after_the_first_text())
        self.m["log"].assert_called_once()
        self.assertEqual(self.m["log"].call_args[0][:2], ("u1", 3 + 3))  # the first piece and the late second
        self.assert_nothing_held()

    def test_leaving_before_the_first_byte_still_frees_the_reservation(self):
        """The generator never runs, so only the response itself can settle."""
        response = agnostic_service._StreamedAnswer(agnostic_service._AnswerStream(self.pending_in_loop()))
        self.assertGreater(inflight.in_flight_tokens("w1", "u1")[1], 0)

        async def gone(message):
            raise OSError("client gone")

        async def never():
            pass

        scope = {"type": "http", "asgi": {"spec_version": "2.4"}}  # a server that reports a send failure
        with self.assertRaises(ClientDisconnect):
            asyncio.run(response(scope, never, gone))
        self.assert_nothing_held()
        self.m["log"].assert_not_called()

    def test_settling_twice_counts_once(self):
        answer = agnostic_service._AnswerStream(self.pending_in_loop())
        asyncio.run(answer.settle())
        asyncio.run(answer.settle())
        self.assert_nothing_held()

    def test_the_router_call_that_brought_the_message_here_is_counted_when_the_client_leaves(self):
        pending = self.pending_in_loop()
        pending.routing_tokens = 30
        answer = agnostic_service._AnswerStream(pending)
        asyncio.run(answer.settle())
        self.assertEqual(self.m["log"].call_args[0][:2], ("u1", 30))
        self.assert_nothing_held()

    def test_a_model_that_had_not_stopped_keeps_the_reservation_until_it_ends_and_counts_its_late_tokens(self):
        class LateModel:
            total_tokens = 3

            async def stop(self):
                return False  # still waiting for the network when it was stopped

            async def wait_finished(self, timeout):
                self.total_tokens = 7  # the chunk that arrived after the stop
                return True

        answer = agnostic_service._AnswerStream(self.pending_in_loop())
        answer._model = LateModel()
        answer._plan = SimpleNamespace(metadata={})

        async def leave():
            await answer.settle()
            self.assertEqual(self.m["log"].call_args[0][:2], ("u1", 3))
            self.assertNotEqual(inflight.in_flight_tokens("w1", "u1"), (0, 0))  # still held
            await asyncio.gather(*agnostic_service._late_settlements)

        asyncio.run(leave())
        self.assertEqual([call[0][:2] for call in self.m["log"].call_args_list], [("u1", 3), ("u1", 4)])
        self.assert_nothing_held()

    def test_an_answer_dropped_without_ever_running_still_frees_the_reservation(self):
        answer = agnostic_service._AnswerStream(self.pending_in_loop())
        self.assertNotEqual(inflight.in_flight_tokens("w1", "u1"), (0, 0))
        del answer
        gc.collect()
        self.assert_nothing_held()

    def pending_in_loop(self):
        """A PendingAnswer whose reservation is still held, as the route leaves it."""
        req = AgnosticQueryRequest(question=QUESTION)
        stop = agnostic_service.StopCheck()
        return asyncio.run(agnostic_service._answer_query(req, "http://test", self.user, stop))
