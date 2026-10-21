"""MS-906: ModelStream runs a blocking llm.stream in a thread and hands the text
over to the event loop, counts the tokens of every chunk it receives, and can
be stopped between chunks."""

import asyncio
import threading
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk

from services.rag.model_stream import ModelStream


def chunk(text, tokens=0):
    return AIMessageChunk(
        content=text, usage_metadata={"input_tokens": 0, "output_tokens": tokens, "total_tokens": tokens},
    )


class ScriptedLLM:
    """llm.stream yields `chunks`; an Exception in the list is raised at that point.
    `closed` is set when the consumer closes the generator."""

    def __init__(self, chunks, gate=None):
        self.chunks = chunks
        self.gate = gate
        self.closed = threading.Event()

    def stream(self, prompt):
        try:
            for i, item in enumerate(self.chunks):
                if i == 1 and self.gate is not None:
                    self.gate.wait(2)
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            self.closed.set()


async def collect(stream):
    return [text async for text in stream.chunks()]


class ModelStreamTest(unittest.TestCase):
    def test_text_comes_in_order_and_tokens_are_summed_over_every_chunk(self):
        # Gemini reports a delta per chunk and 0 on the last one.
        llm = ScriptedLLM([chunk("Hello ", 3), chunk("world", 4), chunk("", 0)])
        stream = ModelStream(llm, "prompt")
        self.assertEqual(asyncio.run(collect(stream)), ["Hello ", "world"])
        self.assertEqual(stream.total_tokens, 7)

    def test_a_model_that_only_returns_strings_is_one_chunk_with_no_tokens(self):
        stream = ModelStream(ScriptedLLM(["The whole answer"]), "prompt")
        self.assertEqual(asyncio.run(collect(stream)), ["The whole answer"])
        self.assertEqual(stream.total_tokens, 0)

    def test_a_failing_model_raises_in_the_consumer_after_the_text_before_it(self):
        stream = ModelStream(ScriptedLLM([chunk("Hello ", 3), RuntimeError("model down")]), "prompt")
        received = []

        async def consume():
            async for text in stream.chunks():
                received.append(text)

        with self.assertRaisesRegex(RuntimeError, "model down"):
            asyncio.run(consume())
        self.assertEqual(received, ["Hello "])
        self.assertEqual(stream.total_tokens, 3)

    def test_stopped_after_the_first_chunk_counts_the_chunk_that_arrives_late_and_closes_the_model(self):
        gate = threading.Event()
        llm = ScriptedLLM([chunk("one ", 3), chunk("two ", 4), chunk("three", 5)], gate=gate)
        stream = ModelStream(llm, "prompt")

        async def run():
            chunks = stream.chunks()
            first = await anext(chunks)
            threading.Timer(0.05, gate.set).start()  # the model's next chunk arrives while stopping
            await stream.stop()
            await chunks.aclose()
            return first

        self.assertEqual(asyncio.run(run()), "one ")
        self.assertEqual(stream.total_tokens, 3 + 4)  # "three" is never read
        self.assertTrue(llm.closed.is_set())

    def test_stop_reports_whether_the_model_call_ended_and_a_late_one_can_be_waited_for(self):
        gate = threading.Event()
        llm = ScriptedLLM([chunk("one ", 3), chunk("two ", 4), chunk("three", 5)], gate=gate)
        stream = ModelStream(llm, "prompt")

        async def run():
            chunks = stream.chunks()
            await anext(chunks)
            with patch("services.rag.model_stream.STOP_GRACE_SECONDS", 0.05):
                ended_in_time = await stream.stop()  # the model is blocked: its next chunk has not arrived
            threading.Timer(0.05, gate.set).start()
            ended_later = await stream.wait_finished(2)
            await chunks.aclose()
            return ended_in_time, ended_later

        self.assertEqual(asyncio.run(run()), (False, True))
        self.assertEqual(stream.total_tokens, 3 + 4)  # the chunk that arrived late is counted once it is over

    def test_stopping_a_stream_that_never_started_is_harmless(self):
        stream = ModelStream(ScriptedLLM([chunk("x", 1)]), "prompt")
        asyncio.run(stream.stop())
        self.assertEqual(stream.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
