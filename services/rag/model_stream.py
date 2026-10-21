"""A model call streamed chunk by chunk (MS-906).

`llm.stream` is a blocking generator that waits on the network between
chunks, so it cannot run on the event loop. ModelStream runs it in a thread of
its own and hands the text over through a queue. That also lets the caller
stop it, and read what the chunks received so far cost.
"""
import asyncio
import logging
import threading
from typing import Any, AsyncIterator, Callable, Optional

from services.rag.answer import text_of
from services.rag.shared import _extract_total_tokens

logger = logging.getLogger(__name__)

# The worker only notices a stop when the model sends its next chunk. A
# stopped stream is given this long to do so, so that chunk's tokens are
# counted too.
STOP_GRACE_SECONDS = 5.0

_END = object()


class ModelStream:
    """One streamed model call. Iterate `chunks()` for the text; `total_tokens`
    is the sum of the usage every chunk reported (Gemini reports a delta per
    chunk, and 0 on the last one, so the total is never read from the last
    chunk alone). A provider that streams nothing smaller than a whole reply
    (HuggingFace) gives one chunk and 0 tokens."""

    def __init__(self, llm: Any, prompt: str):
        self._llm = llm
        self._prompt = prompt
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = asyncio.Event()
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.total_tokens = 0

    async def chunks(self) -> AsyncIterator[str]:
        """The reply's text, in order. Raises what the model call raised."""
        self._start()
        while True:
            item = await self._queue.get()
            if item is _END:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    async def stop(self) -> bool:
        """Stop the model call and wait for the worker to finish. Safe to call
        more than once, and before `chunks()` was ever iterated. False when the
        worker was still running after STOP_GRACE_SECONDS (it is blocked on the
        network until the model sends its next chunk): the call is then still
        costing tokens, see `wait_finished`."""
        self._stopped.set()
        if self._thread is None:
            return True
        return await self.wait_finished(STOP_GRACE_SECONDS)

    async def wait_finished(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds for the worker to end. `total_tokens` is
        final once this returns True."""
        try:
            await asyncio.wait_for(self._finished.wait(), timeout)
        except asyncio.TimeoutError:
            logger.warning("model stream did not stop within %.0fs", timeout)
            return False
        return True

    def _start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run, name="model-stream", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """The worker thread: forward every chunk until the reply ends, the
        model fails or the stream is stopped."""
        chunks = None
        try:
            chunks = self._llm.stream(self._prompt)
            for chunk in chunks:
                # Counted before the stop check: a chunk that arrived was paid for.
                self.total_tokens += _extract_total_tokens(chunk)
                if self._stopped.is_set():
                    break
                text = text_of(chunk)
                if text:
                    self._hand_over(self._queue.put_nowait, text)
        except Exception as exc:
            self._hand_over(self._queue.put_nowait, exc)
        finally:
            self._close(chunks)
            self._hand_over(self._queue.put_nowait, _END)
            self._hand_over(self._finished.set)

    def _hand_over(self, call: Callable[..., None], *args: Any) -> None:
        """Run `call` on the event loop (asyncio objects are not thread-safe)."""
        try:
            self._loop.call_soon_threadsafe(call, *args)
        except RuntimeError:
            pass  # the loop is closed: the server is shutting down, nobody is listening

    @staticmethod
    def _close(chunks: Any) -> None:
        """Close the provider's generator so it drops the connection to the model."""
        close = getattr(chunks, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:
            logger.warning("closing the model stream failed", exc_info=True)
