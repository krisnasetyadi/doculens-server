"""Coarse loading-bar progress for a source upload: stream a few stage
checkpoints over NDJSON while the upload runs, and keep the same checkpoints
reachable by GET so a page reload can pick the number back up.

Work runs detached from the SSE connection (via an executor), so a client
disconnect only drops the listener — the source still finishes processing
and registers normally. The progress snapshot lives in memory for this one
process only: it does not survive a server/worker restart.
"""

import asyncio
import json
import logging
from copy import deepcopy
from threading import Event, Lock
from time import monotonic
from typing import Callable, Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

ProgressCallback = Callable[[str, int], None]
logger = logging.getLogger(__name__)

# How long a finished record stays queryable — just long enough for a reload
# shortly after completion to catch it, not a real history.
_RETENTION_SECONDS = 10 * 60


class _UploadProgressStore:
    def __init__(self):
        self._records: dict[str, dict] = {}
        self._lock = Lock()

    def _prune(self):
        cutoff = monotonic() - _RETENTION_SECONDS
        self._records = {
            key: record for key, record in self._records.items()
            if record.get("_finished_at", float("inf")) > cutoff
        }

    def start(self, upload_id: str, *, owner_id: str):
        with self._lock:
            self._prune()
            self._records[upload_id] = {
                "status": "uploading", "stage": "reading", "progress": 10,
                "owner_id": owner_id,
            }

    def update(self, upload_id: str, event: dict):
        with self._lock:
            record = self._records.get(upload_id)
            if not record:
                return
            if event["type"] == "progress":
                record.update(stage=event["stage"], progress=event["progress"])
            elif event["type"] == "ready":
                record.update(status="success", stage="ready", progress=100,
                              result=event["result"], _finished_at=monotonic())
            elif event["type"] == "error":
                record.update(status="error", stage=None, progress=None,
                              error=event["detail"], _finished_at=monotonic())

    def get(self, upload_id: str, owner_id: str) -> Optional[dict]:
        with self._lock:
            record = self._records.get(upload_id)
            if not record or record["owner_id"] != owner_id:
                return None
            if record.get("_finished_at", float("inf")) <= monotonic() - _RETENTION_SECONDS:
                return None
            return deepcopy({k: v for k, v in record.items() if k not in ("owner_id", "_finished_at")})


upload_progress_store = _UploadProgressStore()


def progress_response(work: Callable[[ProgressCallback], dict], *, upload_id: str) -> StreamingResponse:
    """Stream actual work checkpoints, followed by exactly one ready/error event.

    Uploaded files must be saved before calling this: FastAPI can close
    multipart file handles before the iterator below is consumed.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    disconnected = Event()
    last_progress = 9
    last_stage = None

    def publish(event: dict):
        upload_progress_store.update(upload_id, event)
        if not disconnected.is_set():
            loop.call_soon_threadsafe(queue.put_nowait, event)

    def report(stage: str, progress: int):
        nonlocal last_progress, last_stage
        progress = max(last_progress, min(99, int(progress)))
        if progress == last_progress and stage == last_stage:
            return
        last_progress, last_stage = progress, stage
        publish({"type": "progress", "stage": stage, "progress": progress})

    def run():
        try:
            result = work(report)
            publish({"type": "ready", "progress": 100, "result": result})
        except HTTPException as exc:
            publish({"type": "error", "detail": exc.detail})
        except Exception:
            logger.exception("Source preparation failed")
            publish({"type": "error", "detail": "Failed to prepare source"})

    loop.run_in_executor(None, run)

    async def events():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    continue
                yield json.dumps({**event, "upload_id": upload_id}) + "\n"
                if event["type"] in ("ready", "error"):
                    break
        finally:
            disconnected.set()

    return StreamingResponse(events(), media_type="application/x-ndjson")
