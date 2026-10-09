"""MS-504: upload size, batch size and workspace storage quota.

Three limits, all resolved per workspace plan (PLAN_QUOTAS in
router/payment.py) and all checked here on the server, whatever the client
already checked:

  * max_file_bytes       -- one file
  * max_batch_files      -- files in a single upload request
  * storage_limit_bytes  -- everything the workspace has stored

Used storage is never kept in a counter. It is SUM(collections.size_bytes)
over the admin and every member created by that admin, so deleting a source
frees its space with no bookkeeping.

The one thing a plain SUM cannot see is an upload that has passed its check
but has not been registered yet (processing can take a while). Those bytes
are held as a reservation for the workspace until the upload finishes, so
two uploads racing for the last free space cannot both get in.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import BinaryIO, Optional

from fastapi import HTTPException

import db
import storage as supabase_storage
from config import config

logger = logging.getLogger(__name__)

MB = 1024 * 1024
GB = 1024 * MB

_COPY_CHUNK = 1 * MB
# A reservation whose owner never released it (crash, abandoned request) must
# not block the workspace forever.
_RESERVATION_TTL_SECONDS = 30 * 60


@dataclass(frozen=True)
class StorageLimits:
    plan_name: str
    storage_limit_bytes: int
    max_file_bytes: int
    max_batch_files: int


def default_limits(plan_name: str = "Default") -> StorageLimits:
    return StorageLimits(
        plan_name=plan_name,
        storage_limit_bytes=config.storage_quota_bytes,
        max_file_bytes=config.max_file_size_bytes,
        max_batch_files=config.max_batch_files,
    )


def format_size(num_bytes: int) -> str:
    """'50 MB', '1.5 GB'. Whole numbers stay whole so messages read like the
    ticket copy ("50 MB maximum size limit"). Rounds down, like the UI, so the
    two never disagree."""
    if num_bytes >= GB:
        value, unit = num_bytes / GB, "GB"
    else:
        value, unit = num_bytes / MB, "MB"
    text = f"{math.floor(value * 10 + 1e-9) / 10:.1f}".rstrip("0").rstrip(".")
    return f"{text or '0'} {unit}"


# ---------------------------------------------------------------------------
# Errors. Status code is the contract with the UI (it picks the banner from
# it); the detail is a plain string so every existing error parser keeps
# working.
# ---------------------------------------------------------------------------

def file_too_large_error(limits: StorageLimits) -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=f"File exceeds the {format_size(limits.max_file_bytes)} maximum size limit.",
    )


def batch_too_large_error(limits: StorageLimits) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"You can only upload up to {limits.max_batch_files} files at a time.",
    )


def quota_exceeded_error() -> HTTPException:
    return HTTPException(
        status_code=402,
        detail="Storage Limit Reached. Please delete older files to free up space.",
    )


# ---------------------------------------------------------------------------
# Used storage
# ---------------------------------------------------------------------------

def get_used_bytes(conn, admin_user_id: str) -> int:
    """Bytes stored by the workspace: the admin's own sources plus every
    member's. Sources with no owner (WhatsApp exports uploaded before
    MS-504) cannot be attributed to a workspace and are not counted."""
    supabase_storage.ensure_schema()  # size_bytes may not exist yet on first call
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(SUM(size_bytes), 0) AS used FROM collections
            WHERE owner_id = %s
               OR owner_id IN (SELECT user_id FROM users WHERE created_by = %s)
            """,
            (admin_user_id, admin_user_id),
        )
        row = cur.fetchone()
    return int(row["used"])


# ---------------------------------------------------------------------------
# Reservations
# ---------------------------------------------------------------------------

_registry_lock = Lock()
_workspace_locks: dict[str, Lock] = {}
_reservations: dict[str, dict[str, tuple[int, float]]] = {}


def _lock_for(workspace_id: str) -> Lock:
    with _registry_lock:
        lock = _workspace_locks.get(workspace_id)
        if lock is None:
            lock = _workspace_locks[workspace_id] = Lock()
        return lock


def _pending_bytes(workspace_id: str) -> int:
    """Caller holds the workspace lock."""
    now = time.monotonic()
    held = _reservations.get(workspace_id, {})
    for token in [t for t, (_, expires) in held.items() if expires <= now]:
        del held[token]
    return sum(size for size, _ in held.values())


class Reservation:
    """Space set aside for one in-flight upload. Release it once the upload
    is registered (the row then counts by itself) or has failed."""

    def __init__(self, workspace_id: str, token: Optional[str], size: int):
        self.workspace_id = workspace_id
        self.token = token
        self.size = size

    def release(self) -> None:
        if self.token is None:
            return
        with _lock_for(self.workspace_id):
            _reservations.get(self.workspace_id, {}).pop(self.token, None)
        self.token = None


def reserve(conn, workspace_id: str, limits: StorageLimits, size: int) -> Reservation:
    """Raise 402 unless `size` more bytes fit in the workspace's quota, else
    hold them until the returned Reservation is released.

    The stored total is read while holding the workspace lock: a competing
    upload registers its row before it releases its own reservation, so
    every byte is seen at least once, by one or the other. Errs towards
    counting twice for an instant, never towards missing it."""
    if conn is None:
        # Without the metering database there is nothing to measure against;
        # size and batch limits still apply.
        logger.warning("storage quota not enforced: database unavailable")
        return Reservation(workspace_id, None, size)

    with _lock_for(workspace_id):
        used = get_used_bytes(conn, workspace_id)
        if used + _pending_bytes(workspace_id) + size > limits.storage_limit_bytes:
            raise quota_exceeded_error()
        token = uuid.uuid4().hex
        _reservations.setdefault(workspace_id, {})[token] = (
            size, time.monotonic() + _RESERVATION_TTL_SECONDS,
        )
    return Reservation(workspace_id, token, size)


def reserve_workspace(workspace_id: str, limits: StorageLimits, size: int) -> Reservation:
    """reserve(), opening (and closing) its own connection to the app database."""
    conn = db.get_conn("storage_limits")
    try:
        return reserve(conn, workspace_id, limits, size)
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# Measuring and saving
# ---------------------------------------------------------------------------

def measure(fileobj: BinaryIO) -> int:
    """Real size of an uploaded file, taken from the bytes themselves and not
    from any client-supplied header."""
    position = fileobj.tell()
    fileobj.seek(0, os.SEEK_END)
    size = fileobj.tell()
    fileobj.seek(position)
    return size


def save_capped(fileobj: BinaryIO, destination: str, max_bytes: int, limits: StorageLimits) -> int:
    """Copy `fileobj` to `destination`, giving up (and deleting the partial
    file) the moment it passes `max_bytes`. Returns the bytes written."""
    written = 0
    try:
        with open(destination, "wb") as out:
            while True:
                chunk = fileobj.read(_COPY_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise file_too_large_error(limits)
                out.write(chunk)
    except BaseException:
        if os.path.exists(destination):
            os.remove(destination)
        raise
    return written


def remove_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
