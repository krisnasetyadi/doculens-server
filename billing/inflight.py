# billing/inflight.py
"""Workspace locks and in-flight token reservations around LLM calls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import db
from billing.plans import resolve_admin_user_id
from router.auth import UserRecord


# Serializes a workspace's check-work-log cycle across enforce_rate_limit /
# enforce_plan_limit / enforce_member_allocation so concurrent requests
# can't all read "not yet blocked" before any of them has logged its
# usage — the classic check-then-act race a flat "check once, log after
# the LLM call returns" design otherwise leaves open.
#
# Keyed by WORKSPACE (the resolved admin_user_id — see resolve_workspace_id
# below), not by the calling user_id: enforce_plan_limit and
# enforce_member_allocation both check state shared across an entire team
# (the workspace's total usage vs. its plan's token_limit, and the
# allocation pool carved out of it), not just the caller's own. Two
# different members of the same team racing concurrently must serialize
# against EACH OTHER for those checks to mean anything — locking only on
# each member's own user_id (as an earlier version of this did) would let
# them race straight past a shared workspace ceiling together. This still
# never serializes different workspaces against each other.
#
# In-process only (fine for this app's current single-worker deployment; a
# multi-worker/multi-instance deployment would need a DB-level lock
# instead — see the module docstring on the one-time-payment model this
# whole file is built on for the same "not built for scale yet" caveat).
# Entries are never evicted; each is one tiny asyncio.Lock object, an
# acceptable tradeoff at this app's scale rather than added complexity.
_workspace_locks: dict[str, asyncio.Lock] = {}


def resolve_workspace_id(user: UserRecord) -> str:
    """The lock key for `user`'s workspace — their own user_id if they're
    an admin, otherwise whichever admin created them. Opens its own
    short-lived connection; call this BEFORE acquiring the lock (it can't
    be resolved from inside it, since resolving it needs a query and the
    point of the lock is to serialize queries). Fails open to the caller's
    own user_id (still safe, just narrower-than-ideal serialization) if
    the metering DB is unreachable."""
    if user.role == "admin":
        return user.user_id
    conn = db.get_conn("payment")
    if not conn:
        return user.user_id
    try:
        return resolve_admin_user_id(conn, user)
    finally:
        conn.close()


def get_workspace_lock(workspace_id: str) -> asyncio.Lock:
    lock = _workspace_locks.get(workspace_id)
    if lock is None:
        lock = asyncio.Lock()
        _workspace_locks[workspace_id] = lock
    return lock


# In-flight reservations — what lets callers hold the workspace lock only
# for the CHECK, not for the whole LLM call. Holding it through the call
# (as agnostic.py/compliance.py used to) made every query in a workspace
# wait for the previous one's Gemini round trip: 5 concurrent questions
# took 2.5s/4.7s/7.0s/9.2s/11.4s (measured), and one gap-check run froze
# the whole team's chat for its full duration.
#
# Instead: under the lock, the enforce_* checks count every request that
# passed its check but hasn't logged usage yet as already spent
# (pending_tokens), then the caller registers its own reservation and
# releases the lock. Concurrent requests therefore can't all slip past a
# cap together — each sees the others' reservations — while the LLM calls
# themselves run in parallel.
#
# What a request reserves is its expected COST (config.query_in_flight_estimate
# for a chat, config.gap_check_token_reserve for a gap check), not the small
# gate headroom (config.query_token_reserve): with parallel calls, any cost
# a reservation under-counts can be overspent once per concurrent request.
# Each reservation is also clamped to half of the cap it counts against (as
# exceeds_cap does for the gate), so one big in-flight run — a 100k gap
# check on a 60k Free pool, or on a member's 5k allocation — can't by itself
# lock every other request out until it finishes.
#
# Mutated only from the event loop thread (never inside asyncio.to_thread),
# so plain dicts need no extra locking. In-process, like _workspace_locks.
_in_flight_workspace_tokens: dict[str, int] = {}
_in_flight_user_tokens: dict[str, int] = {}


@dataclass(frozen=True)
class InFlightReservation:
    workspace_id: str
    user_id: str
    workspace_tokens: int
    user_tokens: int


def reservation_amount(estimate: int, cap: Optional[int]) -> int:
    """`estimate`, clamped to half of `cap` (None/0 = uncapped)."""
    estimate = max(0, estimate)
    return min(estimate, cap // 2) if cap else estimate


def in_flight_tokens(workspace_id: str, user_id: str) -> tuple[int, int]:
    """(tokens reserved by in-flight requests in this workspace, by this
    user) — pass to the enforce_* checks as pending_tokens."""
    return _in_flight_workspace_tokens.get(workspace_id, 0), _in_flight_user_tokens.get(user_id, 0)


def begin_in_flight(
    workspace_id: str,
    user_id: str,
    estimate: int,
    workspace_cap: Optional[int] = None,
    user_cap: Optional[int] = None,
) -> InFlightReservation:
    """Register a request that just passed its checks. The caps are what
    enforce_plan_limit / enforce_member_allocation returned for it."""
    reservation = InFlightReservation(
        workspace_id=workspace_id,
        user_id=user_id,
        workspace_tokens=reservation_amount(estimate, workspace_cap),
        user_tokens=reservation_amount(estimate, user_cap),
    )
    _in_flight_workspace_tokens[workspace_id] = _in_flight_workspace_tokens.get(workspace_id, 0) + reservation.workspace_tokens
    _in_flight_user_tokens[user_id] = _in_flight_user_tokens.get(user_id, 0) + reservation.user_tokens
    return reservation


def end_in_flight(reservation: InFlightReservation) -> None:
    """Call AFTER log_token_usage — so between the two there's never a
    moment where a request's cost is counted neither as reserved nor as
    logged usage."""
    for store, key, tokens in (
        (_in_flight_workspace_tokens, reservation.workspace_id, reservation.workspace_tokens),
        (_in_flight_user_tokens, reservation.user_id, reservation.user_tokens),
    ):
        remaining = store.get(key, 0) - tokens
        if remaining > 0:
            store[key] = remaining
        else:
            store.pop(key, None)
