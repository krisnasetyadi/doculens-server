# billing/enforcement.py
"""The checks run before an LLM call: plan limit, member allocation, gap-check plan."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

import db
from billing.allocation import exceeds_cap, get_user_allocation
from billing.ledger import sum_tokens_for_admin, sum_tokens_for_user
from billing.plans import get_enforced_window, plan_includes_gap_check, resolve_admin_user_id
from billing.quotas import build_quota_tiers, get_quota_configs, quota_windows_for_user, sum_tokens_for_windows
from config import config
from router.auth import UserRecord


def enforce_gap_check_plan(user: UserRecord) -> None:
    """Raise HTTPException(403) unless the user's workspace is on a plan in
    GAP_CHECK_PLAN_IDS for the current (enforced) period — an expired Team
    plan has dropped back to the Free window, so it no longer qualifies.
    Fails open if the metering DB is unreachable, same as the other checks."""
    conn = db.get_conn("payment")
    if not conn:
        return
    try:
        admin_user_id = resolve_admin_user_id(conn, user)
        window = get_enforced_window(conn, admin_user_id)
    finally:
        conn.close()
    if window is None:
        return
    if not plan_includes_gap_check(window):
        raise HTTPException(
            status_code=403,
            detail=(
                f"Compliance Gap Check hanya tersedia untuk plan Individual dan Team (workspace kamu "
                f"saat ini: {window.plan['name']}). Upgrade plan untuk memakai fitur ini."
            ),
        )


def enforce_plan_limit(user: UserRecord, pending_tokens: int = 0, reserve: Optional[int] = None) -> Optional[int]:
    """Raise HTTPException(402) once the WHOLE workspace has used up its
    plan's own token_limit for the current period (MS-248 follow-up) —
    Free/Individual/Team's own ceiling, separate from the flat per-user
    safety net (enforce_rate_limit) and the optional per-member allocation
    (enforce_member_allocation). Every workspace always has a plan window
    now (Free is the fallback in get_latest_plan_window when nobody's
    paid), so this always applies, not just to paying workspaces. Fails
    open if the metering DB is unreachable, same as the other checks.
    `pending_tokens`: the workspace's in-flight reservations; `reserve`:
    this request's own headroom (defaults to config.query_token_reserve).
    Returns the workspace token_limit it enforced (None if it couldn't
    check), for begin_in_flight to clamp the reservation against."""
    conn = db.get_conn("payment")
    if not conn:
        return None
    try:
        admin_user_id = resolve_admin_user_id(conn, user)
        window = get_enforced_window(conn, admin_user_id)
        if not window:
            return None
        used = sum_tokens_for_admin(conn, admin_user_id, window.period_start, window.period_end)
        token_limit = window.plan["token_limit"]
        plan_name = window.plan["name"]
    finally:
        conn.close()
    own_reserve = config.query_token_reserve if reserve is None else reserve
    if exceeds_cap(used + pending_tokens, token_limit, own_reserve):
        raise HTTPException(
            status_code=402,
            detail=f"Jatah token workspace untuk plan {plan_name} sudah habis untuk periode ini.",
        )
    return token_limit


def enforce_member_allocation(user: UserRecord, pending_tokens: int = 0, reserve: Optional[int] = None) -> Optional[int]:
    """Raise HTTPException(403) once this user can't fit another query into
    their token cap this period. Every team member is capped (MS-402): by
    their token_allocations row, or by the workspace's Default Token
    Allocation when they have none — see get_user_allocation. An admin (or
    a self-registered account) is only capped if they have an explicit row
    — for an admin, a slice of the pool they allocated themselves (MS-248
    follow-up), which they can always raise back up since they control it.
    Fails open if the metering DB is unreachable, same as enforce_rate_limit.
    `pending_tokens`: this user's in-flight reservations; `reserve`: as in
    enforce_plan_limit. Returns the allocation it enforced (None when the
    user is uncapped or it couldn't check), like enforce_plan_limit."""
    conn = db.get_conn("payment")
    if not conn:
        return None
    try:
        allocation = get_user_allocation(conn, user)
        if allocation is None:
            return None
        admin_user_id = resolve_admin_user_id(conn, user)
        window = get_enforced_window(conn, admin_user_id)
        if not window:
            return
        quota_row = get_quota_configs(conn, [user.user_id]).get(user.user_id)
        if quota_row:
            now = datetime.now(timezone.utc)
            windows = quota_windows_for_user(user.user_id, quota_row["quota_anchor_at"], now)
            used_by_window = sum_tokens_for_windows(conn, windows)
            tiers = build_quota_tiers(user.user_id, allocation[0], quota_row, used_by_window, now)
            blocked = [tier for tier in tiers if tier.blocked]
            if blocked:
                names = ", ".join(tier.interval.capitalize() for tier in blocked)
                # All active limits must clear, so the latest blocked reset
                # is more useful than promising the earliest one will help.
                reset_at = max(tier.next_reset_date for tier in blocked)
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient Tokens: {names} limit reached. Resets at {reset_at}.",
                )
            return
        used = sum_tokens_for_user(conn, user.user_id, window.period_start, window.period_end)
    finally:
        conn.close()
    own_reserve = config.query_token_reserve if reserve is None else reserve
    if exceeds_cap(used + pending_tokens, allocation[0], own_reserve):
        raise HTTPException(
            status_code=403,
            detail="Token cap yang diberikan admin untuk akun kamu sudah habis untuk periode ini.",
        )
    return allocation[0]
