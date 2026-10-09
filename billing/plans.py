# billing/plans.py
"""Plans, prices and the subscription period a workspace is in."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException

import db
import storage_limits
from config import config
from models import StorageUsageResponse
from router.auth import UserRecord


# amount is already in Stripe's minor unit for IDR (Rupiah x 100) — IDR is a
# standard two-decimal currency for Stripe, not zero-decimal like JPY/KRW.
# Matches the marketing prices in chat-ui's lib/pricing-plans.ts. Enterprise
# is excluded — that plan is sales-assisted, not self-serve checkout.
PLAN_PRICES = {
    "individual": {"name": "Individual", "amount": 6_500_000},
    "team": {"name": "Team", "amount": 50_000_000},
}

# Token allowance granted per subscription period (MS-248). Same plan_ids as
# PLAN_PRICES above; Enterprise is sales-assisted/custom, no self-serve cap.
# "free" has no entry in PLAN_PRICES (nothing to check out) but does need one
# here — see get_latest_plan_window's fallback branch below — so a workspace
# that's never paid still gets a real, enforced cap instead of relying on
# the flat safety-net alone.
# MS-504: every plan also carries its upload/storage limits. They all start
# from the same configured defaults; give a plan its own numbers here to
# tier them (e.g. a smaller storage_limit_bytes for Free).
_STORAGE_DEFAULTS = {
    "storage_limit_bytes": config.storage_quota_bytes,
    "max_file_bytes": config.max_file_size_bytes,
    "max_batch_files": config.max_batch_files,
}

PLAN_QUOTAS = {
    "free": {"name": "Free", "token_limit": config.free_plan_token_limit, **_STORAGE_DEFAULTS},
    "individual": {"name": "Individual", "token_limit": 2_000_000, **_STORAGE_DEFAULTS},
    "team": {"name": "Team", "token_limit": 10_000_000, **_STORAGE_DEFAULTS},
}

# No real recurring billing exists yet (see module docstring — Checkout is
# mode="payment", one-time) — the "current period" is a fixed window rolling
# forward from the most recent succeeded payment, not a Stripe-driven renewal.
SUBSCRIPTION_PERIOD_DAYS = 30


def resolve_admin_user_id(conn, user: UserRecord) -> str:
    """The workspace a user's usage counts against: admins own their
    workspace; a sub-user's workspace is whoever created them (`created_by`,
    see router/auth.py). Falls back to the user's own id if `created_by` is
    somehow unset, so usage still resolves to *some* workspace."""
    if user.role == "admin":
        return user.user_id
    with conn.cursor() as cur:
        cur.execute("SELECT created_by FROM users WHERE user_id = %s", (user.user_id,))
        row = cur.fetchone()
    return (row["created_by"] if row else None) or user.user_id


@dataclass
class PlanWindow:
    plan: dict
    payment_id: Optional[str]  # None for the synthetic free-tier window below
    period_start: datetime
    period_end: datetime
    status: str  # "active" | "expired" — free tier is always "active"
    cancel_at_period_end: bool


def get_latest_plan_window(conn, admin_user_id: str) -> Optional[PlanWindow]:
    """The admin's most recent succeeded payment and the period it grants —
    or, if they've never paid successfully (or the plan_id on file isn't
    one we recognize), a synthetic Free-tier window (MS-248 follow-up) so
    every workspace has a real, enforced cap instead of relying on the
    flat safety-net rate limit alone. Only returns None if the admin's own
    user row can't be found at all, which shouldn't normally happen.
    `status` here only ever reflects whether a PAID period's `period_end`
    has passed — cancellation doesn't cut a period short (see
    cancel_at_period_end), it just stops it renewing. Free tier has
    nothing to expire from; it just keeps rolling to the next period."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT payment_id, plan_id, created_at, cancelled_at FROM payments
            WHERE user_id = %s AND status = 'succeeded'
            ORDER BY created_at DESC LIMIT 1
            """,
            (admin_user_id,),
        )
        row = cur.fetchone()
    plan = PLAN_QUOTAS.get(row["plan_id"]) if row else None

    if row and plan:
        period_start = row["created_at"]
        period_end = period_start + timedelta(days=SUBSCRIPTION_PERIOD_DAYS)
        now = datetime.now(timezone.utc)
        return PlanWindow(
            plan=plan,
            payment_id=row["payment_id"],
            period_start=period_start,
            period_end=period_end,
            status="active" if now <= period_end else "expired",
            cancel_at_period_end=row["cancelled_at"] is not None,
        )

    # Free tier: no succeeded payment on file (or an unrecognized plan_id).
    # Anchor the rolling window to the workspace admin's own account
    # creation date, since there's no payment date to anchor to, and
    # advance it in fixed SUBSCRIPTION_PERIOD_DAYS steps so a long-lived
    # free account keeps getting fresh periods automatically forever
    # rather than being stuck in (or blocked by) its very first one.
    with conn.cursor() as cur:
        cur.execute("SELECT created_at FROM users WHERE user_id = %s", (admin_user_id,))
        user_row = cur.fetchone()
    if not user_row:
        return None
    return free_window(user_row["created_at"])


def free_window(anchor: datetime) -> PlanWindow:
    """The current Free-tier period, rolling forward from `anchor` in fixed
    SUBSCRIPTION_PERIOD_DAYS steps."""
    now = datetime.now(timezone.utc)
    period_length = timedelta(days=SUBSCRIPTION_PERIOD_DAYS)
    periods_elapsed = max(0, (now - anchor) // period_length)
    period_start = anchor + periods_elapsed * period_length
    return PlanWindow(
        plan=PLAN_QUOTAS["free"],
        payment_id=None,
        period_start=period_start,
        period_end=period_start + period_length,
        status="active",
        cancel_at_period_end=False,
    )


def get_enforced_window(conn, admin_user_id: str) -> Optional[PlanWindow]:
    """The window token caps are actually enforced against (MS-402). Same
    as get_latest_plan_window, except an EXPIRED paid plan drops back to
    the Free quota -- rolling from the day it expired, so usage restarts
    fresh -- instead of switching every cap off until the admin renews.
    Member allocations keep applying on top of that Free pool.
    get_latest_plan_window itself stays as-is so Billing can still show
    the paid plan as "expired" and cancel/resume keep their own rules."""
    window = get_latest_plan_window(conn, admin_user_id)
    if window is None or window.status == "active":
        return window
    return free_window(window.period_end)


# Plans that include Compliance Gap Check — every paid plan. A run costs
# 130k-260k tokens (measured), more than the whole Free pool (60k), so Free
# is excluded. Enterprise is sales-assisted and has no plan_id in
# PLAN_QUOTAS yet; add it here once it does.
GAP_CHECK_PLAN_IDS = frozenset({"individual", "team"})


def plan_id_of(window: "PlanWindow") -> Optional[str]:
    return next((plan_id for plan_id, plan in PLAN_QUOTAS.items() if plan is window.plan), None)


def plan_includes_gap_check(window: Optional["PlanWindow"]) -> bool:
    """The one rule both enforce_gap_check_plan (backend gate) and
    get_my_usage's gap_check_available (chat-ui visibility) read."""
    return window is not None and plan_id_of(window) in GAP_CHECK_PLAN_IDS


def resolve_storage_limits(user: UserRecord) -> tuple[storage_limits.StorageLimits, str]:
    """The upload/storage limits that apply to `user`, and the id of the
    workspace (its admin) they are measured against. Limits follow the
    workspace's enforced plan, so an expired paid plan drops to Free like the
    token caps do. Falls back to the configured defaults, measured against the
    user's own id, if the metering database cannot be reached."""
    conn = db.get_conn("payment")
    if not conn:
        return storage_limits.default_limits(), user.user_id
    try:
        admin_user_id = resolve_admin_user_id(conn, user)
        window = get_enforced_window(conn, admin_user_id)
    finally:
        conn.close()
    plan = window.plan if window else PLAN_QUOTAS["free"]
    return (
        storage_limits.StorageLimits(
            plan_name=plan["name"],
            storage_limit_bytes=plan["storage_limit_bytes"],
            max_file_bytes=plan["max_file_bytes"],
            max_batch_files=plan["max_batch_files"],
        ),
        admin_user_id,
    )


def get_storage_usage(user: UserRecord):
    limits, workspace_id = resolve_storage_limits(user)
    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        used = storage_limits.get_used_bytes(conn, workspace_id)
    finally:
        conn.close()

    remaining = max(0, limits.storage_limit_bytes - used)
    percent = (
        min(100.0, used / limits.storage_limit_bytes * 100)
        if limits.storage_limit_bytes > 0 else 100.0
    )
    return StorageUsageResponse(
        plan_name=limits.plan_name,
        used_bytes=used,
        limit_bytes=limits.storage_limit_bytes,
        remaining_bytes=remaining,
        usage_percent=round(percent, 1),
        max_file_bytes=limits.max_file_bytes,
        max_batch_files=limits.max_batch_files,
        blocked=remaining <= 0,
    )
