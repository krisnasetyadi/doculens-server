# billing/ledger.py
"""The token_usage ledger: logging consumption, summing it, and the flat per-user rate limit."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException

import app_db
import db
from config import config
from models import EfficientModeStats, RateLimitStatus
from router.auth import UserRecord

logger = logging.getLogger(__name__)


def sum_tokens_for_admin(conn, admin_user_id: str, period_start, period_end) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(SUM(tokens), 0) AS used FROM token_usage
            WHERE admin_user_id = %s AND created_at >= %s AND created_at < %s
            """,
            (admin_user_id, period_start, period_end),
        )
        return int(cur.fetchone()["used"] or 0)


def sum_tokens_for_user(conn, user_id: str, period_start, period_end) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(SUM(tokens), 0) AS used FROM token_usage
            WHERE user_id = %s AND created_at >= %s AND created_at < %s
            """,
            (user_id, period_start, period_end),
        )
        return int(cur.fetchone()["used"] or 0)


def sum_tokens_by_user(conn, user_ids: list, period_start, period_end) -> dict:
    """Same as sum_tokens_for_user but for a whole team in one query — used
    by get_members_usage so an admin's Billing tab doesn't issue one
    round-trip per member (N+1)."""
    if not user_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT user_id, COALESCE(SUM(tokens), 0) AS used FROM token_usage
            WHERE user_id = ANY(%s) AND created_at >= %s AND created_at < %s
            GROUP BY user_id
            """,
            (user_ids, period_start, period_end),
        )
        rows = cur.fetchall()
    return {r["user_id"]: int(r["used"] or 0) for r in rows}


def get_rate_limit_status(conn, user_id: str) -> RateLimitStatus:
    """Flat, plan-independent safety-net rate limit — sliding window over
    just THIS user's own token_usage rows in the last
    config.rate_limit_window_hours (not the workspace-wide allocation pool
    used by build_subscription_usage). No fixed reset clock: usage simply
    ages out of the window over time, so `reset_at` below is the earliest
    moment that happens naturally, not a scheduled job."""
    window_hours = config.effective_rate_limit_window_hours
    cap = config.rate_limit_token_cap
    window_start = datetime.now(timezone.utc) - timedelta(hours=window_hours)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tokens, created_at FROM token_usage
            WHERE user_id = %s AND created_at >= %s
            ORDER BY created_at ASC
            """,
            (user_id, window_start),
        )
        rows = cur.fetchall()

    used = sum(r["tokens"] for r in rows)
    blocked = cap > 0 and used >= cap
    reset_at = None
    if blocked:
        # Drop rows oldest-first until the remaining sum clears the cap —
        # the row that tips it under is the one whose own age-out moment
        # (its timestamp + the window) is when the user can send again.
        running = used
        for row in rows:
            running -= row["tokens"]
            if running < cap:
                reset_at = row["created_at"] + timedelta(hours=window_hours)
                break

    return RateLimitStatus(
        used_tokens=used,
        cap_tokens=cap,
        window_hours=window_hours,
        blocked=blocked,
        reset_at=app_db.ts(reset_at) if reset_at else None,
    )


def enforce_rate_limit(user_id: str, pending_tokens: int = 0) -> None:
    """Raise HTTPException(429) if this user has hit the flat safety-net
    rate limit. Call this from router/agnostic.py BEFORE the LLM is
    invoked — unlike log_token_usage (best-effort, never raises), this one
    is meant to actually block overage, so callers should let it propagate
    rather than swallowing it. Fails open (never blocks) if the metering DB
    itself is unreachable — a metering outage shouldn't take down chat.
    `pending_tokens`: this user's in-flight reservations (in_flight_tokens)."""
    conn = db.get_conn("payment")
    if not conn:
        return
    try:
        status = get_rate_limit_status(conn, user_id)
    finally:
        conn.close()
    cap = status.cap_tokens
    if status.blocked or (cap > 0 and pending_tokens and status.used_tokens + pending_tokens >= cap):
        raise HTTPException(
            status_code=429,
            detail="Batas token untuk akun kamu sudah tercapai untuk saat ini. Coba lagi setelah beberapa saat.",
        )


def log_token_usage(
    user_id: str,
    tokens: int,
    efficient_mode: bool = False,
    raw_tokens_est: Optional[int] = None,
    final_tokens_est: Optional[int] = None,
) -> None:
    """Best-effort: append one row to the consumption ledger for a query
    that just ran. Called from router/agnostic.py right after a live LLM
    answer is generated. Never raises — a metering hiccup must not break a
    chat response that already succeeded; callers should still wrap this in
    their own try/except as a second line of defense.

    The efficient_mode/*_tokens_est args are MS-247 additions — optional,
    default to the pre-MS-247 no-op values, purely additive to this row.

    Local/free models (e.g. HuggingFace flan-t5) always report tokens=0
    (no API cost to bill), which used to skip this row entirely — but
    that also silently skipped the new efficient_mode/*_tokens_est
    columns, so a user testing Efficient Mode on a local model got a
    correct live popup with nothing ever landing in the aggregate
    dashboard. `efficient_mode=True` keeps the row-worth-writing check
    alive even at tokens=0, purely to preserve that reporting; it still
    inserts tokens=0 (no invented cost)."""
    if tokens <= 0 and not efficient_mode:
        return
    conn = db.get_conn("payment")
    if not conn:
        logger.warning("payment: log_token_usage skipped, no DB connection")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT role, created_by FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
        admin_user_id = user_id
        if row and row["role"] != "admin":
            admin_user_id = row["created_by"] or user_id
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO token_usage
                    (user_id, admin_user_id, tokens, efficient_mode, raw_tokens_est, final_tokens_est)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, admin_user_id, tokens, efficient_mode, raw_tokens_est, final_tokens_est),
            )
    except Exception as exc:
        logger.warning("payment: log_token_usage failed for user %s: %s", user_id, exc)
    finally:
        conn.close()


def get_my_rate_limit(user: UserRecord):
    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        return get_rate_limit_status(conn, user.user_id)
    finally:
        conn.close()


def get_my_efficient_mode_stats(user: UserRecord):
    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        # Inside the try (unlike some sibling handlers in this file) so a
        # DDL failure in here — permissions, a concurrent migration, a
        # lock timeout — still reaches the finally below instead of
        # leaking this connection.
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS queries_tested,
                    COALESCE(SUM(raw_tokens_est), 0) AS total_raw,
                    COALESCE(SUM(final_tokens_est), 0) AS total_final
                FROM token_usage
                WHERE user_id = %s AND efficient_mode = true AND raw_tokens_est IS NOT NULL
                """,
                (user.user_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    queries_tested = int(row["queries_tested"] or 0)
    total_raw = int(row["total_raw"] or 0)
    total_final = int(row["total_final"] or 0)
    return EfficientModeStats(
        queries_tested=queries_tested,
        avg_reduction_pct=(
            round(100 * (1 - total_final / total_raw), 1) if total_raw > 0 else 0.0
        ),
        total_raw_tokens_est=total_raw,
        total_final_tokens_est=total_final,
        total_tokens_saved_est=max(0, total_raw - total_final),
    )
