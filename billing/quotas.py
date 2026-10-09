# billing/quotas.py
"""Daily / weekly / monthly quota windows for a member's allocation (MS-418)."""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta, timezone

import app_db
from models import TokenQuotaTierUsage


def _month_anniversary(anchor: datetime, offset: int) -> datetime:
    """Always calculate from the original day, so Jan 31 -> Feb 28 -> Mar 31."""
    month_index = anchor.year * 12 + anchor.month - 1 + offset
    year, month_index = divmod(month_index, 12)
    month = month_index + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return anchor.replace(year=year, month=month, day=day)


def quota_period(anchor: datetime, interval: str, now: datetime) -> tuple[datetime, datetime]:
    """Current anchored [start, end) window."""
    anchor = anchor.astimezone(timezone.utc)
    now = now.astimezone(timezone.utc)
    if interval in ("daily", "weekly"):
        length = timedelta(days=1 if interval == "daily" else 7)
        elapsed = max(0, (now - anchor) // length)
        start = anchor + elapsed * length
        return start, start + length

    if interval != "monthly":
        raise ValueError(f"Unknown quota interval: {interval}")
    offset = (now.year - anchor.year) * 12 + now.month - anchor.month
    start = _month_anniversary(anchor, offset)
    if start > now:
        offset -= 1
        start = _month_anniversary(anchor, offset)
    return start, _month_anniversary(anchor, offset + 1)


def default_quota_limits(monthly: int) -> tuple[int, int]:
    """(daily, weekly) for a new member: the 2,000 / 50,000 / 200,000 ratio,
    i.e. 1% and 25% of the monthly cap. Rounded up so a small cap never
    yields a 0 limit (0 blocks), which keeps daily <= weekly <= monthly."""
    return -(-monthly // 100), -(-monthly // 4)


def _quota_windows(anchor: datetime, now: datetime) -> dict[str, tuple[datetime, datetime]]:
    return {interval: quota_period(anchor, interval, now) for interval in ("daily", "weekly", "monthly")}


def sum_tokens_for_windows(
    conn, windows: list[tuple[str, str, datetime, datetime]]
) -> dict[tuple[str, str], int]:
    """One indexed ledger query for all members' active quota windows."""
    if not windows:
        return {}
    placeholders = ", ".join("(%s::text, %s::text, %s::timestamptz, %s::timestamptz)" for _ in windows)
    params = [value for window in windows for value in window]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT w.user_id, w.tier, COALESCE(SUM(u.tokens), 0) AS used
            FROM (VALUES {placeholders}) AS w(user_id, tier, period_start, period_end)
            LEFT JOIN token_usage u ON u.user_id = w.user_id
                AND u.created_at >= w.period_start AND u.created_at < w.period_end
            GROUP BY w.user_id, w.tier
            """,
            params,
        )
        rows = cur.fetchall()
    return {(row["user_id"], row["tier"]): int(row["used"] or 0) for row in rows}


def get_quota_configs(conn, user_ids: list[str]) -> dict[str, dict]:
    if not user_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """SELECT user_id, daily_token_quota, weekly_token_quota, quota_anchor_at
               FROM token_allocations WHERE user_id = ANY(%s) AND quota_anchor_at IS NOT NULL""",
            (user_ids,),
        )
        rows = cur.fetchall()
    return {row["user_id"]: row for row in rows}


def quota_windows_for_user(user_id: str, anchor: datetime, now: datetime) -> list[tuple[str, str, datetime, datetime]]:
    return [(user_id, interval, start, end) for interval, (start, end) in _quota_windows(anchor, now).items()]


def build_quota_tiers(
    user_id: str, allocated: int, config_row: dict, used_by_window: dict, now: datetime
) -> list[TokenQuotaTierUsage]:
    limits = {
        "daily": config_row["daily_token_quota"],
        "weekly": config_row["weekly_token_quota"],
        "monthly": allocated,
    }
    periods = _quota_windows(config_row["quota_anchor_at"], now)
    tiers = []
    for interval, limit in limits.items():
        start, end = periods[interval]
        used = used_by_window.get((user_id, interval), 0)
        tiers.append(TokenQuotaTierUsage(
            interval=interval,
            token_limit=limit,
            token_used=used,
            token_remaining=max(0, limit - used),
            period_start=app_db.ts(start),
            next_reset_date=app_db.ts(end),
            blocked=used >= limit,
        ))
    return tiers
