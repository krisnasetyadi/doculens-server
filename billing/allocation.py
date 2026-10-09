# billing/allocation.py
"""Per-member token allocations carved out of the workspace pool (MS-248, MS-402)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

import app_db
import db
from billing.ledger import sum_tokens_by_user, sum_tokens_for_admin, sum_tokens_for_user
from billing.plans import (
    PlanWindow, get_enforced_window, get_latest_plan_window, plan_includes_gap_check, resolve_admin_user_id,
)
from billing.quotas import (
    build_quota_tiers, default_quota_limits, get_quota_configs, quota_windows_for_user, sum_tokens_for_windows,
)
from config import config
from models import (
    MembersUsageResponse,
    MemberTokenUsage,
    MyMemberUsageResponse,
    SubscriptionUsage,
    TokenQuotaTierUsage,
    UpdateMemberAllocationRequest,
    UpdateMemberAllocationResponse,
    UpdateWorkspaceTokenSettingsRequest,
    WorkspaceTokenSettings,
)
from router.auth import UserRecord

logger = logging.getLogger(__name__)


# ===================== DEFAULT / EFFECTIVE ALLOCATION (MS-402) =====================
# A team member with no token_allocations row used to be silently uncapped
# (enforce_member_allocation returned early) while the Billing tab showed
# them as "0" — so they could burn the whole workspace pool. Now every
# team member (an account an admin created) is capped: by their explicit
# row if there is one, otherwise by the workspace's Default Token
# Allocation. The admin keeps the old opt-in behavior (uncapped unless they
# allocate themselves a slice), since they control the pool anyway; so do
# self-registered accounts, which have no admin to raise a cap.

def exceeds_cap(used: int, cap: int, reserve: int) -> bool:
    """True if a new query must be refused: the cap is already reached, or
    what's left can't fit `reserve` more tokens (see
    config.query_token_reserve for why the headroom matters). The reserve
    is limited to half the cap, so a small cap (e.g. 1,000 with a 2,000
    reserve) still allows queries instead of blocking before the first one."""
    reserve = min(max(0, reserve), cap // 2)
    return used >= cap or used + reserve > cap


def clamp_to_pool(requested: int, token_limit: int, allocated_elsewhere: int) -> tuple[int, bool]:
    """(allocation actually granted, whether it had to be reduced) so a new
    member's allocation never pushes the workspace past its token pool."""
    available = max(0, token_limit - allocated_elsewhere)
    if requested <= available:
        return requested, False
    return available, True


def pool_state(
    conn, admin_user_id: str, window: PlanWindow, allocations: dict[str, tuple[int, bool]],
    extra_user_ids: tuple[str, ...] = (),
) -> tuple[int, dict[str, int], dict[str, int]]:
    """(tokens the plan has left this period, each member's still-unspent
    share of their cap, each member's usage). What can be handed out is the
    plan's remaining tokens minus what members' caps already set aside, so
    the Billing pool always agrees with the Plan tab's remaining figure."""
    used_total = sum_tokens_for_admin(conn, admin_user_id, window.period_start, window.period_end)
    remaining = max(0, window.plan["token_limit"] - used_total)
    used_by_user = sum_tokens_by_user(
        conn, list({*allocations, *extra_user_ids}), window.period_start, window.period_end
    )
    reserved = {
        user_id: max(0, tokens - used_by_user.get(user_id, 0))
        for user_id, (tokens, _) in allocations.items()
    }
    return remaining, reserved, used_by_user


def effective_allocations(
    explicit: dict[str, int], member_ids: list[str], default_allocation: int
) -> dict[str, tuple[int, bool]]:
    """user_id -> (enforced allocation, is_default). Explicit rows win
    (including the admin's own optional one); every listed member without
    a row falls back to the workspace default."""
    result = {uid: (tokens, False) for uid, tokens in explicit.items()}
    for uid in member_ids:
        result.setdefault(uid, (default_allocation, True))
    return result


def get_default_allocation(conn, admin_user_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT default_member_allocation FROM workspace_settings WHERE admin_user_id = %s",
            (admin_user_id,),
        )
        row = cur.fetchone()
    return row["default_member_allocation"] if row else config.default_member_token_allocation


def get_workspace_allocations(conn, admin_user_id: str) -> dict[str, tuple[int, bool]]:
    """Effective allocation for everyone in the workspace pool. Only ACTIVE
    members fall back to the default — a deactivated member can't query,
    so counting a default for them would just shrink the pool for nothing."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, allocated_tokens FROM token_allocations WHERE admin_user_id = %s",
            (admin_user_id,),
        )
        explicit = {r["user_id"]: r["allocated_tokens"] for r in cur.fetchall()}
        cur.execute(
            "SELECT user_id FROM users WHERE created_by = %s AND is_active = true",
            (admin_user_id,),
        )
        member_ids = [r["user_id"] for r in cur.fetchall()]
    return effective_allocations(explicit, member_ids, get_default_allocation(conn, admin_user_id))


def get_user_allocation(conn, user: UserRecord) -> Optional[tuple[int, bool]]:
    """(enforced allocation, is_default) for one user, or None if they're
    uncapped: an admin with no explicit row, or a self-registered account
    (created_by NULL). The latter owns its own workspace with no admin to
    raise a cap, so only its plan limit applies, same as before MS-402."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT allocated_tokens FROM token_allocations WHERE user_id = %s",
            (user.user_id,),
        )
        row = cur.fetchone()
    if row:
        return row["allocated_tokens"], False
    if user.role == "admin":
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT created_by FROM users WHERE user_id = %s", (user.user_id,))
        user_row = cur.fetchone()
    if not user_row or not user_row["created_by"]:
        return None
    return get_default_allocation(conn, user_row["created_by"]), True


def member_usage(
    user_id: str, email: str, allocated: int, used: int, is_default: bool,
    quota_anchor_at: Optional[datetime] = None,
    quota_tiers: Optional[list[TokenQuotaTierUsage]] = None,
) -> MemberTokenUsage:
    return MemberTokenUsage(
        user_id=user_id,
        email=email,
        allocated_tokens=allocated,
        used_tokens=used,
        remaining_tokens=max(0, allocated - used),
        usage_percent=round(used / allocated * 100, 2) if allocated > 0 else 0.0,
        is_default_allocation=is_default,
        quota_anchor_at=app_db.ts(quota_anchor_at) if quota_anchor_at else None,
        quota_tiers=quota_tiers or [],
    )


def assign_initial_allocation(
    admin_user_id: str, user_id: str, requested: Optional[int] = None
) -> Optional[tuple[int, bool]]:
    """Called by router/auth.py right after a team member is created.
    `requested` (a custom monthly cap from the create form, else the
    workspace default) is clamped to what's left of the pool and stored as
    an explicit token_allocations row, together with derived daily/weekly
    limits and a quota anchor of now (MS-418), so the new member is under
    all three tiers from the start. Returns (allocated, clamped),
    or None if the metering DB is unreachable; the member is then still
    capped by the default at enforcement time, so this is best-effort."""
    conn = db.get_conn("payment")
    if not conn:
        return None
    try:
        wanted = requested if requested is not None else get_default_allocation(conn, admin_user_id)
        window = get_enforced_window(conn, admin_user_id)
        clamped = False
        if window:
            allocations = get_workspace_allocations(conn, admin_user_id)
            remaining, reserved, _ = pool_state(conn, admin_user_id, window, allocations)
            elsewhere = sum(tokens for uid, tokens in reserved.items() if uid != user_id)
            wanted, clamped = clamp_to_pool(wanted, remaining, elsewhere)
        daily, weekly = default_quota_limits(wanted)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO token_allocations
                    (admin_user_id, user_id, allocated_tokens,
                     daily_token_quota, weekly_token_quota, quota_anchor_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    allocated_tokens = EXCLUDED.allocated_tokens,
                    daily_token_quota = EXCLUDED.daily_token_quota,
                    weekly_token_quota = EXCLUDED.weekly_token_quota,
                    quota_anchor_at = EXCLUDED.quota_anchor_at
                """,
                (admin_user_id, user_id, wanted, daily, weekly, datetime.now(timezone.utc)),
            )
        return wanted, clamped
    except Exception as exc:
        logger.warning("payment: assign_initial_allocation failed for user %s: %s", user_id, exc)
        return None
    finally:
        conn.close()


def release_member_allocation(user_id: str) -> None:
    """Called by router/auth.py when a member is deleted, so their slice
    goes back to the pool instead of staying locked by a user that no
    longer exists. Best-effort, never raises."""
    conn = db.get_conn("payment")
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM token_allocations WHERE user_id = %s", (user_id,))
    except Exception as exc:
        logger.warning("payment: release_member_allocation failed for user %s: %s", user_id, exc)
    finally:
        conn.close()


def build_subscription_usage(
    conn, admin_user_id: str, window: Optional[PlanWindow] = None
) -> Optional[SubscriptionUsage]:
    """`window` lets a caller that already fetched one (e.g. to validate
    status before mutating something) pass it through instead of paying for
    a second identical `get_latest_plan_window` query."""
    if window is None:
        window = get_latest_plan_window(conn, admin_user_id)
    if not window:
        return None
    token_limit = window.plan["token_limit"]
    token_used = sum_tokens_for_admin(conn, admin_user_id, window.period_start, window.period_end)
    # No auto-renewal exists regardless (see module docstring), so
    # next_reset_date only means "you can keep using this plan past
    # period_end without lifting a finger" — not true once cancelled.
    renews = window.status == "active" and not window.cancel_at_period_end
    return SubscriptionUsage(
        plan_name=window.plan["name"],
        subscription_status=window.status,
        token_limit=token_limit,
        token_used=token_used,
        token_remaining=max(0, token_limit - token_used),
        period_start=app_db.ts(window.period_start),
        period_end=app_db.ts(window.period_end),
        next_reset_date=app_db.ts(window.period_end) if renews else None,
        cancel_at_period_end=window.cancel_at_period_end,
        is_paid=window.payment_id is not None,
    )


def get_usage_snapshot(user: UserRecord) -> Optional[dict]:
    """This user's plan + token usage for the current enforced period, as a
    plain dict for the chat assistant's conversation mode ("sisa token saya
    berapa?"). Same numbers as /payments/subscription/me (own allocation)
    plus, for admins only, the workspace pool from Billing — a member never
    sees workspace-wide totals here, same as in the Usage tab. Best-effort:
    returns None on any failure, the assistant then points to /usage."""
    conn = db.get_conn("payment")
    if not conn:
        return None
    try:
        admin_user_id = resolve_admin_user_id(conn, user)
        window = get_enforced_window(conn, admin_user_id)
        if not window:
            return None
        allocation = get_user_allocation(conn, user)
        snapshot = {
            "plan_name": window.plan["name"],
            "period_end": app_db.ts(window.period_end),
            "my_token_used": sum_tokens_for_user(conn, user.user_id, window.period_start, window.period_end),
            "my_token_limit": allocation[0] if allocation else None,
        }
        if user.role == "admin":
            snapshot["workspace_token_limit"] = window.plan["token_limit"]
            snapshot["workspace_token_used"] = sum_tokens_for_admin(
                conn, admin_user_id, window.period_start, window.period_end
            )
        return snapshot
    except Exception as exc:
        logger.warning("payment: get_usage_snapshot failed for user %s: %s", user.user_id, exc)
        return None
    finally:
        conn.close()


def get_my_usage(user: UserRecord):
    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        admin_user_id = resolve_admin_user_id(conn, user)
        window = get_enforced_window(conn, admin_user_id)
        if not window:
            return MyMemberUsageResponse(usage=None)

        # Uncapped admin (no row) keeps reporting 0, as before MS-402.
        allocated, is_default = get_user_allocation(conn, user) or (0, False)
        used = sum_tokens_for_user(conn, user.user_id, window.period_start, window.period_end)
        quota_row = get_quota_configs(conn, [user.user_id]).get(user.user_id)
        quota_tiers = []
        if quota_row:
            now = datetime.now(timezone.utc)
            windows = quota_windows_for_user(user.user_id, quota_row["quota_anchor_at"], now)
            quota_tiers = build_quota_tiers(
                user.user_id, allocated, quota_row, sum_tokens_for_windows(conn, windows), now
            )
            used = quota_tiers[-1].token_used
    finally:
        conn.close()

    usage = member_usage(
        user.user_id, user.email, allocated, used, is_default,
        quota_row["quota_anchor_at"] if quota_row else None, quota_tiers,
    )
    return MyMemberUsageResponse(usage=usage, gap_check_available=plan_includes_gap_check(window))


def get_members_usage(admin: UserRecord):
    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        window = get_latest_plan_window(conn, admin.user_id)
        subscription = build_subscription_usage(conn, admin.user_id, window=window)
        # Member usage and the allocation pool follow the ENFORCED window --
        # the Free quota once a paid plan has expired (MS-402) -- while
        # `subscription` above keeps describing the paid plan itself.
        pool = get_enforced_window(conn, admin.user_id)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, email FROM users WHERE created_by = %s ORDER BY created_at DESC",
                (admin.user_id,),
            )
            team_rows = cur.fetchall()
        pool_rows = [{"user_id": admin.user_id, "email": admin.email}] + list(team_rows)

        # Effective, not just explicit rows (MS-402): a member without a row
        # is shown — and counted against the pool — at the default they're
        # actually enforced at, instead of a misleading 0.
        allocations = get_workspace_allocations(conn, admin.user_id)

        members: list[MemberTokenUsage] = []
        if pool:
            user_ids = [row["user_id"] for row in pool_rows]
            used_by_user = sum_tokens_by_user(
                conn, user_ids, pool.period_start, pool.period_end
            )
            quota_configs = get_quota_configs(conn, user_ids)
            now = datetime.now(timezone.utc)
            quota_windows = [
                window
                for user_id, quota in quota_configs.items()
                for window in quota_windows_for_user(user_id, quota["quota_anchor_at"], now)
            ]
            quota_used = sum_tokens_for_windows(conn, quota_windows)
            for row in pool_rows:
                allocated, is_default = allocations.get(row["user_id"], (0, False))
                used = used_by_user.get(row["user_id"], 0)
                quota = quota_configs.get(row["user_id"])
                quota_tiers = build_quota_tiers(row["user_id"], allocated, quota, quota_used, now) if quota else []
                if quota_tiers:
                    used = quota_tiers[-1].token_used
                members.append(member_usage(
                    row["user_id"], row["email"], allocated, used, is_default,
                    quota["quota_anchor_at"] if quota else None, quota_tiers,
                ))

        token_limit = pool.plan["token_limit"] if pool else 0
        unallocated = 0
        if pool:
            remaining, reserved, _ = pool_state(conn, admin.user_id, pool, allocations)
            unallocated = max(0, remaining - sum(reserved.values()))
    finally:
        conn.close()

    return MembersUsageResponse(
        subscription=subscription,
        members=members,
        unallocated_tokens=unallocated,
        pool_token_limit=token_limit,
        pool_plan_name=pool.plan["name"] if pool else None,
    )


def set_member_allocation(body: UpdateMemberAllocationRequest, admin: UserRecord):
    configuring_quotas = body.daily_token_quota is not None or body.weekly_token_quota is not None
    if configuring_quotas and (body.daily_token_quota is None or body.weekly_token_quota is None):
        raise HTTPException(status_code=400, detail="Daily and weekly quotas must be configured together")

    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        if body.user_id == admin.user_id:
            member_row = {"user_id": admin.user_id, "email": admin.email}
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id, email FROM users WHERE user_id = %s AND created_by = %s",
                    (body.user_id, admin.user_id),
                )
                member_row = cur.fetchone()
            if not member_row:
                raise HTTPException(status_code=404, detail="Team member not found")

        window = get_enforced_window(conn, admin.user_id)
        if not window:
            raise HTTPException(status_code=400, detail="No active subscription to allocate tokens from")
        allocations = get_workspace_allocations(conn, admin.user_id)
        remaining, reserved, used_by_user = pool_state(
            conn, admin.user_id, window, allocations, (body.user_id,)
        )
        reserved_elsewhere = sum(tokens for uid, tokens in reserved.items() if uid != body.user_id)
        new_reserved = max(0, body.allocated_tokens - used_by_user.get(body.user_id, 0))
        current = allocations.get(body.user_id, (0, False))[0]
        # Lowering is always allowed (MS-402): members that fell back to the
        # default can leave an older workspace over-allocated, and refusing
        # every edit there would leave the admin no way to fix it.
        is_increase = body.allocated_tokens > current
        if is_increase and reserved_elsewhere + new_reserved > remaining:
            raise HTTPException(status_code=400, detail="Allocation exceeds the tokens left in the workspace's plan")

        if configuring_quotas:
            daily_limit, weekly_limit = body.daily_token_quota, body.weekly_token_quota
        else:
            existing = get_quota_configs(conn, [body.user_id]).get(body.user_id)
            daily_limit = existing["daily_token_quota"] if existing else None
            weekly_limit = existing["weekly_token_quota"] if existing else None
        if daily_limit is not None and weekly_limit is not None:
            if daily_limit > weekly_limit:
                raise HTTPException(status_code=400, detail="Daily limit cannot exceed the weekly limit")
            if weekly_limit > body.allocated_tokens:
                raise HTTPException(status_code=400, detail="Weekly limit cannot exceed the monthly limit")

        with conn.cursor() as cur:
            if configuring_quotas:
                cur.execute(
                    """
                    INSERT INTO token_allocations
                        (admin_user_id, user_id, allocated_tokens,
                         daily_token_quota, weekly_token_quota, quota_anchor_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        allocated_tokens = EXCLUDED.allocated_tokens,
                        daily_token_quota = EXCLUDED.daily_token_quota,
                        weekly_token_quota = EXCLUDED.weekly_token_quota,
                        quota_anchor_at = COALESCE(token_allocations.quota_anchor_at, EXCLUDED.quota_anchor_at)
                    """,
                    (admin.user_id, body.user_id, body.allocated_tokens,
                     body.daily_token_quota, body.weekly_token_quota, datetime.now(timezone.utc)),
                )
            else:
                # Existing allocation-only clients keep their old behavior;
                # editing a monthly cap later never shifts a quota anchor.
                cur.execute(
                    """
                    INSERT INTO token_allocations (admin_user_id, user_id, allocated_tokens)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET allocated_tokens = EXCLUDED.allocated_tokens
                    """,
                    (admin.user_id, body.user_id, body.allocated_tokens),
                )

        quota_row = get_quota_configs(conn, [body.user_id]).get(body.user_id)
        quota_tiers = []
        if quota_row:
            now = datetime.now(timezone.utc)
            windows = quota_windows_for_user(body.user_id, quota_row["quota_anchor_at"], now)
            quota_tiers = build_quota_tiers(
                body.user_id, body.allocated_tokens, quota_row, sum_tokens_for_windows(conn, windows), now
            )
            used = quota_tiers[-1].token_used
        else:
            used = sum_tokens_for_user(conn, body.user_id, window.period_start, window.period_end)
        unallocated = max(0, remaining - reserved_elsewhere - new_reserved)
    finally:
        conn.close()

    member = member_usage(
        body.user_id, member_row["email"], body.allocated_tokens, used, False,
        quota_row["quota_anchor_at"] if quota_row else None, quota_tiers,
    )
    return UpdateMemberAllocationResponse(member=member, unallocated_tokens=unallocated)


def get_workspace_token_settings(admin: UserRecord):
    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        return WorkspaceTokenSettings(default_member_allocation=get_default_allocation(conn, admin.user_id))
    finally:
        conn.close()


def update_workspace_token_settings(body: UpdateWorkspaceTokenSettingsRequest, admin: UserRecord):
    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        window = get_enforced_window(conn, admin.user_id)
        if window and body.default_member_allocation > window.plan["token_limit"]:
            raise HTTPException(
                status_code=400,
                detail="Default allocation exceeds the workspace's token pool",
            )
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO workspace_settings (admin_user_id, default_member_allocation)
                VALUES (%s, %s)
                ON CONFLICT (admin_user_id) DO UPDATE
                    SET default_member_allocation = EXCLUDED.default_member_allocation,
                        updated_at = now()
                """,
                (admin.user_id, body.default_member_allocation),
            )
    finally:
        conn.close()
    return WorkspaceTokenSettings(default_member_allocation=body.default_member_allocation)
