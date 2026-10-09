# router/payment.py
"""
Routes for plans, token usage and payments under /payments. Each route reads
the request and calls billing/ (see billing/__init__.py), where the rules,
SQL and the Stripe calls live.

Payments are a dummy/test-mode Stripe Checkout flow (MS-90): one-time
payments, prices defined server-side in billing/plans.py and never trusted
from the client.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from billing import allocation, ledger, payments, plans, token_requests
from router.auth import get_current_user, require_role, UserRecord
from models import (
    CreateCheckoutSessionRequest,
    CheckoutSessionResponse,
    PaymentResponse,
    SubscriptionUsage,
    MyMemberUsageResponse,
    MembersUsageResponse,
    UpdateMemberAllocationRequest,
    UpdateMemberAllocationResponse,
    WorkspaceTokenSettings,
    UpdateWorkspaceTokenSettingsRequest,
    RateLimitStatus,
    StorageUsageResponse,
    EfficientModeStats,
    CreateTokenRequestRequest,
    TokenRequestResponse,
    TokenRequestsResponse,
)

router = APIRouter()

# The checkout flow (MS-90) has no login step in its path — Pricing → Select
# Plan → Payment → Gateway → Result — so it must work for a guest. Auth is
# accepted (attributes the payment to a real user_id) but never required:
# reuses get_current_user's own JWT validation instead of duplicating it,
# just swallows the 401 for a missing/invalid token instead of raising.
_optional_bearer = HTTPBearer(auto_error=False)


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_optional_bearer),
) -> Optional[UserRecord]:
    if not credentials:
        return None
    try:
        return get_current_user(credentials)
    except HTTPException:
        return None


@router.post(
    "/payments/checkout-session",
    response_model=CheckoutSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_checkout_session(
    body: CreateCheckoutSessionRequest,
    user: Optional[UserRecord] = Depends(get_optional_user),
):
    return payments.create_checkout_session(body, user)


@router.post("/payments/webhook")
async def stripe_webhook(request: Request):
    """No auth — Stripe calls this directly. First raw-body endpoint in this
    codebase: signature verification needs the exact raw bytes, not a parsed
    Pydantic model."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    return payments.handle_webhook(payload, sig_header)


@router.get("/payments/session/{session_id}", response_model=PaymentResponse)
async def get_payment_by_session(session_id: str):
    """No auth required — same guest-friendly model as checkout-session.
    Authorization here is possession of the Stripe-generated session_id
    itself (only known to the browser Stripe just redirected back), the
    same trust model as a typical guest order-confirmation link."""
    return payments.get_payment_by_session(session_id)


# ===================== TOKEN USAGE & ALLOCATION (MS-248) =====================


@router.get("/payments/subscription/me", response_model=MyMemberUsageResponse)
async def get_my_usage(user: UserRecord = Depends(get_current_user)):
    """Any authenticated user — their own allocation/consumption within
    their workspace's current subscription period, for the Usage tab."""
    return allocation.get_my_usage(user)


@router.get("/payments/subscription/members", response_model=MembersUsageResponse)
async def get_members_usage(admin: UserRecord = Depends(require_role("admin"))):
    """Admin-only — subscription overview plus every team member's own
    allocation/consumption, for the Billing tab's allocation editor. The
    admin is included as the first row (MS-248 follow-up) — they're a
    participant in the same shared pool as their team, not a special case,
    so they can optionally cap their own usage for budget discipline and
    raise it back up themselves whenever they want."""
    return allocation.get_members_usage(admin)


@router.post("/payments/subscription/cancel", response_model=SubscriptionUsage)
async def cancel_subscription(admin: UserRecord = Depends(require_role("admin"))):
    """Admin-only — cancel-at-period-end (not immediate): the admin already
    paid for the current period, so access/token_limit are untouched until
    period_end. This just stops next_reset_date implying it'll keep going
    past that — there's no auto-renewal to actually cancel (see module
    docstring), so all this does is flag the period as non-renewable."""
    return payments.cancel_subscription(admin)


@router.post("/payments/subscription/resume", response_model=SubscriptionUsage)
async def resume_subscription(admin: UserRecord = Depends(require_role("admin"))):
    """Admin-only — undo a pending cancellation, as long as the paid period
    hasn't ended yet (matches standard "resume before it lapses" UX)."""
    return payments.resume_subscription(admin)


@router.post("/payments/subscription/allocations", response_model=UpdateMemberAllocationResponse)
async def set_member_allocation(
    body: UpdateMemberAllocationRequest,
    admin: UserRecord = Depends(require_role("admin")),
):
    """Admin-only — set one team member's token cap, carved out of the
    workspace's token_limit. Scoped to created_by = admin.user_id, same
    guard as auth.py's other per-member admin mutations — except the admin
    can also target their own user_id, to allocate themselves a slice of
    the same pool for their own budget discipline (MS-248 follow-up)."""
    return allocation.set_member_allocation(body, admin)


@router.get("/payments/subscription/settings", response_model=WorkspaceTokenSettings)
async def get_workspace_token_settings(admin: UserRecord = Depends(require_role("admin"))):
    """Admin-only (MS-402) — this workspace's Default Token Allocation."""
    return allocation.get_workspace_token_settings(admin)


@router.put("/payments/subscription/settings", response_model=WorkspaceTokenSettings)
async def update_workspace_token_settings(
    body: UpdateWorkspaceTokenSettingsRequest,
    admin: UserRecord = Depends(require_role("admin")),
):
    """Admin-only (MS-402) — set the Default Token Allocation new members
    get, and that members without an explicit allocation are capped at.
    Can't exceed the plan's whole token_limit — a default no single member
    could ever be granted would only ever be clamped."""
    return allocation.update_workspace_token_settings(body, admin)


@router.get("/payments/storage/usage", response_model=StorageUsageResponse)
async def get_storage_usage(user: UserRecord = Depends(get_current_user)):
    """Workspace storage used against the plan's limit (MS-504). Members see
    their workspace's totals, since the quota is shared."""
    return plans.get_storage_usage(user)


@router.get("/payments/rate-limit/me", response_model=RateLimitStatus)
async def get_my_rate_limit(user: UserRecord = Depends(get_current_user)):
    """Any authenticated user — lets the frontend pre-emptively disable the
    chat composer (and show a reset countdown) instead of only finding out
    they're blocked after a query already 429s."""
    return ledger.get_my_rate_limit(user)


@router.get("/payments/efficient-mode/stats", response_model=EfficientModeStats)
async def get_my_efficient_mode_stats(user: UserRecord = Depends(get_current_user)):
    """MS-247 "Efficient Mode" mini-dashboard (Settings > Efficient Mode).
    Scoped to this user's own queries only (not the whole workspace) —
    this is a personal "did toggling it on actually help" comparison, not
    a billing figure, so it doesn't need admin/workspace aggregation like
    the rest of this file's usage endpoints."""
    return ledger.get_my_efficient_mode_stats(user)


@router.post("/payments/subscription/request-more", response_model=TokenRequestResponse)
async def request_more_tokens(
    body: CreateTokenRequestRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Any authenticated user — ask their workspace admin for a bigger
    allocation. In-app only for now (the admin sees it next time they open
    Billing, via polling) — a real push-notification channel is a
    separate, larger follow-up. One pending request at a time per user;
    dismissing an old one (admin-side) frees them up to ask again."""
    return token_requests.request_more_tokens(body, user)


@router.get("/payments/subscription/requests", response_model=TokenRequestsResponse)
async def list_token_requests(admin: UserRecord = Depends(require_role("admin"))):
    """Admin-only — pending (and recently resolved) token requests from
    their team, for the Billing tab and the sidebar's pending-count badge."""
    return token_requests.list_token_requests(admin)


@router.post("/payments/subscription/requests/{request_id}/dismiss", response_model=TokenRequestResponse)
async def dismiss_token_request(request_id: str, admin: UserRecord = Depends(require_role("admin"))):
    """Admin-only — mark a request as handled (whether or not they actually
    raised the member's allocation via the allocation editor elsewhere in
    the same Billing tab) so it stops showing as pending, and frees that
    member up to send a new request later if they need to."""
    return token_requests.dismiss_token_request(request_id, admin)
