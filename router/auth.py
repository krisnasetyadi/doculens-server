# router/auth.py
"""
Authentication & RBAC endpoints. Each route reads the request and calls
services/auth.py, which also provides get_current_user / require_role.

Endpoints:
  POST /auth/register   — create account (role defaults to "user")
  POST /auth/login      — returns JWT access token
  GET  /auth/me         — returns current user's full profile, refreshed from the DB
  POST /auth/me         — update own display name and/or avatar
  GET    /auth/admin/users           — admin-only: list team members this admin created
  POST   /auth/admin/users           — admin-only: add a team member (role "user"), capped by max_sub_users;
                                       also gives them a token allocation (MS-402)
  POST   /auth/admin/users/activate  — admin-only: activate/deactivate a team member this admin created
  PUT    /auth/admin/users/{id}      — admin-only: rename a team member this admin created
  DELETE /auth/admin/users/{id}      — admin-only: remove a team member this admin created and hard-delete their chat history
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from models import (
    AdminCreateUserRequest,
    AdminResetRequest,
    AdminUpdateUserRequest,
    AdminUserStatusRequest,
    ChangePasswordRequest,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    TeamMember,
    TeamMembersResponse,
    TokenResponse,
    UpdateProfileRequest,
    UserActiveResponse,
    UserDeletedResponse,
    UserRecord,
)
from services import auth
from services.auth import get_current_user, require_role

router = APIRouter()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/auth/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest):
    """Register a new user. First registered user becomes admin automatically."""
    return auth.register(body)


@router.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request):
    """Login and receive a JWT access token."""
    return auth.login(body, request)


@router.get("/auth/me", response_model=UserRecord)
async def me(user: UserRecord = Depends(get_current_user)):
    """Return the current user's full profile, refreshed from the DB — the
    JWT alone doesn't carry avatar_url and its is_active claim can go stale."""
    return auth.me(user)


@router.post("/auth/me", response_model=UserRecord)
async def update_profile(
    body: UpdateProfileRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Update the current user's display name and/or avatar. Pass
    remove_avatar=true to clear the avatar back to the initials fallback."""
    return auth.update_profile(body, user)


@router.post("/auth/change-password", status_code=200, response_model=MessageResponse)
async def change_password(
    body: ChangePasswordRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Change the current user's password (requires valid token)."""
    return auth.change_password(body, user)


@router.post("/auth/admin/reset-password", status_code=200, response_model=MessageResponse)
async def admin_reset_password(
    body: AdminResetRequest,
    admin: UserRecord = Depends(require_role("admin")),
):
    """Admin-only: reset a team member's password. Scoped to created_by =
    admin.user_id — an admin can only reset passwords for members they
    created themselves, not any arbitrary user in the system."""
    return auth.admin_reset_password(body, admin)


@router.get("/auth/admin/users", response_model=TeamMembersResponse)
async def list_admin_users(admin: UserRecord = Depends(require_role("admin"))):
    """Admin-only: list the team members this admin has added, plus their quota."""
    return auth.list_admin_users(admin)


@router.post("/auth/admin/users", response_model=TeamMember, status_code=201)
async def add_admin_user(
    body: AdminCreateUserRequest,
    admin: UserRecord = Depends(require_role("admin")),
):
    """Admin-only: add a team member under this admin, capped by max_sub_users. New members always get role 'user'."""
    return auth.add_admin_user(body, admin)


@router.post("/auth/admin/users/activate", response_model=UserActiveResponse)
async def set_admin_user_status(
    body: AdminUserStatusRequest,
    admin: UserRecord = Depends(require_role("admin")),
):
    """Admin-only: activate/deactivate a team member this admin created.
    Scoped to created_by = admin.user_id, so an admin can never touch a
    member outside their own team (and can never deactivate themselves,
    since their own row has no created_by pointing at themselves)."""
    return auth.set_admin_user_status(body, admin)


@router.put("/auth/admin/users/{user_id}", response_model=TeamMember)
async def update_admin_user(
    user_id: str,
    body: AdminUpdateUserRequest,
    admin: UserRecord = Depends(require_role("admin")),
):
    """Admin-only: rename a team member this admin created. Scoped to
    created_by = admin.user_id, same guard as the other per-member admin
    mutations."""
    return auth.update_admin_user(user_id, body, admin)


@router.delete("/auth/admin/users/{user_id}", response_model=UserDeletedResponse)
async def delete_admin_user(
    user_id: str,
    admin: UserRecord = Depends(require_role("admin")),
):
    """Admin-only: remove a team member this admin created. Scoped to
    created_by = admin.user_id, same guard as the other per-member admin
    mutations (and an admin can never remove themselves this way, since
    their own row has no created_by pointing at themselves).

    Hard-deletes the member's chat history (chat_sessions cascades to
    chat_messages via ON DELETE CASCADE) along with their account.
    Deliberately scoped to just that: their uploaded sources (documents),
    skills, and gap-analysis runs are left untouched —
    what happens to those on removal is a separate, not-yet-scoped piece
    of work, not part of this ticket."""
    return auth.delete_admin_user(user_id, admin)
