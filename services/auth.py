# services/auth.py
"""
Authentication and team management (moved out of router/auth.py by MS-657).

Issues and checks JWTs, hashes passwords, rate-limits failed logins, and
provides the dependencies every router uses:
  get_current_user(token)        -> UserRecord (any authenticated user)
  require_role("admin")          -> raises 403 if role doesn't match
  CurrentUser / AdminOnly        -> FastAPI Depends shortcuts

Schema (created at startup by schema/auth.py):
  users (user_id, email, password_hash, role, is_active, name, avatar_url, created_by, max_sub_users, created_at, updated_at)
  name           — display name, editable via POST /auth/me
  avatar_url     — small avatar image as a data: URL, editable via POST /auth/me
  created_by     — user_id of the admin who added this account (NULL for self-registered users)
  max_sub_users  — how many team members this user may add as an admin; set manually per admin for now
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import db
from models import (
    AdminCreateUserRequest,
    AdminResetRequest,
    AdminUpdateUserRequest,
    AdminUserStatusRequest,
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    TeamMember,
    TeamMembersResponse,
    TokenResponse,
    UpdateProfileRequest,
    UserRecord,
)
from services.billing.allocation import assign_initial_allocation, release_member_allocation

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — read from env with sensible defaults
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET environment variable is not set. Refusing to start with a "
        "guessable default secret — set JWT_SECRET before running the server."
    )
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 h

_bearer = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Login rate limiting — in-memory, per (email, IP). Resets on process restart;
# fine for a single-instance deployment, not shared across multiple workers.
# ---------------------------------------------------------------------------

LOGIN_ATTEMPT_LIMIT = 5
LOGIN_ATTEMPT_WINDOW_SECONDS = 15 * 60  # 15 minutes

_failed_login_attempts: dict[str, list[float]] = defaultdict(list)

# Cached hash used to equalize verify() timing when no user row is found,
# so login doesn't leak "this email doesn't exist" via response latency.
_dummy_hash_cache: dict[str, str] = {}


def _rate_limit_key(email: str, request: Request) -> str:
    client_ip = request.client.host if request.client else "unknown"
    return f"{email.strip().lower()}:{client_ip}"


def _check_rate_limit(key: str) -> None:
    now = time.monotonic()
    # .get() instead of indexing the defaultdict directly — a plain read
    # must not auto-vivify a permanent entry for keys that never fail again
    # (e.g. an attacker probing many distinct emails), or this dict grows
    # for the life of the process with no cleanup path.
    attempts = [t for t in _failed_login_attempts.get(key, []) if now - t < LOGIN_ATTEMPT_WINDOW_SECONDS]
    if attempts:
        _failed_login_attempts[key] = attempts
    else:
        _failed_login_attempts.pop(key, None)
    if len(attempts) >= LOGIN_ATTEMPT_LIMIT:
        retry_minutes = LOGIN_ATTEMPT_WINDOW_SECONDS // 60
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Try again in {retry_minutes} minutes.",
        )


def _record_failed_login(key: str) -> None:
    _failed_login_attempts[key].append(time.monotonic())


def _clear_failed_logins(key: str) -> None:
    _failed_login_attempts.pop(key, None)


# ---------------------------------------------------------------------------
# Lazy imports so startup doesn't fail if packages missing
# ---------------------------------------------------------------------------

def _jose():
    from jose import jwt, JWTError  # noqa: F401
    return jwt

def _passlib():
    from passlib.context import CryptContext
    return CryptContext(schemes=["bcrypt"], deprecated="auto")

def _hash_password(pwd_ctx, password: str) -> str:
    """Hash password (length already validated to fit bcrypt's 72-byte limit)."""
    return pwd_ctx.hash(password)

def _dummy_password_hash(pwd_ctx) -> str:
    """A real bcrypt hash of a fixed, unused password — used only so failed
    logins for a non-existent email take the same time as a wrong-password
    login for a real one."""
    if "hash" not in _dummy_hash_cache:
        _dummy_hash_cache["hash"] = pwd_ctx.hash("dummy-password-for-timing-equalization")
    return _dummy_hash_cache["hash"]


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _create_token(user_id: str, email: str, role: str, name: Optional[str] = None) -> str:
    jwt = _jose()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "email": email, "role": role, "exp": expire}
    if name:
        payload["name"] = name
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict:
    from jose import JWTError
    jwt = _jose()
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# ---------------------------------------------------------------------------
# RBAC dependency helpers (use these in other routers)
# ---------------------------------------------------------------------------

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> UserRecord:
    """Require a valid JWT, then re-check the account against the DB on every
    request (role, is_active, existence) instead of trusting the token's
    claims for its lifetime. This is what makes a deleted or deactivated
    account's access end immediately rather than whenever its token expires."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = _decode_token(credentials.credentials)
    conn = db.get_conn("auth")
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, is_active, name FROM users WHERE user_id = %s",
                    (payload["sub"],),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        if not row or not row["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return UserRecord(
            user_id=payload["sub"],
            email=payload["email"],
            role=row["role"],
            is_active=True,
            name=row["name"],
        )
    # DB unavailable — fall back to trusting the token rather than locking
    # everyone out of every route on a transient DB hiccup.
    return UserRecord(
        user_id=payload["sub"],
        email=payload["email"],
        role=payload["role"],
        is_active=True,
    )


def require_role(*roles: str):
    """
    Factory for role-gated dependencies.

    Usage:
        @router.get("/admin-only")
        def admin_route(user = Depends(require_role("admin"))):
            ...
    """
    def _dep(user: UserRecord = Depends(get_current_user)) -> UserRecord:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {', '.join(roles)}",
            )
        return user
    return _dep


# Convenience aliases
CurrentUser = Depends(get_current_user)
AdminOnly   = Depends(require_role("admin"))


def register(body: RegisterRequest):
    pwd_ctx = _passlib()
    conn = db.get_conn("auth")

    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        with conn.cursor() as cur:
            # Check email already exists
            cur.execute("SELECT user_id FROM users WHERE email = %s", (body.email,))
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="Email already registered")

            # First user in DB gets admin role regardless of request
            cur.execute("SELECT COUNT(*) AS cnt FROM users")
            count = cur.fetchone()["cnt"]
            # Role is never caller-controlled: the first account bootstraps
            # the installation, and every later account starts as a regular
            # user — there is no self-service way to become an admin.
            role = "admin" if count == 0 else "user"

            user_id = str(uuid.uuid4())
            hashed  = _hash_password(pwd_ctx, body.password)

            cur.execute("""
                INSERT INTO users (user_id, email, password_hash, role, name)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING user_id, email, role, name
            """, (user_id, body.email, hashed, role, body.name))
            row = cur.fetchone()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("auth register error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Registration failed") from e
    finally:
        conn.close()

    token = _create_token(row["user_id"], row["email"], row["role"], row["name"])
    return TokenResponse(
        access_token=token,
        user_id=row["user_id"],
        email=row["email"],
        role=row["role"],
        name=row["name"],
    )


def login(body: LoginRequest, request: Request):
    rl_key = _rate_limit_key(body.email, request)
    _check_rate_limit(rl_key)

    pwd_ctx = _passlib()
    conn = db.get_conn("auth")

    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, email, password_hash, role, is_active, name FROM users WHERE email = %s",
                (body.email,)
            )
            row = cur.fetchone()
    except Exception as e:
        logger.error("auth login db error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Login failed") from e
    finally:
        conn.close()

    # Always run a bcrypt verify, even for an unknown email, so response time
    # doesn't reveal whether the address is registered.
    password_hash = row["password_hash"] if row else _dummy_password_hash(pwd_ctx)
    password_ok = pwd_ctx.verify(body.password, password_hash)

    if not row or not password_ok:
        _record_failed_login(rl_key)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not row["is_active"]:
        raise HTTPException(status_code=403, detail="Account is disabled")

    _clear_failed_logins(rl_key)
    token = _create_token(row["user_id"], row["email"], row["role"], row["name"])
    return TokenResponse(
        access_token=token,
        user_id=row["user_id"],
        email=row["email"],
        role=row["role"],
        name=row["name"],
    )


def me(user: UserRecord):
    conn = db.get_conn("auth")
    if not conn:
        return user
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, email, role, is_active, name, avatar_url FROM users WHERE user_id = %s",
                (user.user_id,),
            )
            row = cur.fetchone()
        return UserRecord(**row) if row else user
    finally:
        conn.close()


def update_profile(body: UpdateProfileRequest, user: UserRecord):
    if body.name is None and body.avatar_url is None and not body.remove_avatar:
        raise HTTPException(status_code=400, detail="Nothing to update")
    conn = db.get_conn("auth")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        set_clauses = []
        params = []
        if body.name is not None:
            set_clauses.append("name = %s")
            params.append(body.name)
        if body.remove_avatar:
            set_clauses.append("avatar_url = NULL")
        elif body.avatar_url is not None:
            set_clauses.append("avatar_url = %s")
            params.append(body.avatar_url)
        set_clauses.append("updated_at = now()")
        params.append(user.user_id)
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE users SET {', '.join(set_clauses)} WHERE user_id = %s "
                f"RETURNING user_id, email, role, is_active, name, avatar_url",
                params,
            )
            row = cur.fetchone()
        return UserRecord(**row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("update_profile error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update profile") from e
    finally:
        conn.close()


def change_password(body: ChangePasswordRequest, user: UserRecord):
    pwd_ctx = _passlib()
    conn = db.get_conn("auth")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash FROM users WHERE user_id = %s",
                (user.user_id,)
            )
            row = cur.fetchone()
            if not row or not pwd_ctx.verify(body.current_password, row["password_hash"]):
                raise HTTPException(status_code=401, detail="Current password is incorrect")
            new_hash = _hash_password(pwd_ctx, body.new_password)
            cur.execute(
                "UPDATE users SET password_hash = %s, updated_at = now() WHERE user_id = %s",
                (new_hash, user.user_id)
            )
        return {"message": "Password updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("change_password error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to change password") from e
    finally:
        conn.close()


def admin_reset_password(body: AdminResetRequest, admin: UserRecord):
    pwd_ctx = _passlib()
    conn = db.get_conn("auth")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, email FROM users WHERE user_id = %s AND created_by = %s",
                (body.user_id, admin.user_id),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Team member not found")
            new_hash = _hash_password(pwd_ctx, body.new_password)
            cur.execute(
                "UPDATE users SET password_hash = %s, updated_at = now() WHERE user_id = %s",
                (new_hash, body.user_id)
            )
        return {"message": f"Password reset for {row['email']}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("admin_reset_password error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reset password") from e
    finally:
        conn.close()


def list_admin_users(admin: UserRecord):
    conn = db.get_conn("auth")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT max_sub_users FROM users WHERE user_id = %s", (admin.user_id,))
            admin_row = cur.fetchone()
            max_sub_users = admin_row["max_sub_users"] if admin_row else 0

            cur.execute(
                """
                SELECT user_id, email, role, is_active, name, created_at
                FROM users WHERE created_by = %s ORDER BY created_at DESC
                """,
                (admin.user_id,),
            )
            members = cur.fetchall()
        return TeamMembersResponse(members=members, max_sub_users=max_sub_users)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("list_admin_users error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list team members") from e
    finally:
        conn.close()


def add_admin_user(body: AdminCreateUserRequest, admin: UserRecord):
    pwd_ctx = _passlib()
    conn = db.get_conn("auth")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM users WHERE email = %s", (body.email,))
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="Email already registered")

            cur.execute("SELECT max_sub_users FROM users WHERE user_id = %s", (admin.user_id,))
            admin_row = cur.fetchone()
            max_sub_users = admin_row["max_sub_users"] if admin_row else 0

            cur.execute(
                "SELECT COUNT(*) AS cnt FROM users WHERE created_by = %s AND is_active = true",
                (admin.user_id,),
            )
            current_count = cur.fetchone()["cnt"]
            if current_count >= max_sub_users:
                raise HTTPException(status_code=403, detail="Team member limit reached")

            user_id = str(uuid.uuid4())
            hashed = _hash_password(pwd_ctx, body.password)
            cur.execute(
                """
                INSERT INTO users (user_id, email, password_hash, role, created_by)
                VALUES (%s, %s, %s, 'user', %s)
                RETURNING user_id, email, role, is_active, name, created_at
                """,
                (user_id, body.email, hashed, admin.user_id),
            )
            row = cur.fetchone()
        member = TeamMember(**row)
        allocation = assign_initial_allocation(admin.user_id, user_id, body.allocated_tokens)
        if allocation:
            member.allocated_tokens, member.allocation_clamped = allocation
        return member
    except HTTPException:
        raise
    except Exception as e:
        logger.error("add_admin_user error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to add team member") from e
    finally:
        conn.close()


def set_admin_user_status(body: AdminUserStatusRequest, admin: UserRecord):
    conn = db.get_conn("auth")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id FROM users WHERE user_id = %s AND created_by = %s",
                (body.user_id, admin.user_id),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Team member not found")

            cur.execute(
                "UPDATE users SET is_active = %s, updated_at = now() WHERE user_id = %s",
                (body.active, body.user_id),
            )
        return {"status": "success", "user_id": body.user_id, "active": body.active}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("set_admin_user_status error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update team member") from e
    finally:
        conn.close()


def update_admin_user(user_id: str, body: AdminUpdateUserRequest, admin: UserRecord):
    if body.name is None:
        raise HTTPException(status_code=400, detail="Nothing to update")
    conn = db.get_conn("auth")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id FROM users WHERE user_id = %s AND created_by = %s",
                (user_id, admin.user_id),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Team member not found")

            cur.execute(
                """
                UPDATE users SET name = %s, updated_at = now() WHERE user_id = %s
                RETURNING user_id, email, role, is_active, name, created_at
                """,
                (body.name, user_id),
            )
            row = cur.fetchone()
        return TeamMember(**row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("update_admin_user error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update team member") from e
    finally:
        conn.close()


def delete_admin_user(user_id: str, admin: UserRecord):
    conn = db.get_conn("auth")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id FROM users WHERE user_id = %s AND created_by = %s",
                (user_id, admin.user_id),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Team member not found")

            cur.execute("DELETE FROM chat_sessions WHERE owner_id = %s", (user_id,))
            cur.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
        # MS-402: hand their token slice back to the workspace pool.
        release_member_allocation(user_id)
        return {"status": "success", "user_id": user_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_admin_user error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete team member") from e
    finally:
        conn.close()
