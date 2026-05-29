# router/auth.py
"""
Authentication & RBAC
---------------------
Endpoints:
  POST /auth/register   — create account (role defaults to "user")
  POST /auth/login      — returns JWT access token
  GET  /auth/me         — returns current user (requires valid token)

RBAC dependency helpers (importable by other routers):
  get_current_user(token)        → UserRecord (any authenticated user)
  require_role("admin")          → raises 403 if role doesn't match
  CurrentUser                    → FastAPI Depends shortcut
  AdminOnly                      → FastAPI Depends shortcut (admin only)

Schema (auto-created via ensure_schema in storage.py):
  users (user_id, email, password_hash, role, is_active, created_at, updated_at)
"""

from __future__ import annotations

import os
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, field_validator

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Config — read from env with sensible defaults
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET environment variable is not set")
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 h

_bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Lazy imports so startup doesn't fail if packages missing
# ---------------------------------------------------------------------------

def _jose():
    from jose import jwt, JWTError  # noqa: F401
    return jwt

def _passlib():
    from passlib.context import CryptContext
    return CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__truncate_error=False)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str
    role: str = "user"  # "user" | "admin"

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "admin"):
            raise ValueError("role must be 'user' or 'admin'")
        return v

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    role: str

class UserRecord(BaseModel):
    user_id: str
    email: str
    role: str
    is_active: bool


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_conn():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        url = database_url
        if "sslmode=" not in url:
            sep = "&" if "?" in url else "?"
            url = url + sep + "sslmode=require"
        conn = psycopg2.connect(url, cursor_factory=RealDictCursor, connect_timeout=10)
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.warning("auth: DB connection failed: %s", e)
        return None


def _ensure_users_table(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            BIGSERIAL    PRIMARY KEY,
                    user_id       TEXT         NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    email         TEXT         NOT NULL UNIQUE,
                    password_hash TEXT         NOT NULL,
                    role          TEXT         NOT NULL DEFAULT 'user'
                                  CHECK (role IN ('user', 'admin')),
                    is_active     BOOLEAN      NOT NULL DEFAULT true,
                    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_users_email   ON users (email);
                CREATE INDEX IF NOT EXISTS idx_users_user_id ON users (user_id);
            """)
    except Exception as e:
        logger.warning("auth: ensure users table failed: %s", e)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _create_token(user_id: str, email: str, role: str) -> str:
    jwt = _jose()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "email": email, "role": role, "exp": expire}
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
    """Require a valid JWT. Returns the decoded user record."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = _decode_token(credentials.credentials)
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/auth/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest):
    """Register a new user. First registered user becomes admin automatically."""
    pwd_ctx = _passlib()
    conn = _get_conn()

    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_users_table(conn)

    try:
        with conn.cursor() as cur:
            # Check email already exists
            cur.execute("SELECT user_id FROM users WHERE email = %s", (body.email,))
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="Email already registered")

            # First user in DB gets admin role regardless of request
            cur.execute("SELECT COUNT(*) AS cnt FROM users")
            count = cur.fetchone()["cnt"]
            role = "admin" if count == 0 else body.role

            user_id = str(uuid.uuid4())
            hashed  = pwd_ctx.hash(body.password)

            cur.execute("""
                INSERT INTO users (user_id, email, password_hash, role)
                VALUES (%s, %s, %s, %s)
                RETURNING user_id, email, role
            """, (user_id, body.email, hashed, role))
            row = cur.fetchone()

        conn.close()
        token = _create_token(row["user_id"], row["email"], row["role"])
        return TokenResponse(
            access_token=token,
            user_id=row["user_id"],
            email=row["email"],
            role=row["role"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("auth register error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")


@router.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """Login and receive a JWT access token."""
    pwd_ctx = _passlib()
    conn = _get_conn()

    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_users_table(conn)

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, email, password_hash, role, is_active FROM users WHERE email = %s",
                (body.email,)
            )
            row = cur.fetchone()
        conn.close()
    except Exception as e:
        logger.error("auth login db error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Login failed: {e}")

    if not row or not pwd_ctx.verify(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not row["is_active"]:
        raise HTTPException(status_code=403, detail="Account is disabled")

    token = _create_token(row["user_id"], row["email"], row["role"])
    return TokenResponse(
        access_token=token,
        user_id=row["user_id"],
        email=row["email"],
        role=row["role"],
    )


@router.get("/auth/me", response_model=UserRecord)
async def me(user: UserRecord = Depends(get_current_user)):
    """Return the currently authenticated user."""
    return user
