# router/sessions.py
"""
Chat session persistence endpoints.

Schema (two tables):
  chat_sessions  - metadata (title, collections, timestamps)
  chat_messages  - one row per message (FK to chat_sessions, CASCADE delete)

Falls back to in-memory dict if DATABASE_URL is not set or DB is unreachable.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timezone
import logging
import uuid
import os

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class StoredMessage(BaseModel):
    id: str
    role: str          # "user" | "assistant"
    content: str
    model_used: Optional[str] = None
    created_at: str    # ISO

class UpsertSessionRequest(BaseModel):
    session_id: Optional[str] = None   # if None -> create new
    title: str
    messages: List[StoredMessage]
    pdf_collections: Optional[List[str]] = []
    chat_collections: Optional[List[str]] = []

class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[StoredMessage]
    pdf_collections: List[str]
    chat_collections: List[str]

class SessionSummary(BaseModel):
    session_id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str
    pdf_collections: List[str]
    chat_collections: List[str]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_conn():
    """Return a psycopg2 RealDictCursor connection or None if unavailable."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        # Embed sslmode in URL to avoid kwarg conflict with Supabase pooler DSN
        url = database_url
        if "sslmode=" not in url:
            sep = "&" if "?" in url else "?"
            url = url + sep + "sslmode=require"
        conn = psycopg2.connect(url, cursor_factory=RealDictCursor, connect_timeout=10)
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.warning("sessions: DB connection failed: %s", e)
        return None


def _ensure_tables(conn):
    """Create chat_sessions + chat_messages tables if they do not exist."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id               BIGSERIAL PRIMARY KEY,
                    session_id       TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    title            TEXT        NOT NULL DEFAULT '',
                    pdf_collections  TEXT[]      NOT NULL DEFAULT '{}',
                    chat_collections TEXT[]      NOT NULL DEFAULT '{}',
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_sid
                    ON chat_sessions (session_id);
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated
                    ON chat_sessions (updated_at DESC);

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id          BIGSERIAL   PRIMARY KEY,
                    message_id  TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    session_id  TEXT        NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                    role        TEXT        NOT NULL,
                    content     TEXT        NOT NULL DEFAULT '',
                    model_used  TEXT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                    ON chat_messages (session_id, created_at ASC);
            """)
    except Exception as e:
        logger.warning("sessions: ensure tables failed: %s", e)


def _ts(val) -> str:
    """Convert a datetime or string to ISO string."""
    if val is None:
        return datetime.now(timezone.utc).isoformat()
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


# ---------------------------------------------------------------------------
# In-memory fallback (when no DATABASE_URL or DB unreachable)
# ---------------------------------------------------------------------------

_memory_store: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/sessions/debug/status")
async def sessions_debug():
    """Check if sessions DB is reachable."""
    has_db_url = bool(os.getenv("DATABASE_URL"))
    conn = _get_conn()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS cnt FROM chat_sessions")
                row = cur.fetchone()
                cur.execute("SELECT COUNT(*) AS msg_cnt FROM chat_messages")
                msg_row = cur.fetchone()
            conn.close()
            return {
                "storage": "postgresql",
                "session_count": row["cnt"],
                "message_count": msg_row["msg_cnt"],
                "db_url_set": True,
            }
        except Exception as e:
            return {"storage": "postgresql_error", "error": str(e), "db_url_set": has_db_url}
    return {
        "storage": "memory",
        "session_count": len(_memory_store),
        "db_url_set": has_db_url,
        "note": "DATABASE_URL missing or unreachable - sessions lost on restart",
    }


@router.post("/sessions", response_model=SessionResponse)
async def upsert_session(body: UpsertSessionRequest):
    """
    Create or update a session.
    Session metadata goes into chat_sessions.
    Each message is upserted into chat_messages (by message_id).
    """
    now = datetime.now(timezone.utc).isoformat()
    sid = body.session_id or str(uuid.uuid4())

    conn = _get_conn()
    if conn:
        _ensure_tables(conn)
        try:
            with conn.cursor() as cur:
                # 1. Upsert session row
                cur.execute("""
                    INSERT INTO chat_sessions
                        (session_id, title, pdf_collections, chat_collections, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, now(), now())
                    ON CONFLICT (session_id) DO UPDATE
                        SET title            = EXCLUDED.title,
                            pdf_collections  = EXCLUDED.pdf_collections,
                            chat_collections = EXCLUDED.chat_collections,
                            updated_at       = now()
                    RETURNING session_id, title, pdf_collections, chat_collections,
                              created_at, updated_at
                """, (
                    sid,
                    body.title,
                    body.pdf_collections or [],
                    body.chat_collections or [],
                ))
                session_row = cur.fetchone()

                # 2. Upsert each message (idempotent on message_id)
                for m in body.messages:
                    cur.execute("""
                        INSERT INTO chat_messages
                            (message_id, session_id, role, content, model_used, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (message_id) DO UPDATE
                            SET content    = EXCLUDED.content,
                                model_used = EXCLUDED.model_used
                    """, (
                        m.id,
                        sid,
                        m.role,
                        m.content,
                        m.model_used,
                        m.created_at,
                    ))

                # 3. Return all messages for this session
                cur.execute("""
                    SELECT message_id, role, content, model_used, created_at
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY created_at ASC, id ASC
                """, (sid,))
                msg_rows = cur.fetchall()

            conn.close()
            return SessionResponse(
                session_id=session_row["session_id"],
                title=session_row["title"],
                created_at=_ts(session_row["created_at"]),
                updated_at=_ts(session_row["updated_at"]),
                messages=[
                    StoredMessage(
                        id=r["message_id"],
                        role=r["role"],
                        content=r["content"],
                        model_used=r["model_used"],
                        created_at=_ts(r["created_at"]),
                    )
                    for r in msg_rows
                ],
                pdf_collections=list(session_row["pdf_collections"] or []),
                chat_collections=list(session_row["chat_collections"] or []),
            )
        except Exception as e:
            logger.error("sessions upsert DB error: %s", e)
            try:
                conn.close()
            except Exception:
                pass

    # Memory fallback
    existing = _memory_store.get(sid, {})
    record = {
        "session_id": sid,
        "title": body.title,
        "messages": [m.model_dump() for m in body.messages],
        "pdf_collections": body.pdf_collections or [],
        "chat_collections": body.chat_collections or [],
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    _memory_store[sid] = record
    return SessionResponse(**record)


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions():
    """Return all sessions ordered by most recent, with message counts."""
    conn = _get_conn()
    if conn:
        _ensure_tables(conn)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT s.session_id,
                           s.title,
                           s.pdf_collections,
                           s.chat_collections,
                           s.created_at,
                           s.updated_at,
                           COUNT(m.id) AS message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON m.session_id = s.session_id
                    GROUP BY s.session_id, s.title, s.pdf_collections,
                             s.chat_collections, s.created_at, s.updated_at
                    ORDER BY s.updated_at DESC
                    LIMIT 200
                """)
                rows = cur.fetchall()
            conn.close()
            return [
                SessionSummary(
                    session_id=r["session_id"],
                    title=r["title"],
                    message_count=int(r["message_count"] or 0),
                    created_at=_ts(r["created_at"]),
                    updated_at=_ts(r["updated_at"]),
                    pdf_collections=list(r["pdf_collections"] or []),
                    chat_collections=list(r["chat_collections"] or []),
                )
                for r in rows
            ]
        except Exception as e:
            logger.error("sessions list DB error: %s", e)
            try:
                conn.close()
            except Exception:
                pass

    # Memory fallback
    return sorted(
        [
            SessionSummary(
                session_id=v["session_id"],
                title=v["title"],
                message_count=len(v["messages"]),
                created_at=v["created_at"],
                updated_at=v["updated_at"],
                pdf_collections=v["pdf_collections"],
                chat_collections=v["chat_collections"],
            )
            for v in _memory_store.values()
        ],
        key=lambda s: s.updated_at,
        reverse=True,
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Return full session including all messages from chat_messages table."""
    conn = _get_conn()
    if conn:
        _ensure_tables(conn)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_id, title, pdf_collections, chat_collections,
                           created_at, updated_at
                    FROM chat_sessions WHERE session_id = %s
                """, (session_id,))
                session_row = cur.fetchone()
                if not session_row:
                    conn.close()
                    raise HTTPException(status_code=404, detail="Session not found")

                cur.execute("""
                    SELECT message_id, role, content, model_used, created_at
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY created_at ASC, id ASC
                """, (session_id,))
                msg_rows = cur.fetchall()
            conn.close()
            return SessionResponse(
                session_id=session_row["session_id"],
                title=session_row["title"],
                created_at=_ts(session_row["created_at"]),
                updated_at=_ts(session_row["updated_at"]),
                messages=[
                    StoredMessage(
                        id=r["message_id"],
                        role=r["role"],
                        content=r["content"],
                        model_used=r["model_used"],
                        created_at=_ts(r["created_at"]),
                    )
                    for r in msg_rows
                ],
                pdf_collections=list(session_row["pdf_collections"] or []),
                chat_collections=list(session_row["chat_collections"] or []),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("sessions get DB error: %s", e)
            try:
                conn.close()
            except Exception:
                pass

    # Memory fallback
    record = _memory_store.get(session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(**record)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session. Messages are cascade-deleted automatically."""
    conn = _get_conn()
    if conn:
        _ensure_tables(conn)
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
            conn.close()
        except Exception as e:
            logger.error("sessions delete DB error: %s", e)
            try:
                conn.close()
            except Exception:
                pass

    _memory_store.pop(session_id, None)
    return {"status": "deleted", "session_id": session_id}
