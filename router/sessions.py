# router/sessions.py
"""
Chat session persistence endpoints.
Stores full Q&A conversation history in PostgreSQL (chat_sessions table).
Falls back to in-memory dict if DATABASE_URL is not set.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from datetime import datetime, timezone
import logging
import uuid
import json
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
    session_id: Optional[str] = None   # if None → create new
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
# DB helpers (psycopg2 via DATABASE_URL)
# ---------------------------------------------------------------------------

def _get_conn():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(database_url, sslmode="require", cursor_factory=RealDictCursor)
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.warning("sessions: DB connection failed: %s", e)
        return None


def _ensure_table(conn):
    """Create chat_sessions table if it doesn't exist."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id              BIGSERIAL PRIMARY KEY,
                    session_id      TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    title           TEXT        NOT NULL DEFAULT '',
                    messages        JSONB       NOT NULL DEFAULT '[]',
                    pdf_collections TEXT[]      NOT NULL DEFAULT '{}',
                    chat_collections TEXT[]     NOT NULL DEFAULT '{}',
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_sid
                    ON chat_sessions (session_id);
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated
                    ON chat_sessions (updated_at DESC);
            """)
    except Exception as e:
        logger.warning("sessions: ensure table failed: %s", e)


# ---------------------------------------------------------------------------
# In-memory fallback (when no DATABASE_URL)
# ---------------------------------------------------------------------------

_memory_store: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/sessions", response_model=SessionResponse)
async def upsert_session(body: UpsertSessionRequest):
    """Create a new session or update an existing one (upsert by session_id)."""
    now = datetime.now(timezone.utc).isoformat()
    sid = body.session_id or str(uuid.uuid4())
    messages_json = [m.model_dump() for m in body.messages]

    conn = _get_conn()
    if conn:
        _ensure_table(conn)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_sessions
                        (session_id, title, messages, pdf_collections, chat_collections, created_at, updated_at)
                    VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE
                        SET title            = EXCLUDED.title,
                            messages         = EXCLUDED.messages,
                            pdf_collections  = EXCLUDED.pdf_collections,
                            chat_collections = EXCLUDED.chat_collections,
                            updated_at       = EXCLUDED.updated_at
                    RETURNING session_id, title, messages, pdf_collections, chat_collections,
                              created_at, updated_at
                """, (
                    sid,
                    body.title,
                    json.dumps(messages_json),
                    body.pdf_collections or [],
                    body.chat_collections or [],
                    now, now,
                ))
                row = cur.fetchone()
            conn.close()
            return SessionResponse(
                session_id=row["session_id"],
                title=row["title"],
                created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
                updated_at=row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else str(row["updated_at"]),
                messages=[StoredMessage(**m) for m in (row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"]))],
                pdf_collections=list(row["pdf_collections"] or []),
                chat_collections=list(row["chat_collections"] or []),
            )
        except Exception as e:
            logger.error("sessions upsert DB error: %s", e)
            conn.close()
            # fall through to memory

    # Memory fallback
    existing = _memory_store.get(sid, {})
    record = {
        "session_id": sid,
        "title": body.title,
        "messages": messages_json,
        "pdf_collections": body.pdf_collections or [],
        "chat_collections": body.chat_collections or [],
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    _memory_store[sid] = record
    return SessionResponse(**record)


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions():
    """Return all sessions ordered by most recent, without full message bodies."""
    conn = _get_conn()
    if conn:
        _ensure_table(conn)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_id, title,
                           jsonb_array_length(messages) AS message_count,
                           pdf_collections, chat_collections,
                           created_at, updated_at
                    FROM chat_sessions
                    ORDER BY updated_at DESC
                    LIMIT 200
                """)
                rows = cur.fetchall()
            conn.close()
            return [
                SessionSummary(
                    session_id=r["session_id"],
                    title=r["title"],
                    message_count=r["message_count"] or 0,
                    created_at=r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"]),
                    updated_at=r["updated_at"].isoformat() if hasattr(r["updated_at"], "isoformat") else str(r["updated_at"]),
                    pdf_collections=list(r["pdf_collections"] or []),
                    chat_collections=list(r["chat_collections"] or []),
                )
                for r in rows
            ]
        except Exception as e:
            logger.error("sessions list DB error: %s", e)
            conn.close()

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
    """Return full session including all messages."""
    conn = _get_conn()
    if conn:
        _ensure_table(conn)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_id, title, messages, pdf_collections, chat_collections,
                           created_at, updated_at
                    FROM chat_sessions WHERE session_id = %s
                """, (session_id,))
                row = cur.fetchone()
            conn.close()
            if not row:
                raise HTTPException(status_code=404, detail="Session not found")
            msgs = row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"])
            return SessionResponse(
                session_id=row["session_id"],
                title=row["title"],
                created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
                updated_at=row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else str(row["updated_at"]),
                messages=[StoredMessage(**m) for m in msgs],
                pdf_collections=list(row["pdf_collections"] or []),
                chat_collections=list(row["chat_collections"] or []),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("sessions get DB error: %s", e)
            conn.close()

    # Memory fallback
    record = _memory_store.get(session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(**record)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session by ID."""
    conn = _get_conn()
    if conn:
        _ensure_table(conn)
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
            conn.close()
        except Exception as e:
            logger.error("sessions delete DB error: %s", e)
            conn.close()

    _memory_store.pop(session_id, None)
    return {"status": "deleted", "session_id": session_id}
