# router/sessions.py
"""
Chat session persistence endpoints. Each route reads the request and calls
services/sessions.py, where the SQL and the paging/search rules live.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from models import (
    RenameSessionRequest,
    SessionQuestionsResponse,
    SessionResponse,
    SessionSummary,
    UpsertSessionRequest,
)
from router.auth import get_current_user, UserRecord
from services import sessions
from services.sessions import PAGE_SIZE_DEFAULT

router = APIRouter()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/sessions", response_model=SessionResponse)
async def upsert_session(
    body: UpsertSessionRequest,
    user: UserRecord = Depends(get_current_user),
):
    """
    Create or update a session.
    Session metadata goes into chat_sessions.
    Each message is upserted into chat_messages (by message_id).
    """
    return sessions.upsert_session(body, user)


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions(
    q: Optional[str] = None,
    user: UserRecord = Depends(get_current_user),
):
    """Return sessions owned by the current user (all sessions for admins),
    ordered by most recent, with message counts.

    If `q` is given, also matches sessions whose title OR any message
    content contains it (case-insensitive), not just the title.
    """
    return sessions.list_sessions(q, user)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    limit: int = Query(PAGE_SIZE_DEFAULT, ge=1, le=100),
    before: Optional[str] = None,
    user: UserRecord = Depends(get_current_user),
):
    """Return a session's most recent `limit` chats — one chat being one
    question plus the answer(s) that followed it, so `limit=5` is 5 complete
    exchanges, not 5 message rows. MS-237: was "all messages, always"; a
    long-lived conversation no longer dumps its entire history into one
    response. Pass the previous response's `next_cursor` as `before` to page
    further back; `has_more`/`next_cursor` come back falsy once the start of
    the conversation is reached."""
    return sessions.get_session(session_id, limit, before, user)


@router.get("/sessions/{session_id}/questions", response_model=SessionQuestionsResponse)
async def get_session_questions(
    session_id: str,
    user: UserRecord = Depends(get_current_user),
):
    """Every question asked in this session, as a flat index: turn number,
    the message id to scroll to, and a short preview.

    MS-237: this backs the chat navigation panel, which lists the whole
    conversation while the thread itself is still paginated 5 chats at a
    time. Content is truncated in SQL so the response stays small no matter
    how long the individual messages are — the panel only ever shows one
    line per question."""
    return sessions.get_session_questions(session_id, user)


@router.put("/sessions/{session_id}", response_model=SessionSummary)
async def rename_session(
    session_id: str,
    body: RenameSessionRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Rename a session. Title only — messages are untouched, so this never
    needs the client to resend the whole conversation just to relabel it."""
    return sessions.rename_session(session_id, body, user)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: UserRecord = Depends(get_current_user)):
    """Delete a session. Messages are cascade-deleted automatically."""
    return sessions.delete_session(session_id, user)
