# billing/token_requests.py
"""Members asking their workspace admin for a bigger allocation."""

from __future__ import annotations

import uuid

from fastapi import HTTPException

import app_db
import db
from billing.plans import resolve_admin_user_id
from models import (
    CreateTokenRequestRequest,
    TokenRequestRecord,
    TokenRequestResponse,
    TokenRequestsResponse,
)
from router.auth import UserRecord


def request_more_tokens(body: CreateTokenRequestRequest, user: UserRecord):
    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        admin_user_id = resolve_admin_user_id(conn, user)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT request_id FROM token_requests WHERE user_id = %s AND status = 'pending'",
                (user.user_id,),
            )
            existing = cur.fetchone()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="You already have a pending request — wait for your admin to respond to it first.",
            )

        request_id = f"treq_{uuid.uuid4().hex}"
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO token_requests (request_id, user_id, admin_user_id, message)
                VALUES (%s, %s, %s, %s)
                RETURNING created_at
                """,
                (request_id, user.user_id, admin_user_id, body.message),
            )
            created_at = cur.fetchone()["created_at"]
    finally:
        conn.close()

    return TokenRequestResponse(
        request=TokenRequestRecord(
            request_id=request_id,
            user_id=user.user_id,
            email=user.email,
            message=body.message,
            status="pending",
            created_at=app_db.ts(created_at),
        )
    )


def list_token_requests(admin: UserRecord):
    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.request_id, r.user_id, u.email, r.message, r.status, r.created_at
                FROM token_requests r
                JOIN users u ON u.user_id = r.user_id
                WHERE r.admin_user_id = %s
                ORDER BY (r.status = 'pending') DESC, r.created_at DESC
                LIMIT 50
                """,
                (admin.user_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    requests = [
        TokenRequestRecord(
            request_id=row["request_id"],
            user_id=row["user_id"],
            email=row["email"],
            message=row["message"],
            status=row["status"],
            created_at=app_db.ts(row["created_at"]),
        )
        for row in rows
    ]
    pending_count = sum(1 for r in requests if r.status == "pending")
    return TokenRequestsResponse(requests=requests, pending_count=pending_count)


def dismiss_token_request(request_id: str, admin: UserRecord):
    conn = db.get_conn("payment")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE token_requests SET status = 'resolved', resolved_at = now()
                WHERE request_id = %s AND admin_user_id = %s
                RETURNING user_id, message, created_at
                """,
                (request_id, admin.user_id),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Request not found")

        with conn.cursor() as cur:
            cur.execute("SELECT email FROM users WHERE user_id = %s", (row["user_id"],))
            user_row = cur.fetchone()
    finally:
        conn.close()

    return TokenRequestResponse(
        request=TokenRequestRecord(
            request_id=request_id,
            user_id=row["user_id"],
            email=user_row["email"] if user_row else "",
            message=row["message"],
            status="resolved",
            created_at=app_db.ts(row["created_at"]),
        )
    )
