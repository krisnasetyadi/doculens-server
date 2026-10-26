# services/telegram.py
"""
Telegram as a live chat source (Telethon/MTProto), not a file upload.

Flow: connect (phone -> OTP -> optional 2FA) -> pick
which chats/groups to bring in -> sync (repeatable) -> each selected chat
becomes a searchable ChatCollection via the same pipeline WhatsApp uploads
use (services/ingest/chat_ingest.py).

History range (MS-912): each selected chat keeps the last month, 3 months, year
or everything (telegram_selected_chats.history_range). Sync fetches only that
range and rebuilds the collection from it, so every later sync drops the messages
that have fallen out of it. A single chat can be removed on its own, with all its
stored data, without deleting the connection (remove_chat).

The Telegram *application* every new connection signs in through is
DocuLens's own registration from server config (TELEGRAM_API_ID /
TELEGRAM_API_HASH, MS-911), so admins never register one. A start request may
still carry its own api_id/api_hash (clients older than MS-911), and every
saved connection keeps the pair it was made with, because a session only
works with the app that created it.

Mirrors services/database_connections.py's structure (its own small schema in
the app's own database, admin-only, connect/list/activate/delete) since a
Telegram connection is the same shape of thing: a live, credentialed,
user-managed source rather than a one-time file.

SECURITY: a saved Telethon session is equivalent to full access to the
Telegram account that logged in — read every chat, send messages as that
user. Both the session and the connection's api_hash are encrypted at rest
(services/security/telegram_crypto.py) and never returned to the client; the
phone number is only stored masked (schema/telegram.py masks older rows). Losing
TELEGRAM_SESSION_ENCRYPTION_KEY makes stored secrets permanently
undecryptable (by design — there is no server-side escape hatch).

QR sign-in (the default since MS-911): /connect/qr/start asks Telegram for a
login token and returns it as a tg://login link the client draws as a QR code.
A background task per flow waits for the scan, renews the token whenever it
expires and gives up after QR_LOGIN_LIMIT_SECONDS. The client polls
/connect/qr/status; an account with two-step verification finishes on
/connect/verify with the same flow_id.

KNOWN LIMITATION: the connect flow's in-progress Telethon client lives in an
in-memory dict between /connect/start and /connect/verify. It does not
survive a backend restart or a multi-worker deployment — acceptable for this
app's single-process deployment, same assumption services/rag/processor.py's in-memory
vector_store_cache already makes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from services.storage import app_db
import db
from config import config
from models import (
    ChatMessage,
    ChatPlatform,
    SetTelegramConnectionActiveRequest,
    TelegramChatRangeRequest,
    TelegramChatRemovedResponse,
    TelegramConnectionsResponse,
    TelegramConnectionSource,
    TelegramConnectStartRequest,
    TelegramConnectStartResponse,
    TelegramConnectVerifyRequest,
    TelegramConnectVerifyResponse,
    TelegramDialog,
    TelegramDialogsResponse,
    TelegramQrCancelRequest,
    TelegramQrLoginState,
    TelegramQrStartRequest,
    TelegramSelectedChat,
    TelegramSyncRequest,
    TelegramSyncResponse,
    TelegramSyncResult,
    UserRecord,
)
from services import chat as chat_service
from services.billing.plans import resolve_storage_limits
from services.ingest.chat_ingest import ingest_chat_messages
from services.storage import limits as storage_limits
from services.storage import supabase as supabase_storage
from services.security.telegram_crypto import encrypt_secret, decrypt_secret

logger = logging.getLogger(__name__)

PENDING_LOGIN_TTL_SECONDS = 10 * 60
QR_LOGIN_LIMIT_SECONDS = 3 * 60

# flow_id -> {"client": TelegramClient, "api_id": int, "api_hash": str,
#             "phone": Optional[str], "phone_code_hash": str, "label": Optional[str],
#             "created_at": float}
# A QR flow has no phone/phone_code_hash and adds "qr", "task", "status",
# "connection" and "error" (see start_qr_connect).
_pending_logins: Dict[str, Dict[str, Any]] = {}


def _require_telegram_enabled() -> None:
    if not config.telegram_enabled:
        raise HTTPException(
            status_code=503,
            detail="Telegram integration is not configured — set TELEGRAM_SESSION_ENCRYPTION_KEY in the backend .env",
        )


def _app_credentials(api_id: Optional[int], api_hash: Optional[str]) -> tuple[int, str]:
    """The Telegram app a new sign-in goes through: the one the request names
    (clients older than MS-911), otherwise DocuLens's own registration."""
    if api_id is not None or api_hash is not None:
        if not api_id:
            raise HTTPException(status_code=400, detail="API ID is required")
        if not (api_hash or "").strip():
            raise HTTPException(status_code=400, detail="API hash is required")
        return api_id, api_hash.strip()
    if not config.telegram_app_configured:
        raise HTTPException(
            status_code=503,
            detail="Telegram is not configured on this server: set TELEGRAM_API_ID and TELEGRAM_API_HASH in the backend .env",
        )
    return config.telegram_api_id, config.telegram_api_hash.strip()


def _cleanup_pending_logins() -> None:
    now = time.time()
    expired = [fid for fid, entry in _pending_logins.items() if now - entry["created_at"] > PENDING_LOGIN_TTL_SECONDS]
    for fid in expired:
        _drop_pending_login(fid)


def _drop_pending_login(flow_id: str) -> None:
    """Forget a sign-in: stop its QR wait (if any) and let its client go."""
    entry = _pending_logins.pop(flow_id, None)
    if not entry:
        return
    task = entry.get("task")
    if task and not task.done():
        task.cancel()
    _release_client(entry)


def _release_client(entry: Dict[str, Any]) -> None:
    """Disconnect a pending sign-in's client, without waiting for it."""
    client, entry["client"] = entry.get("client"), None
    if not client:
        return
    try:
        # Telethon's disconnect() hands back a coroutine while the event loop
        # runs (every caller here is a request handler), and it does nothing
        # until that coroutine is scheduled.
        result = client.disconnect()
        if asyncio.iscoroutine(result):
            asyncio.ensure_future(result)
    except Exception:
        pass


def _mask_phone(phone: str) -> str:
    digits = phone.strip()
    if not digits:
        return ""
    if len(digits) <= 6:
        return digits[0] + "*" * max(len(digits) - 1, 0)
    # Always mask at least one character between the kept head/tail — at
    # exactly 7 chars, len(digits) - 7 == 0 would otherwise leave nothing masked.
    return f"{digits[:4]}{'*' * max(len(digits) - 7, 1)}{digits[-3:]}"


def _dialog_type(entity: Any) -> str:
    if getattr(entity, "broadcast", False):
        return "channel"
    if getattr(entity, "megagroup", False) or type(entity).__name__ == "Chat":
        return "group"
    if type(entity).__name__ == "User":
        return "user"
    return "channel"


def _telegram_transcript_name(title: str, dialog_id: int) -> str:
    safe_title = re.sub(r"[^\w .()-]", "_", title).strip(" ._")[:80]
    return f"telegram-{safe_title or dialog_id}.txt"


def _write_telegram_transcript(path: str, messages: List[ChatMessage]) -> None:
    """One preview line per indexed message, including sender and timestamp."""
    with open(path, "w", encoding="utf-8") as transcript:
        for message in messages:
            sender = message.sender.replace("\r", " ").replace("\n", " ")
            content = message.content.replace("\\", "\\\\").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
            transcript.write(f"{message.timestamp:%Y-%m-%d %H:%M:%S} | {sender}: {content}\n")


# ---------------------------------------------------------------------------
# App-DB storage (connection metadata + selected chats — NOT the chat content
# itself, which flows through services/ingest/chat_ingest.py into chat_collections like any
# other chat source).
# ---------------------------------------------------------------------------

def _telegram_user_id() -> str:
    return os.getenv("DB_CONNECTIONS_USER_ID", os.getenv("DEV_USER_ID", "db-connections-local-user"))


_SELECTED_CHAT_COLUMNS = (
    "dialog_id, dialog_title, dialog_type, chat_collection_id, message_count, history_range, status, last_synced_at"
)


def _as_selected_chat(r: Dict[str, Any]) -> TelegramSelectedChat:
    return TelegramSelectedChat(
        dialog_id=r["dialog_id"],
        title=r["dialog_title"],
        type=r["dialog_type"],
        chat_collection_id=r.get("chat_collection_id"),
        message_count=r.get("message_count"),
        history_range=r["history_range"],
        status=r["status"],
        last_synced_at=app_db.ts(r["last_synced_at"]) if r.get("last_synced_at") else None,
    )


def _selected_chats_for(cur, connection_id: str) -> List[TelegramSelectedChat]:
    cur.execute(
        f"""
        SELECT {_SELECTED_CHAT_COLUMNS}
        FROM telegram_selected_chats
        WHERE connection_id = %s
        ORDER BY created_at ASC
        """,
        (connection_id,),
    )
    return [_as_selected_chat(r) for r in cur.fetchall() or []]


def _as_connection_source(cur, row: Dict[str, Any]) -> TelegramConnectionSource:
    return TelegramConnectionSource(
        connection_id=row["connection_id"],
        label=row["label"],
        phone_masked=_mask_phone(row["phone"]),
        status=row["status"],
        created_at=app_db.ts(row["created_at"]),
        selected_chats=_selected_chats_for(cur, row["connection_id"]),
    )


def _get_connection_row(connection_id: str) -> Optional[Dict[str, Any]]:
    conn = db.get_conn("telegram")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM telegram_connections WHERE connection_id = %s",
                (connection_id,),
            )
            return cur.fetchone()
    finally:
        conn.close()


def _upsert_selected_chat(
    connection_id: str, dialog_id: str, title: str, dialog_type: str,
    chat_collection_id: str, message_count: int, history_range: str,
) -> None:
    conn = db.get_conn("telegram")
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO telegram_selected_chats
                    (connection_id, dialog_id, dialog_title, dialog_type, chat_collection_id, message_count,
                     history_range, last_synced_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (connection_id, dialog_id) DO UPDATE SET
                    dialog_title       = EXCLUDED.dialog_title,
                    dialog_type        = EXCLUDED.dialog_type,
                    chat_collection_id = EXCLUDED.chat_collection_id,
                    message_count      = EXCLUDED.message_count,
                    history_range      = EXCLUDED.history_range,
                    last_synced_at     = now()
                """,
                (connection_id, dialog_id, title, dialog_type, chat_collection_id, message_count, history_range),
            )
    finally:
        conn.close()


def _get_selected_chat(connection_id: str, dialog_id: str) -> Optional[Dict[str, Any]]:
    """The saved row for one chat of a connection (its collection and range), or None."""
    conn = db.get_conn("telegram")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chat_collection_id, history_range FROM telegram_selected_chats
                WHERE connection_id = %s AND dialog_id = %s
                """,
                (connection_id, dialog_id),
            )
            return cur.fetchone()
    finally:
        conn.close()


# How far back each range reaches. A "month" is 30 days and a "year" 365: the
# range is a cut-off for what to keep, not a calendar, so it needs no more.
_HISTORY_RANGE_DAYS: Dict[str, Optional[int]] = {"1m": 30, "3m": 90, "1y": 365, "all": None}
_HISTORY_RANGE_WORDS = {"1m": "month", "3m": "3 months", "1y": "year"}


def _utcnow() -> datetime:
    """Now as a naive UTC time, the same kind as the message dates below."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _range_cutoff(history_range: str, now: datetime) -> Optional[datetime]:
    """Messages older than this are not kept; None keeps everything."""
    days = _HISTORY_RANGE_DAYS[history_range]
    return None if days is None else now - timedelta(days=days)


def _touch_connection_synced(connection_id: str) -> None:
    conn = db.get_conn("telegram")
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE telegram_connections SET last_synced_at = now() WHERE connection_id = %s",
                (connection_id,),
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Telethon client helpers
# ---------------------------------------------------------------------------

def _new_client(api_id: int, api_hash: str, session_string: Optional[str] = None):
    from telethon import TelegramClient
    from telethon.sessions import StringSession

    # device_model is the name the account owner sees in Telegram's
    # Settings > Devices, where they can also revoke this session.
    return TelegramClient(StringSession(session_string), api_id, api_hash, device_model="DocuLens")


async def _connect_with_saved_session(row: Dict[str, Any]):
    session_string = decrypt_secret(row["session_enc"])
    api_hash = decrypt_secret(row["api_hash_enc"])
    if session_string is None or api_hash is None:
        raise HTTPException(
            status_code=500,
            detail="Could not decrypt the saved Telegram credentials — reconnect this account.",
        )
    client = _new_client(row["api_id"], api_hash, session_string)
    await client.connect()
    if not await client.is_user_authorized():
        await client.disconnect()
        raise HTTPException(status_code=401, detail="Telegram session expired — reconnect this account.")
    return client


async def start_connect(body: TelegramConnectStartRequest):
    _require_telegram_enabled()
    _cleanup_pending_logins()

    phone = body.phone.strip()
    if not phone:
        raise HTTPException(status_code=400, detail="Phone number is required")
    api_id, api_hash = _app_credentials(body.api_id, body.api_hash)

    client = _new_client(api_id, api_hash)
    try:
        await client.connect()
        sent = await client.send_code_request(phone)
    except Exception as exc:
        try:
            await client.disconnect()
        except Exception:
            pass
        logger.error("telegram: send_code_request failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Could not send login code: {exc}")

    flow_id = str(uuid.uuid4())
    _pending_logins[flow_id] = {
        "client": client,
        "api_id": api_id,
        "api_hash": api_hash,
        "phone": phone,
        "phone_code_hash": sent.phone_code_hash,
        "label": (body.label or "").strip() or None,
        "created_at": time.time(),
    }
    return TelegramConnectStartResponse(flow_id=flow_id, phone=phone)


async def verify_connect(body: TelegramConnectVerifyRequest):
    _require_telegram_enabled()

    entry = _pending_logins.get(body.flow_id)
    if not entry:
        raise HTTPException(status_code=400, detail="Login session expired or not found — start over.")

    if "qr" in entry and entry["status"] != "password_required":
        raise HTTPException(status_code=400, detail="This QR sign-in is not waiting for a password")

    client = entry["client"]
    from telethon.errors import SessionPasswordNeededError, PhoneCodeInvalidError, PhoneCodeExpiredError

    try:
        if body.password:
            await client.sign_in(password=body.password)
        else:
            await client.sign_in(
                phone=entry["phone"],
                code=body.code,
                phone_code_hash=entry["phone_code_hash"],
            )
    except SessionPasswordNeededError:
        return TelegramConnectVerifyResponse(status="password_required", connection=None)
    except (PhoneCodeInvalidError, PhoneCodeExpiredError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid or expired code: {exc}")
    except Exception as exc:
        logger.error("telegram: sign_in failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Sign-in failed: {exc}")

    _pending_logins.pop(body.flow_id, None)
    source = await _save_signed_in_connection(entry)
    return TelegramConnectVerifyResponse(status="connected", connection=source)


async def _save_signed_in_connection(entry: Dict[str, Any]) -> TelegramConnectionSource:
    """Store the account entry["client"] has just signed in to, then let the
    client go. Shared by the code flow and the QR flow."""
    client = entry["client"]
    phone = entry.get("phone")
    if not phone:
        # A QR sign-in never typed a number: take it from the account itself.
        me = await client.get_me()
        raw = (getattr(me, "phone", None) or "").strip()
        phone = raw if not raw or raw.startswith("+") else f"+{raw}"
    session_string = client.session.save()
    await client.disconnect()

    conn = db.get_conn("telegram")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    connection_id = str(uuid.uuid4())
    label = entry["label"] or (f"Telegram ({_mask_phone(phone)})" if phone else "Telegram")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO telegram_connections
                    (connection_id, user_id, api_id, api_hash_enc, phone, label, session_enc, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
                RETURNING connection_id, label, phone, status, created_at
                """,
                (
                    connection_id,
                    _telegram_user_id(),
                    entry["api_id"],
                    encrypt_secret(entry["api_hash"]),
                    # Only the masked number is kept: nothing needs the full
                    # one after sign-in, and the list shows it masked anyway.
                    _mask_phone(phone),
                    label,
                    encrypt_secret(session_string),
                ),
            )
            row = cur.fetchone()
            return _as_connection_source(cur, row)
    except Exception as exc:
        logger.error("telegram: failed to persist connection: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save Telegram connection")
    finally:
        conn.close()


async def start_qr_connect(body: TelegramQrStartRequest) -> TelegramQrLoginState:
    _require_telegram_enabled()
    _cleanup_pending_logins()
    api_id, api_hash = _app_credentials(None, None)

    client = _new_client(api_id, api_hash)
    try:
        await client.connect()
        qr = await client.qr_login()
    except Exception as exc:
        try:
            await client.disconnect()
        except Exception:
            pass
        logger.error("telegram: qr_login failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Could not start the QR sign-in: {exc}")

    flow_id = str(uuid.uuid4())
    entry = {
        "client": client,
        "api_id": api_id,
        "api_hash": api_hash,
        "phone": None,
        "label": (body.label or "").strip() or None,
        "created_at": time.time(),
        "qr": qr,
        "status": "waiting",
        "connection": None,
        "error": None,
    }
    _pending_logins[flow_id] = entry
    # Kept on the entry so the task is not garbage-collected and can be cancelled.
    entry["task"] = asyncio.ensure_future(_wait_for_qr_scan(flow_id, entry))
    return _qr_state(flow_id, entry)


async def _wait_for_qr_scan(flow_id: str, entry: Dict[str, Any]) -> None:
    """Wait for the phone to scan, renewing the token as it expires, until
    QR_LOGIN_LIMIT_SECONDS. Telegram only reports the scan to a client that is
    already waiting, which is why this runs in the background, not per poll."""
    from telethon.errors import SessionPasswordNeededError

    qr = entry["qr"]
    deadline = entry["created_at"] + QR_LOGIN_LIMIT_SECONDS
    try:
        while True:
            left = deadline - time.time()
            if left <= 0:
                entry["status"] = "expired"
                _release_client(entry)
                return
            token_left = (qr.expires - datetime.now(timezone.utc)).total_seconds()
            try:
                await qr.wait(timeout=max(1.0, min(token_left, left)))
                break
            except asyncio.TimeoutError:
                if time.time() < deadline:
                    await qr.recreate()
            except SessionPasswordNeededError:
                # Scanned; the client stays signed in halfway for connect/verify.
                entry["status"] = "password_required"
                return
        entry["connection"] = await _save_signed_in_connection(entry)
        entry["client"] = None
        entry["status"] = "connected"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("telegram: qr sign-in %s failed: %s", flow_id, exc)
        entry["status"] = "error"
        entry["error"] = exc.detail if isinstance(exc, HTTPException) else "Telegram sign-in failed"
        _release_client(entry)


def _qr_state(flow_id: str, entry: Dict[str, Any]) -> TelegramQrLoginState:
    waiting = entry["status"] == "waiting"
    qr = entry["qr"]
    return TelegramQrLoginState(
        flow_id=flow_id,
        status=entry["status"],
        qr_url=qr.url if waiting else None,
        expires_at=qr.expires if waiting else None,
        connection=entry["connection"],
        error=entry["error"],
    )


async def qr_connect_status(flow_id: str) -> TelegramQrLoginState:
    _require_telegram_enabled()
    entry = _pending_logins.get(flow_id)
    if not entry or "qr" not in entry:
        raise HTTPException(status_code=404, detail="QR sign-in not found or expired, start again")
    return _qr_state(flow_id, entry)


async def cancel_qr_connect(body: TelegramQrCancelRequest) -> dict:
    """Closing the dialog: stop waiting for a scan and let the client go. A
    flow that already finished or was never there is fine to cancel."""
    _drop_pending_login(body.flow_id)
    return {"status": "cancelled", "flow_id": body.flow_id}


async def list_connections():
    conn = db.get_conn("telegram")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM telegram_connections ORDER BY created_at DESC",
            )
            rows = cur.fetchall() or []
            connections = [_as_connection_source(cur, r) for r in rows]
        return TelegramConnectionsResponse(
            connections=connections, count=len(connections), app_configured=config.telegram_app_configured,
        )
    except Exception as exc:
        logger.error("telegram: list connections failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list Telegram connections")
    finally:
        conn.close()


async def list_dialogs(connection_id: str):
    _require_telegram_enabled()
    row = _get_connection_row(connection_id)
    if not row:
        raise HTTPException(status_code=404, detail="Telegram connection not found")

    client = await _connect_with_saved_session(row)
    try:
        # No limit: the picker lists every chat, group and channel and the client
        # searches them, so an older chat can still be found (MS-912).
        dialogs = await client.get_dialogs()
        result = [
            TelegramDialog(
                dialog_id=str(d.id),
                title=d.name or str(d.id),
                type=_dialog_type(d.entity),
                participants_count=getattr(d.entity, "participants_count", None),
            )
            for d in dialogs
        ]
    finally:
        await client.disconnect()

    return TelegramDialogsResponse(dialogs=result, count=len(result))


async def sync_connection(connection_id: str, body: TelegramSyncRequest, user: UserRecord):
    _require_telegram_enabled()
    row = _get_connection_row(connection_id)
    if not row:
        raise HTTPException(status_code=404, detail="Telegram connection not found")
    if not body.dialog_ids:
        raise HTTPException(status_code=400, detail="Select at least one chat to sync")
    limits, workspace_id = resolve_storage_limits(user)

    client = await _connect_with_saved_session(row)
    results: List[TelegramSyncResult] = []

    try:
        for dialog_id_str in body.dialog_ids:
            title = dialog_id_str  # until Telegram gives the name, so a failure still says which chat
            try:
                dialog_id = int(dialog_id_str)
                entity = await client.get_entity(dialog_id)
                title = getattr(entity, "title", None) or getattr(entity, "first_name", None) or dialog_id_str

                sender_names: Dict[int, str] = {}
                try:
                    me = await client.get_me()
                except Exception:
                    me = None  # Sender IDs still preserve attribution if profile lookup fails.
                if me:
                    own_name = " ".join(filter(None, [getattr(me, "first_name", None), getattr(me, "last_name", None)]))
                    sender_names[me.id] = own_name or (getattr(me, "username", None) or str(me.id))
                if _dialog_type(entity) == "user":
                    peer_name = " ".join(filter(None, [getattr(entity, "first_name", None), getattr(entity, "last_name", None)]))
                    sender_names[entity.id] = peer_name or (getattr(entity, "username", None) or str(entity.id))
                try:
                    participants = await client.get_participants(entity, limit=200)
                    for p in participants:
                        name = " ".join(filter(None, [getattr(p, "first_name", None), getattr(p, "last_name", None)]))
                        sender_names[p.id] = name or (getattr(p, "username", None) or str(p.id))
                except Exception:
                    pass  # channels/broadcasts often can't list participants — fall back to sender id below

                saved = _get_selected_chat(connection_id, dialog_id_str)
                history_range = body.ranges.get(dialog_id_str) or (saved["history_range"] if saved else "all")
                cutoff = _range_cutoff(history_range, _utcnow())

                messages_by_id: Dict[int, ChatMessage] = {}
                # Messages come newest first, so the first one older than the range ends
                # the fetch. The collection is rebuilt from what was fetched, which is how
                # a re-sync drops messages that have fallen out of the range.
                fetched_count = outgoing_count = 0
                async for msg in client.iter_messages(entity, limit=None):
                    sent_at = msg.date.replace(tzinfo=None) if msg.date else None
                    if cutoff and sent_at and sent_at < cutoff:
                        break
                    fetched_count += 1
                    outgoing_count += bool(getattr(msg, "out", False))
                    text = getattr(msg, "text", None) or getattr(msg, "message", None)
                    if not text:
                        media = getattr(msg, "media", None)
                        if not media:
                            continue  # Service events have no searchable content.
                        media_type = type(media).__name__.removeprefix("MessageMedia")
                        text = f"[Media: {media_type}]"
                    sender_id = getattr(msg, "sender_id", None)
                    sender_name = sender_names.get(sender_id, str(sender_id) if sender_id else title)
                    messages_by_id[msg.id] = ChatMessage(
                        message_id=f"tg_{dialog_id}_{msg.id}",
                        sender=sender_name,
                        timestamp=sent_at or datetime.now(),
                        content=text,
                        platform=ChatPlatform.TELEGRAM,
                    )
                messages = [
                    message for _, message in sorted(
                        messages_by_id.items(), key=lambda item: (item[1].timestamp, item[0])
                    )
                ]
                logger.info(
                    "telegram: dialog %s fetched %d messages (%d outgoing, %d incoming), indexed %d",
                    dialog_id_str, fetched_count, outgoing_count, fetched_count - outgoing_count, len(messages),
                )

                if not messages:
                    # Nothing is written, so a chat synced before keeps what it has.
                    empty = (
                        "No messages with text or media found in this chat"
                        if cutoff is None
                        else f"No messages with text or media in the last {_HISTORY_RANGE_WORDS[history_range]}"
                    )
                    results.append(
                        TelegramSyncResult(
                            dialog_id=dialog_id_str, title=title, chat_collection_id="",
                            message_count=0, status="error", error=empty,
                        )
                    )
                    continue

                collection_id = (saved and saved["chat_collection_id"]) or str(uuid.uuid4())
                file_name = _telegram_transcript_name(title, dialog_id)
                os.makedirs(config.chat_upload_folder, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=config.chat_upload_folder, prefix=".telegram-sync-", delete=False
                ) as staged:
                    staged_path = staged.name
                try:
                    _write_telegram_transcript(staged_path, messages)

                    # The stored transcript counts toward the workspace quota like an
                    # uploaded chat file (MS-504), checked before the slow indexing
                    # starts so a refused sync leaves what the chat had untouched.
                    transcript_bytes = os.path.getsize(staged_path)
                    if transcript_bytes > limits.max_file_bytes:
                        raise storage_limits.file_too_large_error(limits)
                    # A re-sync replaces the old size, which the quota already counts,
                    # so only what the chat grows by needs room. A sync that does not grow
                    # it needs none, so a workspace over its limit can still shrink a chat.
                    growth = max(0, transcript_bytes - supabase_storage.get_chat_collection_size(collection_id))
                    reservation = (
                        storage_limits.reserve_workspace(workspace_id, limits, growth)
                        if growth
                        else storage_limits.Reservation(workspace_id, None, 0)
                    )
                    try:
                        collection = await ingest_chat_messages(
                            collection_id,
                            messages,
                            file_name=file_name,
                            platform=ChatPlatform.TELEGRAM,
                            raw_file_path=staged_path,
                            owner_id=user.user_id,
                            size_bytes=transcript_bytes,
                            replace_size=True,
                        )
                    finally:
                        reservation.release()
                    upload_dir = os.path.join(config.chat_upload_folder, collection_id)
                    os.makedirs(upload_dir, exist_ok=True)
                    os.replace(staged_path, os.path.join(upload_dir, file_name))
                    for old_name in os.listdir(upload_dir):
                        if old_name != file_name and old_name.startswith("telegram-") and old_name.endswith(".txt"):
                            os.remove(os.path.join(upload_dir, old_name))
                finally:
                    if os.path.exists(staged_path):
                        os.remove(staged_path)

                # Rebuilding a collection under the same ID must also replace
                # its in-memory search index, otherwise queries see stale text.
                from services.rag.processor import processor
                with processor._cache_lock:
                    processor.vector_store_cache.pop(f"chat_{collection_id}", None)

                _upsert_selected_chat(
                    connection_id, dialog_id_str, title, _dialog_type(entity), collection_id,
                    collection.message_count, history_range,
                )

                results.append(
                    TelegramSyncResult(
                        dialog_id=dialog_id_str, title=title, chat_collection_id=collection_id,
                        message_count=collection.message_count, status="success",
                    )
                )
            except Exception as exc:
                logger.error("telegram: sync failed for dialog %s: %s", dialog_id_str, exc)
                results.append(
                    TelegramSyncResult(
                        dialog_id=dialog_id_str, title=title, chat_collection_id="",
                        message_count=0, status="error",
                        # A refusal (storage full, file too large) has a message made for the user.
                        error=exc.detail if isinstance(exc, HTTPException) else str(exc),
                    )
                )
    finally:
        await client.disconnect()

    _touch_connection_synced(connection_id)
    return TelegramSyncResponse(results=results)


async def set_connection_active(body: SetTelegramConnectionActiveRequest):
    conn = db.get_conn("telegram")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE telegram_connections SET status = %s WHERE connection_id = %s RETURNING connection_id",
                ("active" if body.active else "inactive", body.connection_id),
            )
            updated = cur.fetchone()
        if not updated:
            raise HTTPException(status_code=404, detail="Telegram connection not found")
        return {"status": "success", "connection_id": body.connection_id, "active": body.active}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("telegram: activate failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update active status")
    finally:
        conn.close()


async def delete_connection(connection_id: str):
    """Disconnecting leaves nothing DocuLens took from the account behind: the
    session is logged out of Telegram, every chat it synced is deleted, then
    the connection (its selected chats go with it, ON DELETE CASCADE).
    Telegram being unreachable does not stop the rest; a synced chat that
    cannot be deleted does, so it is never left searchable without a way to
    retry."""
    row = _get_connection_row(connection_id)
    if not row:
        raise HTTPException(status_code=404, detail="Telegram connection not found")

    await _log_out_saved_session(row)
    failed = await _delete_synced_chats(connection_id)
    if failed:
        raise HTTPException(
            status_code=500,
            detail=f"Could not delete {failed} synced chat(s), so the connection was kept. Try again.",
        )

    conn = db.get_conn("telegram")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM telegram_connections WHERE connection_id = %s RETURNING connection_id",
                (connection_id,),
            )
            deleted = cur.fetchone()
        if not deleted:
            raise HTTPException(status_code=404, detail="Telegram connection not found")
        return {"status": "success", "message": "Telegram connection deleted", "connection_id": connection_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("telegram: delete failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to delete Telegram connection")
    finally:
        conn.close()


async def _log_out_saved_session(row: Dict[str, Any]) -> None:
    """Remove DocuLens from the account's Telegram > Settings > Devices. Best
    effort: an unreachable Telegram or an already revoked session is logged."""
    connection_id = row["connection_id"]
    try:
        client = await _connect_with_saved_session(row)
    except Exception as exc:
        logger.warning("telegram: could not reach %s to log it out: %s", connection_id, getattr(exc, "detail", exc))
        return
    try:
        await client.log_out()
    except Exception as exc:
        logger.warning("telegram: log out of %s failed: %s", connection_id, exc)
        try:
            await client.disconnect()
        except Exception:
            pass


async def _delete_chat_data(collection_id: str) -> bool:
    """Delete everything stored for one synced chat: the transcript and search
    index in Storage, its collections row, the local copies and the in-memory
    index. True once it is gone; one that was already gone (404) counts as
    deleted. Removing a single chat and disconnecting both go through here."""
    try:
        await chat_service.delete_chat_collection(collection_id)
    except HTTPException as exc:
        if exc.status_code != 404:
            logger.error("telegram: deleting chat %s failed: %s", collection_id, exc.detail)
            return False
    except Exception as exc:
        logger.error("telegram: deleting chat %s failed: %s", collection_id, exc)
        return False
    return True


async def set_chat_range(connection_id: str, dialog_id: str, body: TelegramChatRangeRequest) -> TelegramSelectedChat:
    """Change how far back a chat is kept. This only saves the setting: the stored
    messages are rebuilt by the next sync, which is also what lets the user
    choose when older messages are dropped."""
    conn = db.get_conn("telegram")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE telegram_selected_chats SET history_range = %s
                WHERE connection_id = %s AND dialog_id = %s
                RETURNING {_SELECTED_CHAT_COLUMNS}
                """,
                (body.history_range, connection_id, dialog_id),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Telegram chat not found")
        return _as_selected_chat(row)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("telegram: changing the range of chat %s failed: %s", dialog_id, exc)
        raise HTTPException(status_code=500, detail="Failed to change the history range")
    finally:
        conn.close()


async def remove_chat(connection_id: str, dialog_id: str) -> TelegramChatRemovedResponse:
    """Remove one chat from a connection, with all its stored data. The data goes
    first and the row last: a chat whose data could not be deleted keeps its
    row, so it is never left searchable without a way to try again. Nothing is
    sent to Telegram, so the chat there is not affected."""
    chat = _get_selected_chat(connection_id, dialog_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Telegram chat not found")
    collection_id = chat["chat_collection_id"]
    if collection_id and not await _delete_chat_data(collection_id):
        raise HTTPException(
            status_code=500,
            detail="Could not delete the chat's stored data, so the chat was kept. Try again.",
        )

    conn = db.get_conn("telegram")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM telegram_selected_chats WHERE connection_id = %s AND dialog_id = %s",
                (connection_id, dialog_id),
            )
        return TelegramChatRemovedResponse(
            status="success", message="Telegram chat removed", connection_id=connection_id, dialog_id=dialog_id,
        )
    except Exception as exc:
        logger.error("telegram: removing chat %s failed: %s", dialog_id, exc)
        raise HTTPException(status_code=500, detail="Failed to remove Telegram chat")
    finally:
        conn.close()


async def _delete_synced_chats(connection_id: str) -> int:
    """Delete every chat collection this connection synced; returns how many
    could not be deleted."""
    conn = db.get_conn("telegram")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chat_collection_id FROM telegram_selected_chats
                WHERE connection_id = %s AND chat_collection_id IS NOT NULL AND chat_collection_id <> ''
                """,
                (connection_id,),
            )
            collection_ids = [r["chat_collection_id"] for r in cur.fetchall() or []]
    finally:
        conn.close()

    failed = 0
    for collection_id in collection_ids:
        if not await _delete_chat_data(collection_id):
            failed += 1
    return failed
