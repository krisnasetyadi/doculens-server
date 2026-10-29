# router/telegram.py
"""Telegram connection endpoints (admin-only). Each route reads the request
and calls services/telegram.py, where the connect flow, Telethon calls,
SQL and chat sync live."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from models import (
    ConnectionActiveResponse,
    TelegramChatRangeRequest,
    TelegramChatRemovedResponse,
    ConnectionDeletedResponse,
    SetTelegramConnectionActiveRequest,
    TelegramConnectStartRequest,
    TelegramConnectStartResponse,
    TelegramConnectVerifyRequest,
    TelegramConnectVerifyResponse,
    TelegramConnectionsResponse,
    TelegramDialogsResponse,
    TelegramQrCancelRequest,
    TelegramQrLoginState,
    TelegramQrStartRequest,
    TelegramSelectedChat,
    TelegramSyncRequest,
    TelegramSyncResponse,
    UserRecord,
)
from services import telegram as telegram_service
from services.auth import require_role

router = APIRouter()


@router.post("/telegram-connections/connect/start", response_model=TelegramConnectStartResponse)
async def telegram_connect_start(
    body: TelegramConnectStartRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    return await telegram_service.start_connect(body)


@router.post("/telegram-connections/connect/verify", response_model=TelegramConnectVerifyResponse)
async def telegram_connect_verify(
    body: TelegramConnectVerifyRequest,
    user: UserRecord = Depends(require_role("admin")),
):
    return await telegram_service.verify_connect(body)


@router.post("/telegram-connections/connect/qr/start", response_model=TelegramQrLoginState)
async def telegram_qr_start(
    body: TelegramQrStartRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    return await telegram_service.start_qr_connect(body)


@router.get("/telegram-connections/connect/qr/status", response_model=TelegramQrLoginState)
async def telegram_qr_status(
    flow_id: str,
    _: UserRecord = Depends(require_role("admin")),
):
    return await telegram_service.qr_connect_status(flow_id)


@router.post("/telegram-connections/connect/qr/cancel")
async def telegram_qr_cancel(
    body: TelegramQrCancelRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    return await telegram_service.cancel_qr_connect(body)


@router.get("/telegram-connections", response_model=TelegramConnectionsResponse)
async def list_telegram_connections(_: UserRecord = Depends(require_role("admin"))):
    return await telegram_service.list_connections()


@router.get("/telegram-connections/{connection_id}/dialogs", response_model=TelegramDialogsResponse)
async def list_telegram_dialogs(
    connection_id: str,
    _: UserRecord = Depends(require_role("admin")),
):
    return await telegram_service.list_dialogs(connection_id)


@router.post("/telegram-connections/{connection_id}/sync", response_model=TelegramSyncResponse)
async def sync_telegram_connection(
    connection_id: str,
    body: TelegramSyncRequest,
    user: UserRecord = Depends(require_role("admin")),
):
    return await telegram_service.sync_connection(connection_id, body, user)


@router.post("/telegram-connections/activate", response_model=ConnectionActiveResponse)
async def set_telegram_connection_active(
    body: SetTelegramConnectionActiveRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    return await telegram_service.set_connection_active(body)


@router.patch(
    "/telegram-connections/{connection_id}/chats/{dialog_id}",
    response_model=TelegramSelectedChat,
)
async def set_telegram_chat_range(
    connection_id: str,
    dialog_id: str,
    body: TelegramChatRangeRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    """Save how far back one synced chat is kept (MS-912). The next sync applies it."""
    return await telegram_service.set_chat_range(connection_id, dialog_id, body)


@router.delete(
    "/telegram-connections/{connection_id}/chats/{dialog_id}",
    response_model=TelegramChatRemovedResponse,
)
async def remove_telegram_chat(
    connection_id: str,
    dialog_id: str,
    _: UserRecord = Depends(require_role("admin")),
):
    """Remove one synced chat and everything stored for it (MS-912), keeping the
    connection. The chat in Telegram is not touched."""
    return await telegram_service.remove_chat(connection_id, dialog_id)


@router.delete("/telegram-connections/{connection_id}", response_model=ConnectionDeletedResponse)
async def delete_telegram_connection(
    connection_id: str,
    _: UserRecord = Depends(require_role("admin")),
):
    """Log the session out of Telegram and delete every chat it synced, then
    the connection itself (MS-911): nothing taken from the account stays
    searchable."""
    return await telegram_service.delete_connection(connection_id)
