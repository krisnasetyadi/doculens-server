# router/chat.py
"""Chat collection endpoints (admin-only, and each user only ever reaches their
own chats, MS-913). Each route reads the request and calls services/chat.py."""

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from models import (
    ChatCollectionsResponse,
    ChatMessagesResponse,
    ChatPreviewResponse,
    ChatUploadResponse,
    CollectionActiveResponse,
    CollectionDeletedResponse,
    CollectionFolderResponse,
    MoveToFolderRequest,
    SetChatCollectionActiveRequest,
    UserRecord,
)
from services import chat as chat_service
from services.auth import require_role

router = APIRouter()


@router.get('/chat-collections/uploads/{upload_id}')
async def get_chat_upload_status(upload_id: str, user: UserRecord = Depends(require_role("admin"))):
    return await chat_service.get_upload_status(upload_id, user)


@router.post('/chat-collections/upload', response_model=ChatUploadResponse,
             responses={200: {"content": {"application/x-ndjson": {}}}})
async def upload_chat(
    file: UploadFile = File(...),
    platform: str = Form(default="whatsapp"),
    user: UserRecord = Depends(require_role("admin")),
    stream_progress: bool = Query(False),
):
    """
    Upload and process a chat export file

    - **file**: Chat export file (TXT for WhatsApp)
    - **platform**: Chat platform (whatsapp, teams, slack). Default: whatsapp
    """
    return await chat_service.upload_chat(file, platform, user, stream_progress)


@router.get('/chat-collections', response_model=ChatCollectionsResponse)
async def list_chat_collections(user: UserRecord = Depends(require_role("admin"))):
    """List the logged-in user's own chat collections"""
    return await chat_service.list_chat_collections(user)


@router.get('/chat-collections/{collection_id}/preview', response_model=ChatPreviewResponse)
async def preview_chat_collection(
    collection_id: str,
    max_chars: int = Query(default=20000, ge=500, le=200000),
    user: UserRecord = Depends(require_role("admin")),
):
    """Return plain-text preview content from an uploaded chat collection file."""
    return await chat_service.preview(collection_id, max_chars, user)


@router.get('/chat-collections/{collection_id}/messages', response_model=ChatMessagesResponse)
async def get_chat_collection_messages(
    collection_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    user: UserRecord = Depends(require_role("admin")),
):
    """Preview recognized WhatsApp messages, or raw text lines for other sources.
    Both formats are read-only and paginated (MS-415).
    """
    return await chat_service.get_messages(collection_id, offset, limit, user)


@router.post('/chat-collections/activate', response_model=CollectionActiveResponse)
async def set_chat_collection_active(
    body: SetChatCollectionActiveRequest,
    user: UserRecord = Depends(require_role("admin")),
):
    """Toggle a chat collection's active status (used as a knowledge source)."""
    return await chat_service.set_active(body, user)


@router.post('/chat-collections/move-to-folder', response_model=CollectionFolderResponse)
async def move_chat_collection_to_folder(
    body: MoveToFolderRequest,
    user: UserRecord = Depends(require_role("admin")),
):
    """Assign (or unassign) one of the user's own chat collections to one of
    their own folders (MS-274, MS-913)."""
    return await chat_service.move_to_folder(body, user)


@router.delete('/chat-collections/{collection_id}', response_model=CollectionDeletedResponse)
async def delete_chat_collection(
    collection_id: str,
    user: UserRecord = Depends(require_role("admin")),
):
    """Delete a chat collection"""
    return await chat_service.delete_chat_collection(collection_id, user)
