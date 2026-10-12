# router/chat.py
"""Chat collection endpoints (admin-only). Each route reads the request and
calls services/chat.py."""

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from models import ChatUploadResponse, MoveToFolderRequest, SetChatCollectionActiveRequest, UserRecord
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


@router.get('/chat-collections')
async def list_chat_collections(_: UserRecord = Depends(require_role("admin"))):
    """List all available chat collections"""
    return await chat_service.list_chat_collections()


@router.get('/chat-collections/{collection_id}/preview')
async def preview_chat_collection(
    collection_id: str,
    max_chars: int = Query(default=20000, ge=500, le=200000),
    _: UserRecord = Depends(require_role("admin")),
):
    """Return plain-text preview content from an uploaded chat collection file."""
    return await chat_service.preview(collection_id, max_chars)


@router.get('/chat-collections/{collection_id}/messages')
async def get_chat_collection_messages(
    collection_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    _: UserRecord = Depends(require_role("admin")),
):
    """Preview recognized WhatsApp messages, or raw text lines for other sources.
    Both formats are read-only and paginated (MS-415).
    """
    return await chat_service.get_messages(collection_id, offset, limit)


@router.post('/chat-collections/activate')
async def set_chat_collection_active(
    body: SetChatCollectionActiveRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    """Toggle a chat collection's active status (used as a knowledge source)."""
    return await chat_service.set_active(body)


@router.post('/chat-collections/move-to-folder')
async def move_chat_collection_to_folder(
    body: MoveToFolderRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    """Assign (or unassign) a chat/WhatsApp collection to a folder (MS-274).
    Admin-gated, matching every other chat-collections mutation — chat rows
    carry no per-row owner_id (see storage.py). Every caller here is already
    an admin (require_role("admin") above), so — same as the pdf endpoint's
    admin branch — any folder is a valid target, not just ones the caller
    personally created."""
    return await chat_service.move_to_folder(body)


@router.delete('/chat-collections/{collection_id}')
async def delete_chat_collection(
    collection_id: str,
    _: UserRecord = Depends(require_role("admin")),
):
    """Delete a chat collection"""
    return await chat_service.delete_chat_collection(collection_id)
