# router/chat.py
"""
Chat logs upload and search endpoints
Handles WhatsApp TXT file uploads and indexing to FAISS
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Depends
from fastapi.responses import JSONResponse
import asyncio
import logging
import os
import uuid
import shutil
from typing import Optional, List

import storage as supabase_storage
from config import config
from models import ChatUploadResponse, ChatPlatform, SetChatCollectionActiveRequest, MoveToFolderRequest
from chat_parser import ChatParser
from text_source_preview import read_text_source_page
from chat_ingest import ingest_chat_messages
from processor import processor
from router.auth import require_role, UserRecord
from upload_progress import ProgressCallback, progress_response, upload_progress_store

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get('/chat-collections/uploads/{upload_id}')
async def get_chat_upload_status(upload_id: str, user: UserRecord = Depends(require_role("admin"))):
    record = upload_progress_store.get(upload_id, user.user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Upload not found")
    return JSONResponse(record, headers={"Cache-Control": "no-store"})


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
    logger.info(f"📱 Chat upload received: {file.filename}, platform: {platform}")

    # Validate platform
    if platform.lower() not in config.supported_chat_platforms:
        raise HTTPException(
            status_code=400,
            detail=f"Platform '{platform}' not supported. Supported: {config.supported_chat_platforms}"
        )

    # Validate file extension
    if platform.lower() == "whatsapp" and not file.filename.lower().endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="WhatsApp exports should be .txt files"
        )

    try:
        # Generate collection ID
        collection_id = str(uuid.uuid4())

        # Create directories
        upload_dir = os.path.join(config.chat_upload_folder, collection_id)
        index_dir = os.path.join(config.chat_index_folder, collection_id)
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)

        # Save uploaded file
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        logger.info(f"💾 Saved chat file to: {file_path}")
    except Exception as e:
        logger.error(f"❌ Chat upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process chat file: {str(e)}")

    def prepare(report: ProgressCallback) -> dict:
        try:
            report("reading", 10)
            # Parse chat file
            parser = ChatParser()
            messages, metadata = parser.parse_whatsapp(file_path)

            if not messages:
                raise HTTPException(
                    status_code=400,
                    detail="No messages found in chat file. Please check the file format."
                )
            report("reading", 30)

            # Chunk, embed, index, and register — shared with the Telegram sync
            # path (chat_ingest.py) so both stay searchable through identical code.
            collection = asyncio.run(ingest_chat_messages(
                collection_id,
                messages,
                file_name=file.filename,
                platform=ChatPlatform(platform.lower()),
                raw_file_path=file_path,
                on_progress=report,
            ))
            report("saving", 97)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Chat upload failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process chat file: {str(e)}")

        logger.info(f"✅ Chat collection created: {collection_id} with {len(messages)} messages")
        return {
            "collection_id": collection_id,
            "file_name": file.filename,
            "platform": platform,
            "message_count": collection.message_count,
            "participants": collection.participants,
            "date_range": collection.date_range,
            "status": "success",
        }

    if stream_progress:
        upload_progress_store.start(collection_id, owner_id=user.user_id)
        return progress_response(prepare, upload_id=collection_id)

    return ChatUploadResponse(**await asyncio.to_thread(prepare, lambda stage, progress: None))


@router.get('/chat-collections')
async def list_chat_collections(_: UserRecord = Depends(require_role("admin"))):
    """List all available chat collections"""
    # Query Supabase DB first
    if supabase_storage.has_database():
        try:
            rows = supabase_storage.list_chat_collections()
            if rows:
                return {"collections": rows, "count": len(rows)}
        except Exception as e:
            logger.warning(f"Supabase list_chat_collections failed, using disk: {e}")

    # Disk fallback
    collections = []
    chat_index_folder = config.chat_index_folder
    if not os.path.exists(chat_index_folder):
        return {"collections": []}
    
    for collection_id in os.listdir(chat_index_folder):
        collection_path = os.path.join(chat_index_folder, collection_id)
        if os.path.isdir(collection_path):
            metadata_path = os.path.join(collection_path, "metadata.json")
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    collections.append(metadata)
            else:
                collections.append({"collection_id": collection_id, "status": "no_metadata"})
    
    return {"collections": collections, "count": len(collections)}


def _resolve_chat_collection_file(collection_id: str) -> tuple[str, Optional[dict]]:
    """Locate the raw uploaded file for a chat collection, restoring it from
    the Supabase bucket first if the local disk copy is missing (e.g. an
    ephemeral HF Space restart). Shared by /preview and /messages so both
    read the exact same file.
    """
    collection_info = None
    if supabase_storage.has_database():
        try:
            collection_info = supabase_storage.get_chat_collection(collection_id)
        except Exception as exc:
            logger.warning(f"Chat collection metadata lookup failed for {collection_id}: {exc}")

    upload_dir = os.path.join(config.chat_upload_folder, collection_id)

    preferred_name = None
    if isinstance(collection_info, dict):
        preferred_name = collection_info.get("file_name")

    if (not os.path.isdir(upload_dir) or not os.listdir(upload_dir)) and preferred_name:
        try:
            if supabase_storage.is_enabled():
                supabase_storage.download_chat_file(collection_id, preferred_name, upload_dir)
        except Exception as exc:
            logger.warning(f"Chat file restore from bucket failed for {collection_id}: {exc}")

    if not os.path.isdir(upload_dir):
        raise HTTPException(status_code=404, detail="Chat collection file not found")

    candidate_paths: List[str] = []
    if preferred_name:
        candidate_paths.append(os.path.join(upload_dir, preferred_name))

    for entry in sorted(os.listdir(upload_dir)):
        full_path = os.path.join(upload_dir, entry)
        if os.path.isfile(full_path):
            candidate_paths.append(full_path)

    file_path = next((path for path in candidate_paths if os.path.isfile(path)), None)
    if not file_path:
        raise HTTPException(status_code=404, detail="No chat file available for preview")

    return file_path, collection_info


@router.get('/chat-collections/{collection_id}/preview')
async def preview_chat_collection(
    collection_id: str,
    max_chars: int = Query(default=20000, ge=500, le=200000),
    _: UserRecord = Depends(require_role("admin")),
):
    """Return plain-text preview content from an uploaded chat collection file."""
    file_path, _collection_info = _resolve_chat_collection_file(collection_id)

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as handle:
            content = handle.read(max_chars + 1)
    except Exception as exc:
        logger.error(f"Failed reading chat file preview for {collection_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to read chat file")

    truncated = len(content) > max_chars
    preview_text = content[:max_chars] if truncated else content

    return {
        "collection_id": collection_id,
        "file_name": os.path.basename(file_path),
        "content_preview": preview_text,
        "truncated": truncated,
        "max_chars": max_chars,
    }


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
    file_path, collection_info = _resolve_chat_collection_file(collection_id)
    platform = (collection_info or {}).get("platform")

    try:
        page = read_text_source_page(file_path, platform, offset, limit)
    except Exception as exc:
        logger.error(f"Failed reading chat file for {collection_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to read chat file")

    return {
        "collection_id": collection_id,
        "file_name": os.path.basename(file_path),
        "offset": offset,
        "limit": limit,
        **page,
    }


@router.post('/chat-collections/activate')
async def set_chat_collection_active(
    body: SetChatCollectionActiveRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    """Toggle a chat collection's active status (used as a knowledge source)."""
    if not supabase_storage.is_enabled() and not supabase_storage.has_database():
        raise HTTPException(status_code=503, detail="Database unavailable")
    updated = supabase_storage.set_chat_collection_status(body.collection_id, body.active)
    if not updated:
        raise HTTPException(status_code=404, detail="Chat collection not found")
    return {"status": "success", "collection_id": body.collection_id, "active": body.active}


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
    row = supabase_storage.get_chat_collection(body.collection_id)
    if not row:
        raise HTTPException(status_code=404, detail="Chat collection not found")

    if body.folder_id:
        folder = supabase_storage.get_folder(body.folder_id)
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")

    ok = supabase_storage.set_collection_folder(body.collection_id, body.folder_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to move file")
    return {"status": "success", "collection_id": body.collection_id, "folder_id": body.folder_id}


@router.delete('/chat-collections/{collection_id}')
async def delete_chat_collection(
    collection_id: str,
    _: UserRecord = Depends(require_role("admin")),
):
    """Delete a chat collection"""
    import shutil
    
    # Remove from cache if exists
    cache_key = f"chat_{collection_id}"
    if cache_key in processor.vector_store_cache:
        del processor.vector_store_cache[cache_key]

    # Delete from Supabase (DB row + S3 objects)
    if supabase_storage.is_enabled() or supabase_storage.has_database():
        try:
            supabase_storage.delete_chat_collection_from_db(collection_id)
        except Exception as e:
            logger.warning(f"Supabase delete failed for {collection_id}: {e}")

    # Delete local upload folder
    upload_path = os.path.join(config.chat_upload_folder, collection_id)
    if os.path.exists(upload_path):
        shutil.rmtree(upload_path)

    # Delete local index folder
    index_path = os.path.join(config.chat_index_folder, collection_id)
    if os.path.exists(index_path):
        shutil.rmtree(index_path)

    return {"status": "deleted", "collection_id": collection_id}
