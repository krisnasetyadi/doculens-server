# services/chat.py
"""Chat collections (admin-only): uploading a chat export and indexing it
(through services/ingest/chat_ingest.py, shared with Telegram sync), listing, previewing,
paging messages, activating, moving to a folder and deleting.
router/chat.py calls these."""

import asyncio
import logging
import os
import uuid
from typing import List, Optional

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse

from services.storage import supabase as supabase_storage
from services.storage import limits as storage_limits
from services.ingest.chat_ingest import ingest_chat_messages
from services.ingest.chat_parser import ChatParser
from config import config
from models import ChatPlatform, ChatUploadResponse, MoveToFolderRequest, SetChatCollectionActiveRequest, UserRecord
from services.rag.processor import processor
from services.billing.plans import resolve_storage_limits
from services.ingest.text_preview import read_text_source_page
from services.ingest.progress import ProgressCallback, progress_response, upload_progress_store

logger = logging.getLogger(__name__)


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


async def get_upload_status(upload_id: str, user: UserRecord):
    record = upload_progress_store.get(upload_id, user.user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Upload not found")
    return JSONResponse(record, headers={"Cache-Control": "no-store"})


async def upload_chat(file: UploadFile, platform: str, user: UserRecord, stream_progress: bool):
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

    # MS-504: refuse an oversized file or a full workspace before writing
    # anything, then hold the space until the collection is registered.
    limits, workspace_id = resolve_storage_limits(user)
    file_size = storage_limits.measure(file.file)
    if file_size > limits.max_file_bytes:
        raise storage_limits.file_too_large_error(limits)
    reservation = storage_limits.reserve_workspace(workspace_id, limits, file_size)

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
        storage_limits.save_capped(file.file, file_path, limits.max_file_bytes, limits)

        logger.info(f"💾 Saved chat file to: {file_path}")
    except HTTPException:
        reservation.release()
        raise
    except Exception as e:
        reservation.release()
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
            # path (services/ingest/chat_ingest.py) so both stay searchable through identical code.
            collection = asyncio.run(ingest_chat_messages(
                collection_id,
                messages,
                file_name=file.filename,
                platform=ChatPlatform(platform.lower()),
                raw_file_path=file_path,
                on_progress=report,
                owner_id=user.user_id,
                size_bytes=file_size,
            ))
            report("saving", 97)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Chat upload failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process chat file: {str(e)}")
        finally:
            reservation.release()

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


async def list_chat_collections():
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
                with open(metadata_path, encoding='utf-8') as f:
                    metadata = json.load(f)
                    collections.append(metadata)
            else:
                collections.append({"collection_id": collection_id, "status": "no_metadata"})
    
    return {"collections": collections, "count": len(collections)}


async def preview(collection_id: str, max_chars: int):
    file_path, _collection_info = _resolve_chat_collection_file(collection_id)

    try:
        with open(file_path, encoding='utf-8', errors='replace') as handle:
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


async def get_messages(collection_id: str, offset: int, limit: int):
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


async def set_active(body: SetChatCollectionActiveRequest):
    if not supabase_storage.is_enabled() and not supabase_storage.has_database():
        raise HTTPException(status_code=503, detail="Database unavailable")
    updated = supabase_storage.set_chat_collection_status(body.collection_id, body.active)
    if not updated:
        raise HTTPException(status_code=404, detail="Chat collection not found")
    return {"status": "success", "collection_id": body.collection_id, "active": body.active}


async def move_to_folder(body: MoveToFolderRequest):
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


async def delete_chat_collection(collection_id: str):
    import shutil
    
    # Remove from cache if exists
    cache_key = f"chat_{collection_id}"
    if cache_key in processor.vector_store_cache:
        del processor.vector_store_cache[cache_key]

    deleted = False

    # Delete from Supabase (DB row + S3 objects)
    if supabase_storage.is_enabled() or supabase_storage.has_database():
        try:
            deleted = supabase_storage.delete_chat_collection_from_db(collection_id)
        except Exception as e:
            logger.warning(f"Supabase delete failed for {collection_id}: {e}")

    # Delete local upload folder
    upload_path = os.path.join(config.chat_upload_folder, collection_id)
    if os.path.exists(upload_path):
        shutil.rmtree(upload_path)
        deleted = True

    # Delete local index folder
    index_path = os.path.join(config.chat_index_folder, collection_id)
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
        deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail="Chat collection not found")
    return {"status": "deleted", "collection_id": collection_id}
