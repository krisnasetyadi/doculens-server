# router/collections.py
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse, RedirectResponse
from models import CollectionInfo, SetPdfCollectionActiveRequest, MoveToFolderRequest
from config import config
from processor import processor
from utils import (
    DOCUMENT_EXTRACTORS,
    CONTENT_TYPE_BY_EXT,
    INLINE_VIEWABLE_EXTS,
    content_disposition,
)
import storage as supabase_storage
from router.auth import get_current_user, UserRecord
import os
import shutil
from typing import List, Optional
from datetime import datetime
import logging
import urllib.parse

router = APIRouter()
logger = logging.getLogger(__name__)


def _can_access(row: Optional[dict], user: UserRecord) -> bool:
    """Admins can access everything; everyone else only their own collection."""
    if user.role == "admin":
        return True
    if not row:
        return False
    return row.get("owner_id") == user.user_id


@router.get("/pdf-collections", response_model=List[CollectionInfo])
async def list_collections(user: UserRecord = Depends(get_current_user)):
    """List PDF document collections visible to the current user.

    Admins see every collection (Supabase DB → S3 scan → local disk fallback).
    Non-admins only see DB-backed collections they own — the S3-scan and
    local-disk fallbacks can't be attributed to an owner, so they're
    admin-only.
    """
    try:
        # ── Try Supabase DB first ──────────────────────────────────────────
        # has_database() (DATABASE_URL), not is_enabled() (S3 creds) — this
        # branch only ever does Postgres reads; gating it on S3 config meant
        # a deployment with DATABASE_URL set but no S3 creds would silently
        # skip the DB and fall through to the local-disk/S3-scan branches
        # below, hiding real rows non-admins own (they get `[]` outright).
        if supabase_storage.has_database():
            rows = supabase_storage.list_collections()
            if rows:  # non-empty DB result → use it
                visible_rows = [row for row in rows if _can_access(row, user)]
                result = [
                    CollectionInfo(
                        collection_id=row["collection_id"],
                        document_count=len(row.get("file_names") or []),
                        created_at=row.get("created_at", ""),
                        file_names=row.get("file_names") or [],
                        title=row.get("title") or None,
                        status=row.get("status") or "active",
                        owner_id=row.get("owner_id"),
                        folder_id=row.get("folder_id"),
                    )
                    for row in visible_rows
                ]
                logger.info("Listed %d/%d collections from Supabase DB for user %s",
                            len(result), len(rows), user.user_id)
                return result

            if user.role != "admin":
                return []

            # DB empty → scan S3 to find orphaned collections (admin-only,
            # since these predate ownership tracking and can't be attributed).
            s3_cols = supabase_storage.list_collections_from_s3()
            if s3_cols:
                logger.info("DB empty; found %d collections via S3 scan — auto-registering", len(s3_cols))
                result = []
                for col in s3_cols:
                    cid = col["collection_id"]
                    fnames = col["file_names"]
                    # Auto-register so next call hits DB
                    supabase_storage.register_collection(cid, fnames, len(fnames), owner_id=user.user_id)
                    result.append(CollectionInfo(
                        collection_id=cid,
                        document_count=len(fnames),
                        created_at=col.get("created_at", ""),
                        file_names=fnames,
                        title=col.get("title") or None,
                        owner_id=user.user_id,
                    ))
                return result

        if user.role != "admin":
            return []

        # ── Local disk fallback (admin-only — no owner metadata) ───────────
        collections = []
        if not os.path.exists(config.index_folder):
            return collections

        for entry in os.listdir(config.index_folder):
            entry_path = os.path.join(config.index_folder, entry)
            if os.path.isdir(entry_path):
                index_file = os.path.join(entry_path, "index.faiss")
                if os.path.exists(index_file):
                    created_at = datetime.fromtimestamp(
                        os.path.getmtime(index_file)
                    ).isoformat()
                    upload_path = os.path.join(config.upload_folder, entry)
                    file_names = []
                    if os.path.exists(upload_path):
                        file_names = [
                            f for f in os.listdir(upload_path)
                            if os.path.splitext(f)[1].lower() in DOCUMENT_EXTRACTORS
                        ]
                    collections.append(CollectionInfo(
                        collection_id=entry,
                        document_count=len(file_names) or 1,
                        created_at=created_at,
                        file_names=file_names,
                        title=None,
                    ))

        logger.info("Listed %d collections from local disk", len(collections))
        return collections

    except Exception as e:
        logger.error("Failed to list collections: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf-collections/activate")
async def set_pdf_collection_active(
    body: SetPdfCollectionActiveRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Toggle a PDF collection's active status (used as a knowledge source)."""
    if not supabase_storage.is_enabled() and not supabase_storage.has_database():
        raise HTTPException(status_code=503, detail="Database unavailable")
    row = supabase_storage.get_collection(body.collection_id)
    if not row:
        raise HTTPException(status_code=404, detail="Collection not found")
    if not _can_access(row, user):
        raise HTTPException(status_code=403, detail="Not allowed to modify this collection")
    updated = supabase_storage.set_collection_status(body.collection_id, body.active)
    if not updated:
        raise HTTPException(status_code=404, detail="Collection not found")
    return {"status": "success", "collection_id": body.collection_id, "active": body.active}


@router.post("/pdf-collections/move-to-folder")
async def move_pdf_collection_to_folder(
    body: MoveToFolderRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Assign (or unassign, when folder_id is null) a PDF collection to a
    folder (MS-274). Owner-gated like every other pdf-collections mutation."""
    row = supabase_storage.get_collection(body.collection_id)
    if not row:
        raise HTTPException(status_code=404, detail="Collection not found")
    if not _can_access(row, user):
        raise HTTPException(status_code=403, detail="Not allowed to modify this collection")

    if body.folder_id:
        folder = supabase_storage.get_folder(body.folder_id)
        if not folder or (user.role != "admin" and folder.get("owner_id") != user.user_id):
            raise HTTPException(status_code=404, detail="Folder not found")

    ok = supabase_storage.set_collection_folder(body.collection_id, body.folder_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to move file")
    return {"status": "success", "collection_id": body.collection_id, "folder_id": body.folder_id}


@router.delete("/pdf-collections/{collection_id}")
async def delete_collection(collection_id: str, user: UserRecord = Depends(get_current_user)):
    """Delete a collection from Supabase Storage + DB and local disk."""
    try:
        if supabase_storage.is_enabled() or supabase_storage.has_database():
            row = supabase_storage.get_collection(collection_id)
            if row and not _can_access(row, user):
                raise HTTPException(status_code=403, detail="Not allowed to delete this collection")
            if not row and user.role != "admin":
                # No ownership metadata (e.g. local-disk-only collection) —
                # only admins may delete collections that can't be attributed.
                raise HTTPException(status_code=403, detail="Not allowed to delete this collection")

        deleted = False

        # ── Supabase delete ────────────────────────────────────────────────
        # has_database(), not is_enabled() — delete_collection_from_db is a
        # Postgres DELETE that handles the S3-cleanup step internally on its
        # own (skipping it gracefully when S3 isn't configured), so gating
        # this call on S3 creds meant the DB row could survive a "delete".
        if supabase_storage.has_database():
            ok = supabase_storage.delete_collection_from_db(collection_id)
            if ok:
                deleted = True
                logger.info("Deleted collection from Supabase: %s", collection_id)

        # ── Local disk delete ──────────────────────────────────────────────
        index_path = os.path.join(config.index_folder, collection_id)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
            deleted = True

        upload_dir = os.path.join(config.upload_folder, collection_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
            deleted = True

        if not deleted:
            raise HTTPException(status_code=404, detail="Collection not found")

        processor.invalidate_cache(collection_id)
        return {"status": "success", "message": "Collection deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete collection: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{collection_id}/{file_name:path}")
async def serve_pdf_file(
    collection_id: str,
    file_name: str,
    user: UserRecord = Depends(get_current_user),
):
    """
    Serve an uploaded document (PDF, DOCX, DOC, CSV, XLSX, TXT; the route
    name is kept for backward compatibility with existing callers).
    Priority: Supabase signed URL redirect → local disk FileResponse.

    The response always declares the file's real content type, and marks it
    inline only for the formats a browser can actually render (see
    INLINE_VIEWABLE_EXTS); everything else is sent as a download carrying its
    original filename.
    """
    try:
        decoded_file_name = urllib.parse.unquote(file_name)

        # Security: prevent path traversal
        if ".." in decoded_file_name or decoded_file_name.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid file name")

        row = supabase_storage.get_collection(collection_id)
        if not _can_access(row, user):
            raise HTTPException(status_code=403, detail="Not allowed to access this collection")

        # How this file should reach the browser. Derived up here, before the
        # storage branches, so the signed-URL path and the local-disk path
        # can't drift apart on content type or on inline-vs-download, which
        # they used to, and only the disk path even checked the extension.
        ext = os.path.splitext(decoded_file_name)[1].lower()
        if ext not in DOCUMENT_EXTRACTORS:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        media_type = CONTENT_TYPE_BY_EXT.get(ext, "application/octet-stream")
        disposition_type = "inline" if ext in INLINE_VIEWABLE_EXTS else "attachment"

        # ── Try Supabase signed URL first ──────────────────────────────────
        if supabase_storage.is_enabled():
            signed_url = supabase_storage.get_pdf_signed_url(
                collection_id,
                decoded_file_name,
                content_type=media_type,
                content_disposition=content_disposition(
                    disposition_type, decoded_file_name
                ),
            )
            if signed_url:
                logger.info("Redirecting %s via Supabase signed URL (%s, %s): %s/%s",
                            ext, media_type, disposition_type,
                            collection_id, decoded_file_name)
                return RedirectResponse(url=signed_url)

        # ── Local disk fallback ────────────────────────────────────────────
        upload_folder_abs = os.path.abspath(config.upload_folder)
        file_path = os.path.join(upload_folder_abs, collection_id, decoded_file_name)

        if not os.path.exists(file_path):
            collection_path = os.path.join(config.upload_folder, collection_id)
            if not os.path.exists(collection_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection '{collection_id}' not found"
                )
            try:
                available = os.listdir(collection_path)
            except OSError:
                available = []
            raise HTTPException(
                status_code=404,
                detail=(
                    f"File '{decoded_file_name}' not found. "
                    f"Available: {', '.join(available) or 'none'}"
                ),
            )

        logger.info("Serving file from local disk (%s, %s): %s",
                    media_type, disposition_type, file_path)
        # content_disposition_type rather than a hand-written header: Starlette
        # then RFC 5987-encodes the filename itself. Formatting the header by
        # hand instead threw UnicodeEncodeError (a 500) on any name outside
        # latin-1, since header values go out latin-1 encoded.
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=decoded_file_name,
            content_disposition_type=disposition_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to serve file: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collection/{collection_id}/files")
async def list_collection_files(
    collection_id: str,
    user: UserRecord = Depends(get_current_user),
):
    """List all PDF files in a collection (Supabase DB → local disk fallback)."""
    try:
        db_row = supabase_storage.get_collection(collection_id)
        if not _can_access(db_row, user):
            raise HTTPException(status_code=403, detail="Not allowed to access this collection")

        # ── Try Supabase ───────────────────────────────────────────────────
        # has_database(), not is_enabled() — `row` below is db_row, already
        # fetched from Postgres above; nothing here touches S3.
        if supabase_storage.has_database():
            row = db_row
            if row:
                file_names = row.get("file_names") or []
                files = [
                    {
                        "file_name": fname,
                        "url": f"/api/v1/files/{collection_id}/{urllib.parse.quote(fname)}",
                        "created_at": row.get("created_at", ""),
                    }
                    for fname in file_names
                ]
                return {
                    "collection_id": collection_id,
                    "file_count": len(files),
                    "files": files,
                }

        # ── Local disk fallback ────────────────────────────────────────────
        upload_dir = os.path.join(config.upload_folder, collection_id)
        if not os.path.exists(upload_dir):
            raise HTTPException(status_code=404, detail="Collection not found")

        files = []
        for fname in os.listdir(upload_dir):
            if os.path.splitext(fname)[1].lower() in DOCUMENT_EXTRACTORS:
                fp = os.path.join(upload_dir, fname)
                stat = os.stat(fp)
                files.append({
                    "file_name": fname,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "url": f"/api/v1/files/{collection_id}/{urllib.parse.quote(fname)}",
                })

        return {
            "collection_id": collection_id,
            "file_count": len(files),
            "files": files,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list files: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

