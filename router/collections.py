# router/collections.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from models import CollectionInfo, SetPdfCollectionActiveRequest
from config import config
from processor import processor
import storage as supabase_storage
import os
import shutil
from typing import List
from datetime import datetime
import logging
import urllib.parse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/collections", response_model=List[CollectionInfo])
async def list_collections():
    """List available PDF document collections (Supabase DB → S3 scan → local disk fallback)."""
    try:
        # ── Try Supabase DB first ──────────────────────────────────────────
        if supabase_storage.is_enabled():
            rows = supabase_storage.list_collections()
            if rows:  # non-empty DB result → use it
                result = [
                    CollectionInfo(
                        collection_id=row["collection_id"],
                        document_count=len(row.get("file_names") or []),
                        created_at=row.get("created_at", ""),
                        file_names=row.get("file_names") or [],
                        title=row.get("title") or None,
                        status=row.get("status") or "active",
                    )
                    for row in rows
                ]
                logger.info("Listed %d collections from Supabase DB", len(result))
                return result

            # DB empty → scan S3 to find orphaned collections (with file names)
            s3_cols = supabase_storage.list_collections_from_s3()
            if s3_cols:
                logger.info("DB empty; found %d collections via S3 scan — auto-registering", len(s3_cols))
                result = []
                for col in s3_cols:
                    cid = col["collection_id"]
                    fnames = col["file_names"]
                    # Auto-register so next call hits DB
                    supabase_storage.register_collection(cid, fnames, len(fnames))
                    result.append(CollectionInfo(
                        collection_id=cid,
                        document_count=len(fnames),
                        created_at=col.get("created_at", ""),
                        file_names=fnames,
                        title=col.get("title") or None,
                    ))
                return result

        # ── Local disk fallback ────────────────────────────────────────────
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
                            if f.lower().endswith(".pdf")
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


@router.post("/pdf-collection/activate")
async def set_pdf_collection_active(body: SetPdfCollectionActiveRequest):
    """Toggle a PDF collection's active status (used as a knowledge source)."""
    if not supabase_storage.is_enabled() and not supabase_storage.has_database():
        raise HTTPException(status_code=503, detail="Database unavailable")
    updated = supabase_storage.set_collection_status(body.collection_id, body.active)
    if not updated:
        raise HTTPException(status_code=404, detail="Collection not found")
    return {"status": "success", "collection_id": body.collection_id, "active": body.active}


@router.delete("/collection/{collection_id}")
async def delete_collection(collection_id: str):
    """Delete a collection from Supabase Storage + DB and local disk."""
    try:
        deleted = False

        # ── Supabase delete ────────────────────────────────────────────────
        if supabase_storage.is_enabled():
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
async def serve_pdf_file(collection_id: str, file_name: str):
    """
    Serve a PDF file.
    Priority: Supabase signed URL redirect → local disk FileResponse.
    """
    try:
        decoded_file_name = urllib.parse.unquote(file_name)

        # Security: prevent path traversal
        if ".." in decoded_file_name or decoded_file_name.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid file name")

        # ── Try Supabase signed URL first ──────────────────────────────────
        if supabase_storage.is_enabled():
            signed_url = supabase_storage.get_pdf_signed_url(
                collection_id, decoded_file_name
            )
            if signed_url:
                logger.info("Redirecting PDF via Supabase signed URL: %s/%s",
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

        if not file_path.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        logger.info("Serving PDF from local disk: %s", file_path)
        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            filename=decoded_file_name,
            headers={"Content-Disposition": f'inline; filename="{decoded_file_name}"'},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to serve file: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collection/{collection_id}/files")
async def list_collection_files(collection_id: str):
    """List all PDF files in a collection (Supabase DB → local disk fallback)."""
    try:
        # ── Try Supabase ───────────────────────────────────────────────────
        if supabase_storage.is_enabled():
            row = supabase_storage.get_collection(collection_id)
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
            if fname.lower().endswith(".pdf"):
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

