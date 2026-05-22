# router/collections.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from models import CollectionInfo
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
                    ))

        logger.info("Listed %d collections from local disk", len(collections))
        return collections

    except Exception as e:
        logger.error("Failed to list collections: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


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


@router.get("/collections", response_model=List[CollectionInfo])
async def list_collections():
    """List available PDF document collections"""
    try:
        collections = []
        
        # Check if index folder exists
        if not os.path.exists(config.index_folder):
            logger.warning(f"Index folder not found: {config.index_folder}")
            return collections
            
        for entry in os.listdir(config.index_folder):
            entry_path = os.path.join(config.index_folder, entry)
            if os.path.isdir(entry_path):
                index_file = os.path.join(entry_path, "index.faiss")
                if os.path.exists(index_file):
                    index_mtime = os.path.getmtime(index_file)
                    created_at = datetime.fromtimestamp(index_mtime)
                    
                    # Get file names from uploads folder
                    upload_path = os.path.join(config.upload_folder, entry)
                    file_names = []
                    if os.path.exists(upload_path):
                        file_names = [f for f in os.listdir(upload_path) if f.endswith('.pdf')]
                    
                    # Try to get more info from vector store, but don't fail if it doesn't work
                    source_files = set()
                    try:
                        vector_store = processor.get_vector_store(entry)
                        if vector_store and hasattr(vector_store, 'docstore'):
                            for doc_id in vector_store.docstore._dict:
                                doc = vector_store.docstore._dict[doc_id]
                                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                                    source_files.add(os.path.basename(doc.metadata['source']))
                    except Exception as e:
                        logger.warning(f"Could not load vector store for {entry}: {str(e)}")
                    
                    # Use file_names from uploads if source_files is empty
                    final_file_names = list(source_files) if source_files else file_names
                    
                    collections.append(CollectionInfo(
                        collection_id=entry,
                        document_count=len(final_file_names) if final_file_names else 1,
                        created_at=created_at.isoformat(),
                        file_names=final_file_names
                    ))
        
        logger.info(f"Found {len(collections)} collections")
        return collections
    
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/collection/{collection_id}")
async def delete_collection(collection_id: str):
    """Delete a document collection"""
    try:
        index_path = os.path.join(config.index_folder, collection_id)
        if not os.path.exists(index_path):
            raise HTTPException(status_code=404, detail="Collection not found")
        
        # Delete index files
        index_files = [
            os.path.join(index_path, "index.faiss"),
            os.path.join(index_path, "index.pkl")
        ]
        
        deleted = False
        for f in index_files:
            if os.path.exists(f):
                os.remove(f)
                deleted = True
        
        # Delete upload folder
        upload_dir = os.path.join(config.upload_folder, collection_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
            deleted = True
        
        # Delete index folder
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
            deleted = True
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        # Invalidate cache
        processor.invalidate_vector_store_cache(collection_id)
        
        return {"status": "success", "message": "Collection deleted"}
    
    except Exception as e:
        logger.error(f"Failed to delete collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{collection_id}/{file_name:path}")
async def serve_pdf_file(collection_id: str, file_name: str):
    """
    Serve a PDF file from a collection.
    
    Use with #page=N fragment to jump to specific page in browser's PDF viewer.
    Example: /api/v1/files/abc123/document.pdf#page=5
    """
    try:
        # Decode URL-encoded filename
        decoded_file_name = urllib.parse.unquote(file_name)
        
        # Security: prevent path traversal
        if '..' in decoded_file_name or decoded_file_name.startswith('/'):
            logger.warning(f"🚨 Path traversal attempt: {decoded_file_name}")
            raise HTTPException(status_code=400, detail="Invalid file name")
        
        # Build file path - use absolute path for better compatibility
        upload_folder_abs = os.path.abspath(config.upload_folder)
        file_path = os.path.join(upload_folder_abs, collection_id, decoded_file_name)
        
        # Enhanced logging for debugging
        logger.info(f"🔍 PDF request - Collection: {collection_id}, File: {decoded_file_name}")
        logger.info(f"🔍 CWD: {os.getcwd()}")
        logger.info(f"🔍 Upload folder (config): {config.upload_folder}")
        logger.info(f"🔍 Upload folder (absolute): {upload_folder_abs}")
        logger.info(f"🔍 Looking for file at: {file_path}")
        logger.info(f"🔍 Upload folder exists: {os.path.exists(upload_folder_abs)}")
        logger.info(f"🔍 Collection folder exists: {os.path.exists(os.path.join(upload_folder_abs, collection_id))}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            # Enhanced error details for debugging
            collection_path = os.path.join(config.upload_folder, collection_id)
            if not os.path.exists(collection_path):
                logger.error(f"❌ Collection folder not found: {collection_path}")
                raise HTTPException(status_code=404, detail=f"Collection '{collection_id}' not found")
            
            # List available files in collection for debugging
            try:
                available_files = os.listdir(collection_path)
                logger.error(f"❌ File '{decoded_file_name}' not found in collection '{collection_id}'")
                logger.error(f"📁 Available files: {available_files}")
                raise HTTPException(
                    status_code=404, 
                    detail=f"File '{decoded_file_name}' not found in collection '{collection_id}'. Available files: {', '.join(available_files) if available_files else 'none'}"
                )
            except OSError:
                logger.error(f"❌ Cannot access collection folder: {collection_path}")
                raise HTTPException(status_code=404, detail=f"Cannot access collection '{collection_id}'")
        
        # Check if it's a PDF
        if not file_path.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        logger.info(f"✅ Serving PDF: {file_path}")
        
        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            filename=decoded_file_name,
            # Allow browser to display PDF inline instead of downloading
            headers={
                "Content-Disposition": f'inline; filename="{decoded_file_name}"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collection/{collection_id}/files")
async def list_collection_files(collection_id: str):
    """List all PDF files in a collection"""
    try:
        upload_dir = os.path.join(config.upload_folder, collection_id)
        
        if not os.path.exists(upload_dir):
            raise HTTPException(status_code=404, detail="Collection not found")
        
        files = []
        for file_name in os.listdir(upload_dir):
            if file_name.lower().endswith('.pdf'):
                file_path = os.path.join(upload_dir, file_name)
                file_stat = os.stat(file_path)
                files.append({
                    "file_name": file_name,
                    "size_bytes": file_stat.st_size,
                    "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    "url": f"/api/v1/files/{collection_id}/{urllib.parse.quote(file_name)}"
                })
        
        return {
            "collection_id": collection_id,
            "file_count": len(files),
            "files": files
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))