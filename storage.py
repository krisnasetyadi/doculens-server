"""
storage.py
----------
Supabase Storage wrapper for DoculensAPI.

Responsibilities:
  - Upload raw PDF files to the 'pdf-uploads' bucket
  - Upload FAISS index files (.faiss + .pkl) to 'pdf-indices' bucket
  - Download index files back to a local temp directory (for FAISS.load_local)
  - Register/delete collection metadata in the pdf_collections table
  - List all collections from the DB (replaces os.listdir on disk)

Graceful degradation:
  If SUPABASE_URL or SUPABASE_SERVICE_KEY env vars are not set, all
  operations silently no-op and the caller falls back to local disk.
  This keeps the server working in local-dev without Supabase credentials.

Bucket layout:
  pdf-uploads/
    {collection_id}/{filename}.pdf
  pdf-indices/
    {collection_id}/index.faiss
    {collection_id}/index.pkl
"""

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_UPLOADS_BUCKET = "pdf-uploads"
_INDICES_BUCKET = "pdf-indices"


def _get_client():
    """Return an authenticated Supabase client, or None if not configured."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")  # service-role key, never anon key
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        logger.warning("supabase package not installed — storage features disabled")
        return None
    except Exception as e:
        logger.warning("Supabase client init failed: %s", e)
        return None


def is_enabled() -> bool:
    """True if Supabase credentials are present."""
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"))


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def upload_pdf(collection_id: str, file_path: str, filename: str) -> Optional[str]:
    """
    Upload a raw PDF file to Supabase Storage.

    Returns the storage path on success, None on failure/disabled.
    """
    client = _get_client()
    if not client:
        return None

    storage_path = f"{collection_id}/{filename}"
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        client.storage.from_(_UPLOADS_BUCKET).upload(
            storage_path,
            data,
            {"content-type": "application/pdf", "upsert": "true"},
        )
        logger.info("Uploaded PDF to Supabase: %s", storage_path)
        return storage_path
    except Exception as e:
        logger.warning("PDF upload to Supabase failed (%s): %s", storage_path, e)
        return None


def upload_index(collection_id: str, index_dir: str) -> bool:
    """
    Upload FAISS index files (index.faiss + index.pkl) to Supabase Storage.

    Returns True on success.
    """
    client = _get_client()
    if not client:
        return False

    success = True
    for fname in ("index.faiss", "index.pkl"):
        local_path = os.path.join(index_dir, fname)
        if not os.path.exists(local_path):
            logger.warning("Index file missing, skipping upload: %s", local_path)
            success = False
            continue
        storage_path = f"{collection_id}/{fname}"
        try:
            with open(local_path, "rb") as f:
                data = f.read()
            client.storage.from_(_INDICES_BUCKET).upload(
                storage_path,
                data,
                {"upsert": "true"},
            )
            logger.info("Uploaded index file to Supabase: %s", storage_path)
        except Exception as e:
            logger.warning("Index upload failed (%s): %s", storage_path, e)
            success = False

    return success


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_index(collection_id: str, dest_dir: str) -> bool:
    """
    Download FAISS index files for *collection_id* into *dest_dir*.

    Returns True if both files were downloaded successfully.
    """
    client = _get_client()
    if not client:
        return False

    os.makedirs(dest_dir, exist_ok=True)
    success = True
    for fname in ("index.faiss", "index.pkl"):
        storage_path = f"{collection_id}/{fname}"
        dest_path = os.path.join(dest_dir, fname)
        if os.path.exists(dest_path):
            # Already cached locally
            continue
        try:
            data = client.storage.from_(_INDICES_BUCKET).download(storage_path)
            with open(dest_path, "wb") as f:
                f.write(data)
            logger.info("Downloaded index file from Supabase: %s → %s", storage_path, dest_path)
        except Exception as e:
            logger.warning("Index download failed (%s): %s", storage_path, e)
            success = False

    return success


def get_pdf_signed_url(collection_id: str, filename: str, expires_in: int = 3600) -> Optional[str]:
    """
    Return a signed URL to download a PDF from Supabase Storage.
    Expires in *expires_in* seconds (default 1 hour).
    """
    client = _get_client()
    if not client:
        return None
    try:
        storage_path = f"{collection_id}/{filename}"
        res = client.storage.from_(_UPLOADS_BUCKET).create_signed_url(
            storage_path, expires_in
        )
        return res.get("signedURL") or res.get("signedUrl")
    except Exception as e:
        logger.warning("Signed URL generation failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# pdf_collections metadata table
# ---------------------------------------------------------------------------

def register_collection(
    collection_id: str,
    file_names: List[str],
    chunk_count: int,
    storage_paths: Optional[List[str]] = None,
) -> bool:
    """
    Insert/upsert a row in the pdf_collections table.

    Returns True on success.
    """
    client = _get_client()
    if not client:
        return False
    try:
        client.table("pdf_collections").upsert({
            "collection_id": collection_id,
            "file_names": file_names,
            "chunk_count": chunk_count,
            "storage_paths": storage_paths or [],
        }).execute()
        logger.info("Registered collection in Supabase DB: %s", collection_id)
        return True
    except Exception as e:
        logger.warning("Collection registration failed: %s", e)
        return False


def delete_collection_from_db(collection_id: str) -> bool:
    """Delete collection metadata from the pdf_collections table."""
    client = _get_client()
    if not client:
        return False
    try:
        client.table("pdf_collections").delete().eq(
            "collection_id", collection_id
        ).execute()
        # Also delete storage objects
        for bucket, fname_list in [
            (_UPLOADS_BUCKET, []),      # unknown filenames — iterate
            (_INDICES_BUCKET, ["index.faiss", "index.pkl"]),
        ]:
            if fname_list:
                paths = [f"{collection_id}/{f}" for f in fname_list]
            else:
                # For uploads bucket list files first
                try:
                    items = client.storage.from_(bucket).list(collection_id)
                    paths = [f"{collection_id}/{item['name']}" for item in items]
                except Exception:
                    paths = []
            if paths:
                client.storage.from_(bucket).remove(paths)
        logger.info("Deleted collection from Supabase: %s", collection_id)
        return True
    except Exception as e:
        logger.warning("Collection deletion from Supabase failed: %s", e)
        return False


def list_collections() -> List[Dict[str, Any]]:
    """
    Return all rows from pdf_collections table, ordered newest first.

    Each row has: collection_id, file_names, chunk_count, storage_paths, created_at.
    Returns empty list if Supabase is disabled or query fails.
    """
    client = _get_client()
    if not client:
        return []
    try:
        res = (
            client.table("pdf_collections")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.warning("list_collections from Supabase failed: %s", e)
        return []


def get_collection(collection_id: str) -> Optional[Dict[str, Any]]:
    """Return a single pdf_collections row, or None."""
    client = _get_client()
    if not client:
        return None
    try:
        res = (
            client.table("pdf_collections")
            .select("*")
            .eq("collection_id", collection_id)
            .single()
            .execute()
        )
        return res.data
    except Exception as e:
        logger.warning("get_collection failed for %s: %s", collection_id, e)
        return None
