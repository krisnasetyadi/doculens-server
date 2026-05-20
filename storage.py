"""
storage.py
----------
Supabase Storage wrapper using the S3-compatible API (boto3).

Credentials needed (set in .env / HF Space secrets):
  SUPABASE_S3_ACCESS_KEY_ID  — S3 Access Key ID from Supabase Storage settings
  SUPABASE_S3_SECRET_KEY     — S3 Secret Access Key from Supabase Storage settings
  SUPABASE_S3_ENDPOINT       — e.g. https://<ref>.storage.supabase.co/storage/v1/s3
  SUPABASE_S3_REGION         — e.g. ap-southeast-1

Find these at: Supabase Dashboard → Storage → S3 Access Keys

Graceful degradation:
  If any S3 env var is missing, file operations silently no-op and the caller
  falls back to local disk.  Metadata table ops use DATABASE_URL directly via
  psycopg2 — they work independently of S3.

Bucket layout:
  pdf-uploads/{collection_id}/{filename}.pdf
  pdf-indices/{collection_id}/index.faiss
  pdf-indices/{collection_id}/index.pkl
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_UPLOADS_BUCKET = "pdf-uploads"
_INDICES_BUCKET = "pdf-indices"

_migration_done = False


# ---------------------------------------------------------------------------
# Auto-migration  (creates pdf_collections table via psycopg2)
# ---------------------------------------------------------------------------

def ensure_schema():
    """
    Create the pdf_collections table if it doesn't already exist.
    Called automatically on the first metadata operation each session.
    """
    global _migration_done
    if _migration_done:
        return

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        _migration_done = True
        return

    try:
        import psycopg2
        conn = psycopg2.connect(database_url, sslmode="require")
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_collections (
                    id            BIGSERIAL PRIMARY KEY,
                    collection_id TEXT        NOT NULL UNIQUE,
                    file_names    TEXT[]      NOT NULL DEFAULT '{}',
                    chunk_count   INTEGER     NOT NULL DEFAULT 0,
                    storage_paths TEXT[]      NOT NULL DEFAULT '{}',
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_pdf_collections_cid
                    ON pdf_collections (collection_id);
                CREATE INDEX IF NOT EXISTS idx_pdf_collections_created
                    ON pdf_collections (created_at DESC);
            """)
        conn.close()
        logger.info("pdf_collections schema ensured.")
    except Exception as e:
        logger.warning("Auto-migration skipped (will use disk fallback): %s", e)

    _migration_done = True


# ---------------------------------------------------------------------------
# S3 client
# ---------------------------------------------------------------------------

def _s3_client():
    """Return a boto3 S3 client pointed at Supabase Storage, or None."""
    access_key = os.getenv("SUPABASE_S3_ACCESS_KEY_ID")
    secret_key = os.getenv("SUPABASE_S3_SECRET_KEY")
    endpoint   = os.getenv("SUPABASE_S3_ENDPOINT")
    region     = os.getenv("SUPABASE_S3_REGION", "ap-southeast-1")

    if not (access_key and secret_key and endpoint):
        return None

    try:
        import boto3
        from botocore.config import Config
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(signature_version="s3v4"),
        )
    except ImportError:
        logger.warning("boto3 not installed — storage features disabled")
        return None
    except Exception as e:
        logger.warning("S3 client init failed: %s", e)
        return None


def is_enabled() -> bool:
    """True if S3 credentials are present."""
    return bool(
        os.getenv("SUPABASE_S3_ACCESS_KEY_ID")
        and os.getenv("SUPABASE_S3_SECRET_KEY")
        and os.getenv("SUPABASE_S3_ENDPOINT")
    )


def has_database() -> bool:
    """True if DATABASE_URL is set."""
    return bool(os.getenv("DATABASE_URL"))


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def upload_pdf(collection_id: str, file_path: str, filename: str) -> Optional[str]:
    """Upload a raw PDF file to Supabase Storage via S3. Returns key or None."""
    s3 = _s3_client()
    if not s3:
        return None
    key = f"{collection_id}/{filename}"
    try:
        s3.upload_file(
            file_path, _UPLOADS_BUCKET, key,
            ExtraArgs={"ContentType": "application/pdf"},
        )
        logger.info("Uploaded PDF to S3: %s/%s", _UPLOADS_BUCKET, key)
        return key
    except Exception as e:
        logger.warning("PDF upload failed (%s): %s", key, e)
        return None


def upload_index(collection_id: str, index_dir: str) -> bool:
    """Upload FAISS index files (index.faiss + index.pkl) to Supabase Storage."""
    s3 = _s3_client()
    if not s3:
        return False
    success = True
    for fname in ("index.faiss", "index.pkl"):
        local_path = os.path.join(index_dir, fname)
        if not os.path.exists(local_path):
            success = False
            continue
        key = f"{collection_id}/{fname}"
        try:
            s3.upload_file(local_path, _INDICES_BUCKET, key)
            logger.info("Uploaded index to S3: %s/%s", _INDICES_BUCKET, key)
        except Exception as e:
            logger.warning("Index upload failed (%s): %s", key, e)
            success = False
    return success


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_index(collection_id: str, dest_dir: str) -> bool:
    """Download FAISS index files for collection_id into dest_dir."""
    s3 = _s3_client()
    if not s3:
        return False
    os.makedirs(dest_dir, exist_ok=True)
    success = True
    for fname in ("index.faiss", "index.pkl"):
        key = f"{collection_id}/{fname}"
        dest_path = os.path.join(dest_dir, fname)
        if os.path.exists(dest_path):
            continue  # already cached locally
        try:
            s3.download_file(_INDICES_BUCKET, key, dest_path)
            logger.info("Downloaded index from S3: %s", key)
        except Exception as e:
            logger.warning("Index download failed (%s): %s", key, e)
            success = False
    return success


def get_pdf_signed_url(collection_id: str, filename: str, expires_in: int = 3600) -> Optional[str]:
    """Return a pre-signed URL to download a PDF (expires in expires_in seconds)."""
    s3 = _s3_client()
    if not s3:
        return None
    key = f"{collection_id}/{filename}"
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": _UPLOADS_BUCKET, "Key": key},
            ExpiresIn=expires_in,
        )
        return url
    except Exception as e:
        logger.warning("Presigned URL generation failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# pdf_collections metadata table  (psycopg2 direct — no supabase-py needed)
# ---------------------------------------------------------------------------

def _db_conn():
    """Return a psycopg2 RealDictCursor connection using DATABASE_URL, or None."""
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(url, sslmode="require",
                                cursor_factory=RealDictCursor)
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.warning("DB connection for storage metadata failed: %s", e)
        return None


def register_collection(
    collection_id: str,
    file_names: List[str],
    chunk_count: int,
    storage_paths: Optional[List[str]] = None,
) -> bool:
    """Upsert a row in pdf_collections. Returns True on success."""
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pdf_collections
                    (collection_id, file_names, chunk_count, storage_paths)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (collection_id) DO UPDATE SET
                    file_names    = EXCLUDED.file_names,
                    chunk_count   = EXCLUDED.chunk_count,
                    storage_paths = EXCLUDED.storage_paths,
                    updated_at    = now()
            """, (collection_id, file_names, chunk_count, storage_paths or []))
        conn.close()
        logger.info("Registered collection in DB: %s", collection_id)
        return True
    except Exception as e:
        logger.warning("register_collection failed: %s", e)
        return False


def delete_collection_from_db(collection_id: str) -> bool:
    """Delete collection metadata row and S3 storage objects."""
    ensure_schema()

    # Delete metadata row
    conn = _db_conn()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM pdf_collections WHERE collection_id = %s",
                    (collection_id,)
                )
            conn.close()
        except Exception as e:
            logger.warning("delete row failed: %s", e)

    # Delete S3 objects
    s3 = _s3_client()
    if s3:
        try:
            for bucket, keys in [
                (_INDICES_BUCKET, [
                    f"{collection_id}/index.faiss",
                    f"{collection_id}/index.pkl",
                ]),
                (_UPLOADS_BUCKET, []),  # list first
            ]:
                if not keys:
                    try:
                        resp = s3.list_objects_v2(
                            Bucket=bucket, Prefix=f"{collection_id}/"
                        )
                        keys = [
                            obj["Key"]
                            for obj in resp.get("Contents", [])
                        ]
                    except Exception:
                        keys = []
                for key in keys:
                    try:
                        s3.delete_object(Bucket=bucket, Key=key)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("S3 object deletion failed: %s", e)

    logger.info("Deleted collection: %s", collection_id)
    return True


def list_collections() -> List[Dict[str, Any]]:
    """
    Return all rows from pdf_collections ordered newest first.
    Returns empty list if DB unavailable.
    """
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT collection_id, file_names, chunk_count,
                       storage_paths, created_at, updated_at
                FROM pdf_collections
                ORDER BY created_at DESC
            """)
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("list_collections failed: %s", e)
        return []


def get_collection(collection_id: str) -> Optional[Dict[str, Any]]:
    """Return a single pdf_collections row, or None."""
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT collection_id, file_names, chunk_count,
                       storage_paths, created_at, updated_at
                FROM pdf_collections
                WHERE collection_id = %s
            """, (collection_id,))
            row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.warning("get_collection failed for %s: %s", collection_id, e)
        return None


Responsibilities:
  - Upload raw PDF files to the 'pdf-uploads' bucket
  - Upload FAISS index files (.faiss + .pkl) to 'pdf-indices' bucket
  - Download index files back to a local temp directory (for FAISS.load_local)
  - Register/delete collection metadata in the pdf_collections table
  - List all collections from the DB (replaces os.listdir on disk)
  - Auto-migrate: creates pdf_collections table on first run if it doesn't exist

Graceful degradation:
  If SUPABASE_URL or SUPABASE_SERVICE_KEY env vars are not set, all
  Storage operations silently no-op and the caller falls back to local disk.
  Auto-migration always runs as long as DATABASE_URL is available.

Bucket layout:
  pdf-uploads/
    {collection_id}/{filename}.pdf
  pdf-indices/
    {collection_id}/index.faiss
    {collection_id}/index.pkl
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_UPLOADS_BUCKET = "pdf-uploads"
_INDICES_BUCKET = "pdf-indices"

# Set to True once auto-migration has been attempted this session
_migration_done = False


# ---------------------------------------------------------------------------
# Auto-migration  (runs via DATABASE_URL / psycopg2, no supabase-py needed)
# ---------------------------------------------------------------------------

def ensure_schema():
    """
    Create the pdf_collections table if it doesn't already exist.
    Called automatically on the first storage operation each session.
    Uses DATABASE_URL so it works even before SUPABASE_SERVICE_KEY is set.
    """
    global _migration_done
    if _migration_done:
        return

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        _migration_done = True
        return

    try:
        import psycopg2
        # Parse sslmode — Supabase requires 'require'
        conn = psycopg2.connect(database_url, sslmode="require")
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_collections (
                    id            BIGSERIAL PRIMARY KEY,
                    collection_id TEXT        NOT NULL UNIQUE,
                    file_names    TEXT[]      NOT NULL DEFAULT '{}',
                    chunk_count   INTEGER     NOT NULL DEFAULT 0,
                    storage_paths TEXT[]      NOT NULL DEFAULT '{}',
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE INDEX IF NOT EXISTS idx_pdf_collections_cid
                    ON pdf_collections (collection_id);

                CREATE INDEX IF NOT EXISTS idx_pdf_collections_created
                    ON pdf_collections (created_at DESC);
            """)
        conn.close()
        logger.info("pdf_collections schema ensured.")
    except Exception as e:
        logger.warning("Auto-migration skipped (will use disk fallback): %s", e)

    _migration_done = True


def _get_client():
    """Return an authenticated Supabase client, or None if not configured."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
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
    """True if Supabase Storage credentials are present."""
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"))


def has_database() -> bool:
    """True if DATABASE_URL is set (used for metadata table operations)."""
    return bool(os.getenv("DATABASE_URL"))


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def upload_pdf(collection_id: str, file_path: str, filename: str) -> Optional[str]:
    """Upload a raw PDF file to Supabase Storage. Returns storage path or None."""
    client = _get_client()
    if not client:
        return None
    storage_path = f"{collection_id}/{filename}"
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        client.storage.from_(_UPLOADS_BUCKET).upload(
            storage_path, data,
            {"content-type": "application/pdf", "upsert": "true"},
        )
        logger.info("Uploaded PDF to Supabase: %s", storage_path)
        return storage_path
    except Exception as e:
        logger.warning("PDF upload to Supabase failed (%s): %s", storage_path, e)
        return None


def upload_index(collection_id: str, index_dir: str) -> bool:
    """Upload FAISS index files (index.faiss + index.pkl) to Supabase Storage."""
    client = _get_client()
    if not client:
        return False
    success = True
    for fname in ("index.faiss", "index.pkl"):
        local_path = os.path.join(index_dir, fname)
        if not os.path.exists(local_path):
            success = False
            continue
        storage_path = f"{collection_id}/{fname}"
        try:
            with open(local_path, "rb") as f:
                data = f.read()
            client.storage.from_(_INDICES_BUCKET).upload(
                storage_path, data, {"upsert": "true"},
            )
            logger.info("Uploaded index to Supabase: %s", storage_path)
        except Exception as e:
            logger.warning("Index upload failed (%s): %s", storage_path, e)
            success = False
    return success


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_index(collection_id: str, dest_dir: str) -> bool:
    """Download FAISS index files for collection_id into dest_dir."""
    client = _get_client()
    if not client:
        return False
    os.makedirs(dest_dir, exist_ok=True)
    success = True
    for fname in ("index.faiss", "index.pkl"):
        storage_path = f"{collection_id}/{fname}"
        dest_path = os.path.join(dest_dir, fname)
        if os.path.exists(dest_path):
            continue  # already cached locally
        try:
            data = client.storage.from_(_INDICES_BUCKET).download(storage_path)
            with open(dest_path, "wb") as f:
                f.write(data)
            logger.info("Downloaded index from Supabase: %s", storage_path)
        except Exception as e:
            logger.warning("Index download failed (%s): %s", storage_path, e)
            success = False
    return success


def get_pdf_signed_url(collection_id: str, filename: str, expires_in: int = 3600) -> Optional[str]:
    """Return a signed URL to download a PDF from Supabase Storage."""
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
# pdf_collections metadata table  (via psycopg2 directly — more reliable than
# supabase-py for server-side inserts with complex types like arrays)
# ---------------------------------------------------------------------------

def _db_conn():
    """Return a psycopg2 connection using DATABASE_URL, or None."""
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(url, sslmode="require",
                                cursor_factory=RealDictCursor)
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.warning("DB connection for storage metadata failed: %s", e)
        return None


def register_collection(
    collection_id: str,
    file_names: List[str],
    chunk_count: int,
    storage_paths: Optional[List[str]] = None,
) -> bool:
    """Upsert a row in pdf_collections. Returns True on success."""
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pdf_collections
                    (collection_id, file_names, chunk_count, storage_paths)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (collection_id) DO UPDATE SET
                    file_names    = EXCLUDED.file_names,
                    chunk_count   = EXCLUDED.chunk_count,
                    storage_paths = EXCLUDED.storage_paths,
                    updated_at    = now()
            """, (
                collection_id,
                file_names,
                chunk_count,
                storage_paths or [],
            ))
        conn.close()
        logger.info("Registered collection in DB: %s", collection_id)
        return True
    except Exception as e:
        logger.warning("register_collection failed: %s", e)
        return False


def delete_collection_from_db(collection_id: str) -> bool:
    """Delete collection metadata and Supabase Storage objects."""
    ensure_schema()

    # Delete from DB
    conn = _db_conn()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM pdf_collections WHERE collection_id = %s",
                    (collection_id,)
                )
            conn.close()
        except Exception as e:
            logger.warning("delete_collection_from_db DB step failed: %s", e)

    # Delete Storage objects
    client = _get_client()
    if client:
        try:
            for bucket, fname_list in [
                (_INDICES_BUCKET, ["index.faiss", "index.pkl"]),
                (_UPLOADS_BUCKET, []),
            ]:
                if fname_list:
                    paths = [f"{collection_id}/{f}" for f in fname_list]
                else:
                    try:
                        items = client.storage.from_(bucket).list(collection_id)
                        paths = [f"{collection_id}/{item['name']}" for item in items]
                    except Exception:
                        paths = []
                if paths:
                    client.storage.from_(bucket).remove(paths)
        except Exception as e:
            logger.warning("Storage object deletion failed: %s", e)

    logger.info("Deleted collection: %s", collection_id)
    return True


def list_collections() -> List[Dict[str, Any]]:
    """
    Return all rows from pdf_collections ordered newest first.
    Returns empty list if DB unavailable.
    """
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT collection_id, file_names, chunk_count,
                       storage_paths, created_at, updated_at
                FROM pdf_collections
                ORDER BY created_at DESC
            """)
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("list_collections failed: %s", e)
        return []


def get_collection(collection_id: str) -> Optional[Dict[str, Any]]:
    """Return a single pdf_collections row, or None."""
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT collection_id, file_names, chunk_count,
                       storage_paths, created_at, updated_at
                FROM pdf_collections
                WHERE collection_id = %s
            """, (collection_id,))
            row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.warning("get_collection failed for %s: %s", collection_id, e)
        return None


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
