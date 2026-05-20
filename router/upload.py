# router/upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import shutil
import uuid
from utils import process_pdfs
from models import UploadResponse
from config import config
import logging
import storage as supabase_storage

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process PDF files, then persist to Supabase Storage."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    collection_id = str(uuid.uuid4())
    collection_path = os.path.join(config.upload_folder, collection_id)
    os.makedirs(collection_path, exist_ok=True)

    saved_files = []
    file_names = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            continue
        file_path = os.path.join(collection_path, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)
        file_names.append(file.filename)

    if not saved_files:
        raise HTTPException(
            status_code=400, detail="No valid PDF files uploaded")

    try:
        chunk_count = process_pdfs(saved_files, collection_id)
    except Exception as e:
        shutil.rmtree(collection_path, ignore_errors=True)
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process PDFs")

    # ── Persist to Supabase Storage (best-effort, non-blocking) ──────────────
    storage_paths: List[str] = []
    if supabase_storage.is_enabled():
        logger.info("Uploading collection %s to Supabase Storage…", collection_id)

        # 1. Raw PDF files
        for file_path, fname in zip(saved_files, file_names):
            path = supabase_storage.upload_pdf(collection_id, file_path, fname)
            if path:
                storage_paths.append(path)

        # 2. FAISS index files
        index_dir = os.path.join(config.index_folder, collection_id)
        supabase_storage.upload_index(collection_id, index_dir)

        # 3. Register in pdf_collections table
        supabase_storage.register_collection(
            collection_id=collection_id,
            file_names=file_names,
            chunk_count=chunk_count,
            storage_paths=storage_paths,
        )
        logger.info("Collection %s persisted to Supabase (%d file(s))",
                    collection_id, len(storage_paths))
    else:
        logger.info("Supabase not configured — collection stored on local disk only")

    return UploadResponse(
        collection_id=collection_id,
        file_count=len(saved_files),
        status="success"
    )
