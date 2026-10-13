# router/upload.py
"""PDF collection upload endpoints. Each route reads the request and calls
services/upload.py, where saving, remote imports and Drive listing live."""

from typing import List, Literal

from fastapi import APIRouter, Depends, File, Query, UploadFile

from models import (
    DriveFolderItemsResponse,
    DriveFolderRequest,
    UploadFromUrlRequest,
    UploadFromUrlsRequest,
    UploadResponse,
    UserRecord,
)
from services import upload as upload_service
from services.auth import get_current_user

router = APIRouter()


@router.post("/pdf-collections/drive/folder-items", response_model=DriveFolderItemsResponse)
async def list_drive_folder_items(
    payload: DriveFolderRequest,
    user: UserRecord = Depends(get_current_user),
):
    return await upload_service.list_drive_folder_items(payload)


@router.get("/pdf-collections/uploads/{upload_id}")
async def get_pdf_upload_status(upload_id: str, user: UserRecord = Depends(get_current_user)):
    return await upload_service.get_upload_status(upload_id, user)


@router.post("/pdf-collections/upload", response_model=UploadResponse,
             responses={200: {"content": {"application/x-ndjson": {}}}})
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    persist_mode: Literal["auto", "local", "database"] = Query(
        "auto",
        description="Persistence mode: auto (default), local (disk only), database (require DATABASE_URL)",
    ),
    user: UserRecord = Depends(get_current_user),
    stream_progress: bool = Query(False),
):
    """Upload and process document files (PDF, DOC, DOCX, CSV, XLSX, TXT), then
    persist to Supabase Storage."""
    return await upload_service.upload_files(files, persist_mode, user, stream_progress)


@router.post("/pdf-collections/upload-from-url", response_model=UploadResponse)
async def upload_pdf_from_url(
    payload: UploadFromUrlRequest,
    persist_mode: Literal["auto", "local", "database"] = Query(
        "auto",
        description="Persistence mode: auto (default), local (disk only), database (require DATABASE_URL)",
    ),
    user: UserRecord = Depends(get_current_user),
):
    """Download a public PDF from a remote URL (including Google Drive public links) and process it as a collection."""
    return await upload_service.upload_from_url(payload, persist_mode, user)


@router.post("/pdf-collections/upload-from-urls", response_model=UploadResponse)
async def upload_pdfs_from_urls(
    payload: UploadFromUrlsRequest,
    persist_mode: Literal["auto", "local", "database"] = Query(
        "auto",
        description="Persistence mode: auto (default), local (disk only), database (require DATABASE_URL)",
    ),
    user: UserRecord = Depends(get_current_user),
):
    return await upload_service.upload_from_urls(payload, persist_mode, user)
