# router/collections.py
"""PDF collection endpoints. Each route reads the request and calls
services/collections.py, where the storage lookups and file handling live."""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from models import (
    CollectionActiveResponse,
    CollectionFilesResponse,
    CollectionFolderResponse,
    CollectionInfo,
    CollectionReferenceFrameworkResponse,
    MoveToFolderRequest,
    SetPdfCollectionActiveRequest,
    SetReferenceFrameworkRequest,
    StatusMessageResponse,
    TextContentResponse,
    UserRecord,
)
from services import collections as collection_service
from services.auth import get_current_user

router = APIRouter()


@router.get("/pdf-collections", response_model=List[CollectionInfo])
async def list_collections(user: UserRecord = Depends(get_current_user)):
    """List PDF document collections visible to the current user.

    Admins see every collection (Supabase DB → S3 scan → local disk fallback).
    Non-admins only see DB-backed collections they own — the S3-scan and
    local-disk fallbacks can't be attributed to an owner, so they're
    admin-only.
    """
    return await collection_service.list_collections(user)


@router.post("/pdf-collections/activate", response_model=CollectionActiveResponse)
async def set_pdf_collection_active(
    body: SetPdfCollectionActiveRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Toggle a PDF collection's active status (used as a knowledge source)."""
    return await collection_service.set_collection_active(body, user)


@router.post("/pdf-collections/move-to-folder", response_model=CollectionFolderResponse)
async def move_pdf_collection_to_folder(
    body: MoveToFolderRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Assign (or unassign, when folder_id is null) a PDF collection to a
    folder (MS-274). Owner-gated like every other pdf-collections mutation."""
    return await collection_service.move_to_folder(body, user)


@router.post("/pdf-collections/reference-framework", response_model=CollectionReferenceFrameworkResponse)
async def set_pdf_collection_reference_framework(
    body: SetReferenceFrameworkRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Flag or unflag a PDF collection as a reference framework, the documents
    Gap Check offers as its reference (MS-848). Owner-gated."""
    return await collection_service.set_reference_framework(body, user)


@router.delete("/pdf-collections/{collection_id}", response_model=StatusMessageResponse)
async def delete_collection(collection_id: str, user: UserRecord = Depends(get_current_user)):
    """Delete a collection from Supabase Storage + DB and local disk."""
    return await collection_service.delete_collection(collection_id, user)


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
    return await collection_service.serve_file(collection_id, file_name, user)


@router.get("/pdf-collections/{collection_id}/text-content", response_model=TextContentResponse)
async def get_pdf_text_content(
    collection_id: str,
    file_name: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    slide: Optional[int] = Query(default=None, ge=1),
    user: UserRecord = Depends(get_current_user),
):
    """Paginated plain-text content for a .txt or .pptx document source
    (kind='pdf'), read line by line, for the formatted text viewer (MS-415).
    A .pptx is shown slide by slide, each slide opening with a "Slide N"
    line; `slide` starts the page at that line instead of `offset` (MS-910).
    Read-only — never writes to the stored file.
    """
    return await collection_service.get_text_content(collection_id, file_name, offset, limit, user, slide)


@router.get("/collection/{collection_id}/files", response_model=CollectionFilesResponse, response_model_exclude_unset=True)
async def list_collection_files(
    collection_id: str,
    user: UserRecord = Depends(get_current_user),
):
    """List all PDF files in a collection (Supabase DB → local disk fallback)."""
    return await collection_service.list_files(collection_id, user)
