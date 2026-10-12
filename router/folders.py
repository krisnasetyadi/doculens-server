# router/folders.py
"""Source folder endpoints (MS-274/MS-557). Each route reads the request and
calls services/folders.py, where the ownership rules live."""
from typing import List

from fastapi import APIRouter, Depends

from models import Folder, FolderCreate, FolderRename, UserRecord
from services import folders as folder_service
from services.auth import get_current_user

router = APIRouter()


@router.post("/source-folders", response_model=Folder, status_code=201)
async def create_folder(body: FolderCreate, user: UserRecord = Depends(get_current_user)):
    return await folder_service.create_folder(body, user)


@router.get("/source-folders", response_model=List[Folder])
async def list_folders(user: UserRecord = Depends(get_current_user)):
    """Folders this account may use: its own, or every folder for an admin."""
    return await folder_service.list_folders(user)


@router.put("/source-folders/{folder_id}", response_model=Folder)
async def rename_folder(
    folder_id: str,
    body: FolderRename,
    user: UserRecord = Depends(get_current_user),
):
    """Owner-only rename. An admin can see another user's folder but not
    rename it, same rule as skill edits (router/skills.py)."""
    return await folder_service.rename_folder(folder_id, body, user)


@router.delete("/source-folders/{folder_id}")
async def delete_folder(folder_id: str, user: UserRecord = Depends(get_current_user)):
    """Owner-only delete. Direct child folders and sources move up one level."""
    return await folder_service.delete_folder(folder_id, user)
