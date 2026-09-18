# router/folders.py
"""
Source folders CRUD (MS-274).

A folder groups `collections` rows (kind='pdf' or 'chat') for the Files tab.
Owner-scoped exactly like collections themselves — admins see every folder,
everyone else only their own. Deleting a folder never deletes the sources
inside it: storage.delete_folder relies on the folder_id FK's
ON DELETE SET NULL, so sources are unassigned, not removed.

Moving a *source* into a folder is not here — that endpoint lives with the
source's own table (router/collections.py for pdf, router/chat.py for chat)
so it can reuse each one's existing authorization rule (owner-gated vs
admin-gated).
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import asyncio
import uuid

from models import Folder, FolderCreate, FolderRename
import storage as supabase_storage
from router.auth import get_current_user, UserRecord

router = APIRouter()


def _can_access_folder(row: Optional[dict], user: UserRecord) -> bool:
    """Admins can access everything; everyone else only their own folder."""
    if user.role == "admin":
        return True
    return bool(row) and row.get("owner_id") == user.user_id


@router.post("/source-folders", response_model=Folder, status_code=201)
async def create_folder(body: FolderCreate, user: UserRecord = Depends(get_current_user)):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Folder name cannot be empty")

    folder_id = str(uuid.uuid4())
    row = await asyncio.to_thread(
        supabase_storage.create_folder, folder_id, name, user.user_id
    )
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create folder")
    return Folder(**row)


@router.get("/source-folders", response_model=List[Folder])
async def list_folders(user: UserRecord = Depends(get_current_user)):
    """Folders this account may use: its own, or every folder for an admin."""
    rows = await asyncio.to_thread(
        supabase_storage.list_folders_for_user, user.user_id, user.role == "admin"
    )
    return [Folder(**row) for row in rows]


@router.put("/source-folders/{folder_id}", response_model=Folder)
async def rename_folder(
    folder_id: str,
    body: FolderRename,
    user: UserRecord = Depends(get_current_user),
):
    """Owner-only rename. An admin can see another user's folder but not
    rename it, same rule as skill edits (router/skills.py)."""
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Folder name cannot be empty")

    row = await asyncio.to_thread(supabase_storage.get_folder, folder_id)
    if not row or not _can_access_folder(row, user):
        raise HTTPException(status_code=404, detail="Folder not found")
    if row["owner_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to rename this folder")

    updated = await asyncio.to_thread(supabase_storage.rename_folder, folder_id, name)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to rename folder")
    return Folder(**updated)


@router.delete("/source-folders/{folder_id}")
async def delete_folder(folder_id: str, user: UserRecord = Depends(get_current_user)):
    """Owner-only delete. Sources inside the folder are unassigned
    (folder_id -> NULL via the FK), never deleted — see storage.delete_folder."""
    row = await asyncio.to_thread(supabase_storage.get_folder, folder_id)
    if not row or not _can_access_folder(row, user):
        raise HTTPException(status_code=404, detail="Folder not found")
    if row["owner_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to delete this folder")

    ok = await asyncio.to_thread(supabase_storage.delete_folder, folder_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to delete folder")
    return {"folder_id": folder_id, "deleted": True}
