# services/folders.py
"""
Source folders CRUD (MS-274/MS-557).

A folder groups `collections` rows (kind='pdf' or 'chat') for the Files tab.
Owner-scoped exactly like collections themselves — admins see every folder,
everyone else only their own. Deleting a folder moves its direct sources and
child folders to its parent without deleting their contents.

Moving a *source* into a folder is not here — that endpoint lives with the
source's own table (services/collections.py for pdf, services/chat.py for chat)
so it can reuse each one's existing authorization rule (owner-gated vs
admin-gated).
"""
import asyncio
import uuid
from typing import Optional

from fastapi import HTTPException

from services.storage import supabase as supabase_storage
from models import Folder, FolderCreate, FolderRename, UserRecord


def _can_access_folder(row: Optional[dict], user: UserRecord) -> bool:
    """Admins can access everything; everyone else only their own folder."""
    if user.role == "admin":
        return True
    return bool(row) and row.get("owner_id") == user.user_id


async def create_folder(body: FolderCreate, user: UserRecord):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Folder name cannot be empty")

    if body.parent_folder_id:
        parent = await asyncio.to_thread(supabase_storage.get_folder, body.parent_folder_id)
        if not parent or parent["owner_id"] != user.user_id:
            raise HTTPException(status_code=404, detail="Parent folder not found")

    folder_id = str(uuid.uuid4())
    try:
        row = await asyncio.to_thread(
            supabase_storage.create_folder, folder_id, name, user.user_id, body.parent_folder_id
        )
    except supabase_storage.FolderHierarchyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create folder")
    return Folder(**row)


async def list_folders(user: UserRecord):
    rows = await asyncio.to_thread(
        supabase_storage.list_folders_for_user, user.user_id, user.role == "admin"
    )
    return [Folder(**row) for row in rows]


async def rename_folder(folder_id: str, body: FolderRename, user: UserRecord):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Folder name cannot be empty")

    row = await asyncio.to_thread(supabase_storage.get_folder, folder_id)
    if not row or not _can_access_folder(row, user):
        raise HTTPException(status_code=404, detail="Folder not found")
    if row["owner_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to rename this folder")

    parent_folder_id = (
        body.parent_folder_id
        if "parent_folder_id" in body.model_fields_set
        else row["parent_folder_id"]
    )
    if parent_folder_id:
        parent = await asyncio.to_thread(supabase_storage.get_folder, parent_folder_id)
        if not parent or parent["owner_id"] != user.user_id:
            raise HTTPException(status_code=404, detail="Parent folder not found")

    try:
        updated = await asyncio.to_thread(
            supabase_storage.rename_folder, folder_id, name, parent_folder_id
        )
    except supabase_storage.FolderHierarchyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to rename folder")
    return Folder(**updated)


async def delete_folder(folder_id: str, user: UserRecord):
    row = await asyncio.to_thread(supabase_storage.get_folder, folder_id)
    if not row or not _can_access_folder(row, user):
        raise HTTPException(status_code=404, detail="Folder not found")
    if row["owner_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to delete this folder")

    ok = await asyncio.to_thread(supabase_storage.delete_folder, folder_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to delete folder")
    return {"folder_id": folder_id, "deleted": True}
