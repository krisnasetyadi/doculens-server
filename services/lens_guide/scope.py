"""
Lens Guide scope (MS-1180): the documents and folders a user may open, with
each folder's full path.

Access is not decided here. It comes from the two existing storage queries
(owner_id; admins see everything; kind='pdf' and active only), so this is the
one place a later folder-sharing feature has to widen. Folders and paths are
built only from folders inside that scope: a path never shows the name of a
folder the user cannot open, and a collection whose folder is outside the
scope counts as being at the root.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import UserRecord
from services.storage import supabase as supabase_storage


@dataclass
class ScopeFolder:
    folder_id: str
    name: str
    parent_folder_id: Optional[str]
    path: List[str]             # names from the top folder down, this one last
    file_count: int = 0         # files in this folder and every subfolder
    owner_id: Optional[str] = None      # another user's id only shows up for admins


@dataclass
class ScopeCollection:
    collection_id: str
    title: str
    file_count: int
    folder_id: Optional[str]    # None: at the root (or in a folder outside the scope)
    folder_path: List[str]      # [] at the root


@dataclass
class Scope:
    folders: List[ScopeFolder] = field(default_factory=list)
    collections: List[ScopeCollection] = field(default_factory=list)
    # False when the folders could not be read: "no folders" would then be a lie.
    folders_loaded: bool = True


def _paths(folders: List[dict]) -> Dict[str, List[str]]:
    """folder_id -> names from the top down. A parent outside the list ends the
    path there; a cycle (never valid, see _validate_folder_parent) is cut."""
    by_id = {f["folder_id"]: f for f in folders}
    paths: Dict[str, List[str]] = {}
    for folder_id in by_id:
        names, seen, current = [], set(), folder_id
        while current in by_id and current not in seen:
            seen.add(current)
            names.append(by_id[current]["name"])
            current = by_id[current].get("parent_folder_id")
        paths[folder_id] = names[::-1]
    return paths


def load_scope(user: UserRecord, active_only: bool = True) -> Scope:
    """Blocking (two DB queries): call through asyncio.to_thread.

    active_only=False also takes the collections the user switched off (kept
    out of chat): an analysis may still check them (MS-1114)."""
    is_admin = user.role == "admin"
    try:
        folder_rows = supabase_storage.list_folders_for_user(user.user_id, is_admin, strict=True)
        folders_loaded = True
    except Exception:
        folder_rows, folders_loaded = [], False
    list_collections = (
        supabase_storage.list_active_collections_for_user if active_only else supabase_storage.list_pdf_collections_for_user
    )
    collection_rows = list_collections(user.user_id, is_admin)

    paths = _paths(folder_rows)
    folders = {
        f["folder_id"]: ScopeFolder(
            folder_id=f["folder_id"],
            name=f["name"],
            parent_folder_id=f.get("parent_folder_id") if f.get("parent_folder_id") in paths else None,
            path=paths[f["folder_id"]],
            owner_id=f.get("owner_id"),
        )
        for f in folder_rows
    }

    collections = []
    for row in collection_rows:
        folder_id = row.get("folder_id") if row.get("folder_id") in folders else None
        file_count = len(row.get("file_names") or [])
        collections.append(ScopeCollection(
            collection_id=row["collection_id"],
            title=supabase_storage.collection_display_title(row),
            file_count=file_count,
            folder_id=folder_id,
            folder_path=list(paths[folder_id]) if folder_id else [],
        ))
        # Count the files in this folder and in every folder above it.
        seen = set()
        while folder_id in folders and folder_id not in seen:
            seen.add(folder_id)
            folders[folder_id].file_count += file_count
            folder_id = folders[folder_id].parent_folder_id

    return Scope(folders=list(folders.values()), collections=collections, folders_loaded=folders_loaded)
