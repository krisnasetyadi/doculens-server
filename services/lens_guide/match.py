"""
Lens Guide folder matching (MS-1181): which folders in the user's scope a name
the user said refers to. Deterministic and only over Scope.folders, so a folder
the user cannot open can never be matched, named, or suggested.
"""
import re
from typing import List

from services.lens_guide.scope import Scope, ScopeFolder

# Words people put around a folder name that are not part of it.
_FILLER = {"project", "projek", "proyek", "projects", "modul", "module", "folder", "folders"}
# "&" is not a word, so "Backup & Pemulihan" must also match "backup dan pemulihan".
_CONNECTORS = {"dan", "and"}


def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[^\W_]+", (text or "").lower()) if w not in _CONNECTORS]


def _words(text: str) -> List[str]:
    return [w for w in _tokens(text) if w not in _FILLER]


def match_folders(name: str, scope: Scope) -> List[ScopeFolder]:
    """Folders `name` refers to, at any level, ordered by path.

    A folder whose own name is exactly `name` wins outright. Otherwise a folder
    matches when every word of `name` is somewhere in its path and at least one
    is in its own name: "Delivery Klien Y" finds Delivery › Proyek Klien Y but
    not its subfolder Delivery › Proyek Klien Y › Backend."""
    wanted = _words(name)
    if not wanted:
        return []
    exact = [f for f in scope.folders if _words(f.name) == wanted]
    found = exact or [
        f for f in scope.folders
        if set(wanted) <= set(_words(" ".join(f.path))) and set(wanted) & set(_words(f.name))
    ]
    return sorted(found, key=lambda f: [p.lower() for p in f.path])


def named_exactly(name: str, folder: ScopeFolder) -> bool:
    """`name` is the folder's whole name, not a part of it or of its path
    (MS-1182: such a folder is confirmed together with the first question)."""
    return bool(_words(name)) and _words(folder.path[-1]) == _words(name)


def folder_label(folder: ScopeFolder) -> str:
    """The path as written in replies and passed back by the router."""
    return " › ".join(folder.path)


def subfolders(folder: ScopeFolder, scope: Scope) -> List[ScopeFolder]:
    """Every folder below `folder`, at any depth."""
    children: dict = {}
    for f in scope.folders:
        children.setdefault(f.parent_folder_id, []).append(f)
    found, todo, seen = [], [folder.folder_id], {folder.folder_id}
    while todo:
        for child in children.get(todo.pop(), []):
            if child.folder_id not in seen:
                seen.add(child.folder_id)
                found.append(child)
                todo.append(child.folder_id)
    return found


def collection_ids_in(folders: List[ScopeFolder], scope: Scope) -> List[str]:
    """Collections filed in these folders or any folder below them."""
    wanted = set()
    for f in folders:
        wanted |= {f.folder_id} | {s.folder_id for s in subfolders(f, scope)}
    return [c.collection_id for c in scope.collections if c.folder_id in wanted]


def resolve_folder_label(label: str, scope: Scope) -> List[ScopeFolder]:
    """The folder(s) a path from the router names: exact path first, else
    matched like a name (the router writes free text past the enum cap)."""
    exact = [f for f in scope.folders if folder_label(f).lower() == label.strip().lower()]
    return exact or match_folders(label, scope)


def find_mentioned_folders(question: str, scope: Scope) -> List[ScopeFolder]:
    """Folders the message names: a name of two or more words ("Proyek Klien
    Y"), or a one-word name right after "proyek", "modul", "folder", ...
    ("modul Backend"). A lone common word ("laporan", "policy") that happens
    to be a folder name is not taken as one."""
    text = " " + " ".join(_tokens(question)) + " "
    found = []
    for f in scope.folders:
        words = _words(f.name)
        name = " ".join(words)
        if len(name) < 4:
            continue
        if len(words) >= 2:
            named = f" {name} " in text
        else:
            named = any(f" {filler} {name} " in text for filler in _FILLER)
        if named:
            found.append(f)
    return found
