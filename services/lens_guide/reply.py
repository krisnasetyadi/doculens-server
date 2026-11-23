"""
Lens Guide replies for find_folder (MS-1181): the text and the cards, built by
the server from the user's own scope, never written by the model, so a folder
outside the scope can never be named.

The text always carries the full folder path: the cards are not saved with the
message, and the next turn only sees the conversation as text.

Blocking (an owner-name lookup for an admin's matches): call through
asyncio.to_thread.
"""
import re
from typing import Dict, List, Optional, Tuple

from models import Guide, GuideFolder
from services.lens_guide.match import folder_label, subfolders
from services.lens_guide.scope import Scope, ScopeFolder
from services.storage import supabase as supabase_storage

# shortcut: a long list of same-named folders is cut, the user is asked for a
# fuller name instead; upgrade to a searchable picker if admins hit it often.
MAX_CHOICES = 10
MAX_PICKER = 50  # same cap as the router's enums (SEARCH_TOOL_MAX_ENUM)

# The no-match reply quotes the name the user asked for; requested_name reads
# it back so a Drive folder pasted next can be imported under that name.
_NOT_FOUND = {
    "id": 'Saya belum menemukan folder "{name}"',
    "en": 'I couldn\'t find a folder "{name}"',
}


def _owner_names(folders: List[ScopeFolder], user_id: str) -> Dict[str, str]:
    others = sorted({f.owner_id for f in folders if f.owner_id and f.owner_id != user_id})
    return supabase_storage.get_user_names(others) if others else {}


def _card(folder: ScopeFolder, scope: Scope, user_id: str, names: Dict[str, str], language: str) -> GuideFolder:
    owner = None
    if folder.owner_id and folder.owner_id != user_id:
        owner = names.get(folder.owner_id) or ("pengguna lain" if language == "id" else "another user")
    return GuideFolder(
        path=list(folder.path),
        file_count=folder.file_count,
        subfolder_count=len(subfolders(folder, scope)),
        owner=owner,
    )


def _cards(folders: List[ScopeFolder], scope: Scope, user_id: str, language: str) -> List[GuideFolder]:
    names = _owner_names(folders, user_id)
    return [_card(f, scope, user_id, names, language) for f in folders]


def folder_reply(
    name: str, matches: List[ScopeFolder], scope: Scope, user_id: str, language: str,
) -> Tuple[str, Guide]:
    """The reply to find_folder(name): confirm one, ask which of several, or
    ask where the documents are."""
    # The card shows the counts and the owner, so the text does not repeat
    # them. The one folder's path stays in the text: a "ya" typed instead of
    # clicked still carries the scope to the next turn. Several folders are
    # only on the cards; the one clicked sends its path as the user's reply.
    if len(matches) == 1:
        card = _cards(matches, scope, user_id, language)[0]
        path = f"**{' › '.join(card.path)}**"
        if language == "id":
            text = f"Saya menemukan folder {path}. Benar ini folder yang kamu maksud?"
        else:
            text = f"I found the folder {path}. Is this the one you mean?"
        return text, Guide(folder=card)

    if matches:
        cards = _cards(matches[:MAX_CHOICES], scope, user_id, language)
        if language == "id":
            text = f'Ada {len(matches)} folder yang cocok dengan "{name}" dan bisa kamu buka. Yang mana yang kamu maksud?'
            if len(matches) > MAX_CHOICES:
                text += f" Di sini ada {MAX_CHOICES} yang pertama; kalau tidak ada, sebutkan nama atau lokasinya lebih lengkap."
        else:
            text = f'{len(matches)} folders you can open match "{name}". Which one do you mean?'
            if len(matches) > MAX_CHOICES:
                text += f" The first {MAX_CHOICES} are shown; if it isn't there, tell me its full name or location."
        return text, Guide(folders=cards)

    # Only the user's own folders to pick from, even for an admin: "one of your folders".
    own = [f for f in scope.folders if not f.owner_id or f.owner_id == user_id]
    picker = sorted(own, key=lambda f: folder_label(f).lower())[:MAX_PICKER]
    not_found = _NOT_FOUND["id" if language == "id" else "en"].format(name=name)
    if language == "id":
        text = (
            f"{not_found} di dokumen yang bisa kamu buka. Dokumennya ada di mana? Pilih salah satu foldermu, "
            "atau tempel link folder Google Drive publiknya di sini untuk saya impor."
        )
    else:
        text = (
            f"{not_found} among the documents you can open. Where are its documents? Pick one of your "
            "folders, or paste the public Google Drive folder link here for me to import."
        )
    return text, Guide(missing_source=True, folders=_cards(picker, scope, user_id, language) or None)


def requested_name(memory: Optional[List[Dict[str, str]]]) -> Optional[str]:
    """The project name of the latest no-match reply, if the latest reply was one."""
    last = next((t.get("content", "") for t in reversed(memory or []) if t.get("role") == "assistant"), "")
    for template in _NOT_FOUND.values():
        prefix = re.escape(template.split('"')[0])
        found = re.match(prefix + r'"([^"]+)"', last)
        if found:
            return found.group(1)
    return None


def empty_folder_reply(folders: List[ScopeFolder], language: str) -> str:
    """A confirmed folder with no active document in it or below it: say so
    rather than search everything instead."""
    names = ", ".join(f"**{folder_label(f)}**" for f in folders)
    if language == "id":
        return f"Folder {names} belum berisi dokumen aktif, jadi belum ada yang bisa saya cari di sana."
    return f"The folder {names} has no active documents yet, so there is nothing there for me to search."
