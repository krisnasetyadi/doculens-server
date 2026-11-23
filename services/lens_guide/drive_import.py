"""
Lens Guide (MS-1181): a public Google Drive link pasted in the chat is offered
for import. Nothing is imported here: the reply carries the PDF links the
existing importer found (services/upload.py), and the UI imports them through
the existing upload-from-urls only when the user agrees, with its usual quota
and size checks.
"""
import re
from typing import Optional, Tuple

from fastapi import HTTPException

from models import DriveFolderRequest, Guide, GuideImportLink
from services.upload import list_drive_folder_items

# A Drive file or folder link, with or without the scheme ("drive.google.com/drive/folders/1kZ...").
_DRIVE_LINK = re.compile(
    r"(?:https?://)?(?:www\.)?drive\.google\.com/(?:drive/(?:u/\d+/)?folders/|file/d/)[\w-]+[^\s)\]>\"']*",
    re.IGNORECASE,
)


def find_drive_link(question: str) -> Optional[str]:
    """The first Drive file or folder link in the message, as a full URL."""
    found = _DRIVE_LINK.search(question or "")
    if not found:
        return None
    url = found.group(0).rstrip(".,;")
    return url if url.lower().startswith("http") else f"https://{url}"


async def import_offer(url: str, title: Optional[str], language: str) -> Tuple[str, Optional[Guide]]:
    """The reply to a pasted Drive link: what is behind it, and whether to import it."""
    id_ = language == "id"
    if "/folders/" not in url:
        urls = [url]
    else:
        try:
            listing = await list_drive_folder_items(DriveFolderRequest(url=url))
        except HTTPException:
            if id_:
                return (
                    "Saya tidak bisa membuka link itu. Pastikan folder Drive-nya dibagikan ke siapa saja yang "
                    "memiliki link, lalu tempel lagi di sini.",
                    None,
                )
            return (
                "I couldn't open that link. Make sure the Drive folder is shared with anyone who has the link, "
                "then paste it here again.",
                None,
            )
        urls = [item.url for item in listing.files]
        if not urls:
            if id_:
                return "Folder Drive itu tidak berisi PDF. Saat ini hanya file PDF yang bisa saya impor dari link.", None
            return "That Drive folder has no PDFs. For now only PDF files can be imported from a link.", None

    count = len(urls)
    folder = "/folders/" in url
    if id_:
        what = f"folder Google Drive berisi {count} PDF" if folder else "file Google Drive"
        text = f"Ini link {what}. Saya belum membaca isinya, tapi bisa mengimpornya ke Sources supaya bisa dipakai di chat ini."
    else:
        pdfs = f"{count} PDF" + ("" if count == 1 else "s")
        what = f"a Google Drive folder with {pdfs}" if folder else "a Google Drive file"
        text = f"This is a link to {what}. I haven't read it yet, but I can import it into Sources so this chat can use it."
    return text, Guide(import_link=GuideImportLink(url=url, title=title, file_count=count, urls=urls))
