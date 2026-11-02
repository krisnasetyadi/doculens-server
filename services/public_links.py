# services/public_links.py
"""
Public link sources (moved out of router/public_links.py by MS-657): web
pages and Google Drive files/folders a user adds as a knowledge source.

Drive folders are listed from their public embedded view when the link is
added (stored as public_link_items). Since MS-907 each link is then indexed
once, in the background (see INDEXING below), and a question only loads that
index. Every URL passes ssrf_guard first.

Schema (created at startup by schema/public_links.py): public_links,
public_link_items.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi import HTTPException

import db
from config import config
from models import (
    CreatePublicLinkRequest,
    PublicLinkItem,
    PublicLinkSkippedItem,
    PublicLinkSource,
    PublicLinkSyncResponse,
    PublicLinksResponse,
    SetPublicLinkActiveRequest,
    UserRecord,
)
from services.security.ssrf_guard import assert_public_url_safe

logger = logging.getLogger(__name__)

GOOGLE_DRIVE_HOSTS = {"drive.google.com", "docs.google.com"}


def _ts(value: Any) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _derive_title(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc or "Public Link"
    path = parsed.path.strip("/")
    if path:
        return f"{host}/{path.split('/')[-1]}"
    return host


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid public URL")


def _name_from_url(url: str) -> str:
    parsed = urlparse(url)
    tail = parsed.path.strip("/").split("/")[-1] if parsed.path else ""
    return tail or parsed.netloc or "item"


def _normalize_host(raw_host: str) -> str:
    host = raw_host.lower()
    return host[4:] if host.startswith("www.") else host


def _extract_google_drive_file_id(raw_url: str) -> str | None:
    parsed = urlparse(raw_url)
    if _normalize_host(parsed.netloc) not in GOOGLE_DRIVE_HOSTS:
        return None

    path_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", parsed.path)
    if path_match:
        return path_match.group(1)

    query = parse_qs(parsed.query)
    return query.get("id", [None])[0]


def _extract_google_drive_folder_id(raw_url: str) -> str | None:
    parsed = urlparse(raw_url)
    if _normalize_host(parsed.netloc) not in GOOGLE_DRIVE_HOSTS:
        return None

    path_match = re.search(r"/drive/folders/([a-zA-Z0-9_-]+)", parsed.path)
    if path_match:
        return path_match.group(1)

    query = parse_qs(parsed.query)
    return query.get("id", [None])[0]


def _to_drive_absolute_url(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return f"https://drive.google.com{href}"
    return f"https://drive.google.com/{href}"


def _parse_drive_folder_items(html_text: str) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    anchor_pattern = re.compile(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    files: List[Dict[str, str]] = []
    folders: List[Dict[str, str]] = []
    seen_files = set()
    seen_folders = set()

    for href, raw_label in anchor_pattern.findall(html_text):
        full_url = _to_drive_absolute_url(unescape(href))
        label = re.sub(r"<[^>]+>", "", unescape(raw_label)).strip()

        file_id = _extract_google_drive_file_id(full_url)
        if file_id:
            if file_id in seen_files:
                continue
            seen_files.add(file_id)
            files.append({
                "name": label or f"Google Drive file {len(files) + 1}",
                "url": full_url,
                "item_type": "file",
            })
            continue

        folder_id = _extract_google_drive_folder_id(full_url)
        if folder_id:
            if folder_id in seen_folders:
                continue
            seen_folders.add(folder_id)
            folders.append({
                "name": label or f"Google Drive folder {len(folders) + 1}",
                "url": full_url,
                "item_type": "folder",
            })

    return files, folders


def _is_probably_pdf(item_url: str, item_name: str) -> bool:
    if item_name.lower().endswith(".pdf"):
        return True
    parsed = urlparse(item_url)
    return "/file/d/" in parsed.path


async def _fetch_drive_folder_page(folder_id: str) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    embed_url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(
            embed_url,
            headers={"User-Agent": "DocuLens/1.0 (+public-links)"},
        )
        response.raise_for_status()
    return _parse_drive_folder_items(response.text)


async def _extract_items_from_source(url: str) -> List[Dict[str, str]]:
    folder_id = _extract_google_drive_folder_id(url)
    if folder_id:
        try:
            files, folders = await _fetch_drive_folder_page(folder_id)
            pdf_files = [entry for entry in files if _is_probably_pdf(entry["url"], entry["name"])]
            return pdf_files + folders
        except Exception as exc:
            logger.warning("public_links: failed extracting folder items from %s: %s", url, exc)
            return []

    file_id = _extract_google_drive_file_id(url)
    if file_id:
        return [{"name": _name_from_url(url), "url": url, "item_type": "file"}]

    return []


def _replace_items(cur, link_id: str, items: List[Dict[str, str]]) -> None:
    cur.execute("DELETE FROM public_link_items WHERE link_id = %s", (link_id,))
    for item in items:
        cur.execute(
            """
            INSERT INTO public_link_items (item_id, link_id, name, url, item_type)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                link_id,
                item["name"],
                item["url"],
                item.get("item_type", "file"),
            ),
        )


_LINK_COLUMNS = """
    link_id, user_id, workspace_id, title, url, status, created_at,
    index_status, index_stage, index_progress, index_error, index_version,
    fingerprint, skipped_items, index_heartbeat_at, last_synced_at
"""


def _with_items(cur, link: Dict[str, Any]) -> Dict[str, Any]:
    cur.execute(
        """
        SELECT item_id, name, url, item_type
        FROM public_link_items
        WHERE link_id = %s
        ORDER BY created_at ASC
        """,
        (link["link_id"],),
    )
    items = cur.fetchall() or []
    return {
        **link,
        "created_at": _ts(link.get("created_at")),
        "item_count": len(items),
        "items": [
            {
                "id": row["item_id"],
                "name": row["name"],
                "url": row["url"],
                "item_type": row["item_type"],
            }
            for row in items
        ],
    }


def _fetch_links(cur, user_id: str, is_admin: bool = False) -> List[Dict[str, Any]]:
    if not is_admin:
        cur.execute(
            f"SELECT {_LINK_COLUMNS} FROM public_links WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        )
    else:
        cur.execute(f"SELECT {_LINK_COLUMNS} FROM public_links ORDER BY created_at DESC")
    return [_with_items(cur, link) for link in cur.fetchall() or []]


def _as_public_link_source(payload: Dict[str, Any]) -> PublicLinkSource:
    return PublicLinkSource(
        link_id=payload["link_id"],
        workspace_id=payload.get("workspace_id"),
        title=payload["title"],
        url=payload["url"],
        status=payload["status"],
        item_count=payload["item_count"],
        created_at=payload["created_at"],
        items=[
            PublicLinkItem(
                id=item["id"],
                name=item["name"],
                url=item["url"],
                item_type=item["item_type"],
            )
            for item in payload.get("items", [])
        ],
        index_status=payload.get("index_status") or "pending",
        index_ready=bool(payload.get("index_version")),
        index_stale=_is_stale(payload),
        index_stage=payload.get("index_stage"),
        index_progress=payload.get("index_progress"),
        index_error=payload.get("index_error"),
        skipped_items=[PublicLinkSkippedItem(**item) for item in payload.get("skipped_items") or []],
        last_synced_at=_ts(payload["last_synced_at"]) if payload.get("last_synced_at") else None,
    )


async def _resolve_runtime_items(
    url: str,
    *,
    max_depth: int = 5,
    seen_urls: Optional[Set[str]] = None,
    strict: bool = False,
) -> List[Dict[str, str]]:
    """A folder's files, sub-folders included. `strict` raises when a folder
    can't be read instead of leaving its files out."""
    seen_urls = seen_urls or set()
    if url in seen_urls:
        return []
    seen_urls.add(url)

    folder_id = _extract_google_drive_folder_id(url)
    if folder_id:
        if max_depth <= 0:
            return []

        try:
            files, folders = await _fetch_drive_folder_page(folder_id)
        except Exception as exc:
            if strict:
                raise
            logger.warning("public_links: runtime folder expansion failed for %s: %s", url, exc)
            return []

        resolved: List[Dict[str, str]] = []
        for entry in files:
            name = entry.get("name", "")
            item_url = entry.get("url", "")
            lowered = name.lower()
            if _is_probably_pdf(item_url, name) or lowered.endswith((".txt", ".md", ".log", ".pptx")):
                resolved.append(
                    {
                        "name": name or _name_from_url(item_url),
                        "url": item_url,
                        "item_type": "file",
                    }
                )

        for folder in folders:
            resolved.extend(
                await _resolve_runtime_items(
                    folder["url"],
                    max_depth=max_depth - 1,
                    seen_urls=seen_urls,
                    strict=strict,
                )
            )

        return resolved

    return [
        {
            "name": _name_from_url(url),
            "url": url,
            "item_type": "file",
        }
    ]


async def resolve_active_public_link_sources(
    link_ids: Optional[List[str]] = None,
    user_id: Optional[str] = None,
    is_admin: bool = False,
) -> List[Dict[str, Any]]:
    conn = db.get_conn("public_links")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    selected_ids = set(link_ids or [])

    try:
        with conn.cursor() as cur:
            raw_links = _fetch_links(cur, user_id or "", is_admin)
    finally:
        conn.close()

    active_links = [
        link for link in raw_links
        if link.get("status") == "active" and (not selected_ids or link.get("link_id") in selected_ids)
    ]
    await asyncio.to_thread(_keep_indexes_current, active_links)
    return active_links


def _keep_indexes_current(links: List[Dict[str, Any]]) -> None:
    """A link never indexed (added before MS-907) starts its first index, without
    making the question wait; it is left out of this answer. Nothing else: a
    link's changes, and a failed build, are picked up by a manual Sync only, so
    questions put no load on Drive or on the indexing worker."""
    for link in links:
        if link.get("index_version") or link.get("index_status") != "pending":
            continue
        try:
            start_indexing(link["link_id"])
        except Exception as exc:
            logger.warning("public_links: starting the index of %s failed: %s", link["link_id"], exc)


async def list_public_links(user: UserRecord):
    conn = db.get_conn("public_links")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        with conn.cursor() as cur:
            raw_links = _fetch_links(cur, user.user_id, user.role == "admin")
        links = [_as_public_link_source(entry) for entry in raw_links]
        return PublicLinksResponse(links=links, count=len(links))
    except Exception as exc:
        logger.error("public_links: list failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list public links")
    finally:
        conn.close()


async def create_public_link(body: CreatePublicLinkRequest, user: UserRecord):
    _validate_url(body.url)
    assert_public_url_safe(body.url)

    title = (body.title or "").strip() or _derive_title(body.url)
    link_id = str(uuid.uuid4())
    item_urls = body.item_urls or []

    extracted_items: List[Dict[str, str]] = []
    if item_urls:
        extracted_items = [
            {
                "name": _name_from_url(item_url),
                "url": item_url,
                "item_type": "file",
            }
            for item_url in item_urls
        ]
    else:
        extracted_items = await _extract_items_from_source(body.url)

    # Check every item before writing anything, so a refused URL leaves no link behind.
    dedup: Dict[str, Dict[str, str]] = {}
    for item in extracted_items:
        _validate_url(item["url"])
        assert_public_url_safe(item["url"])
        dedup[item["url"]] = item

    conn = db.get_conn("public_links")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        # The link and its items are saved together or not at all; db.get_conn
        # rolls back and restores autocommit when the connection goes back.
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public_links (link_id, user_id, workspace_id, title, url, status, index_status)
                VALUES (%s, %s, %s, %s, %s, 'inactive', 'queued')
                RETURNING link_id
                """,
                (link_id, user.user_id, None, title, body.url),
            )
            _replace_items(cur, link_id, list(dedup.values()))
            # Inactive until its index is ready: the build activates it.
            cur.execute(f"SELECT {_LINK_COLUMNS} FROM public_links WHERE link_id = %s", (link_id,))
            created = _with_items(cur, cur.fetchone())
        conn.commit()
    except Exception as exc:
        logger.error("public_links: create failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create public link")
    finally:
        conn.close()

    _start_index_job(_build_index, link_id, None)
    return _as_public_link_source(created)


async def set_public_link_active(body: SetPublicLinkActiveRequest, user: UserRecord):
    conn = db.get_conn("public_links")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")


    try:
        with conn.cursor() as cur:
            if user.role != "admin":
                cur.execute(
                    "SELECT index_version FROM public_links WHERE link_id = %s AND user_id = %s",
                    (body.link_id, user.user_id),
                )
            else:
                cur.execute("SELECT index_version FROM public_links WHERE link_id = %s", (body.link_id,))
            found = cur.fetchone()
            if not found:
                raise HTTPException(status_code=404, detail="Public link not found")
            # Deactivating keeps the index, so turning the link back on needs no
            # re-indexing; a link never indexed has nothing to search yet.
            if body.active and not found["index_version"]:
                raise HTTPException(status_code=409, detail="This link is still being indexed")
            cur.execute(
                "UPDATE public_links SET status = %s WHERE link_id = %s",
                ("active" if body.active else "inactive", body.link_id),
            )

        return {"status": "success", "link_id": body.link_id, "active": body.active}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("public_links: activate failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update active status")
    finally:
        conn.close()


async def delete_public_link(link_id: str, user: UserRecord):
    conn = db.get_conn("public_links")
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")


    try:
        with conn.cursor() as cur:
            if user.role != "admin":
                cur.execute(
                    """
                    DELETE FROM public_links
                    WHERE link_id = %s AND user_id = %s
                    RETURNING link_id
                    """,
                    (link_id, user.user_id),
                )
            else:
                cur.execute(
                    """
                    DELETE FROM public_links
                    WHERE link_id = %s
                    RETURNING link_id
                    """,
                    (link_id,),
                )
            deleted = cur.fetchone()

        if not deleted:
            raise HTTPException(status_code=404, detail="Public link not found")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("public_links: delete failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to delete public link")
    finally:
        conn.close()

    # The row is gone, so nothing searches this index any more; a failure to
    # delete the files only leaves them orphaned in Storage, and is logged.
    await asyncio.to_thread(_delete_index, link_id)
    return {"status": "success", "message": "Public link deleted", "link_id": link_id}


def _delete_index(link_id: str) -> None:
    from services.rag.processor import processor
    from services.storage import supabase as supabase_storage

    processor.drop_public_link_index(link_id)
    if supabase_storage.is_enabled() and not supabase_storage.delete_public_link_index(link_id):
        logger.error("public_links: index of deleted link %s could not be removed from Storage", link_id)


# ===================== INDEXING (MS-907) =====================
# A link is indexed once, in the background, and its index kept in Supabase
# Storage (services/storage/supabase.py, public-links/{link_id}/{version}/).
# A question only loads it (services/rag/public_link_search.py). The row in
# public_links says where it stands:
#
#   pending  -> never indexed (links added before MS-907)
#   queued   -> waiting for the indexing worker
#   indexing -> being built; index_heartbeat_at moves with every file
#   ready | failed
#
# index_version is the build questions use. A rebuild writes a new version and
# moves index_version (and the stored file list) only once it is uploaded, so
# questions keep using the previous one meanwhile, and a failed rebuild leaves
# both as they were.
#
# One build at a time per process (1 worker): a build downloads and embeds
# every file, which a small Space can't do for several links at once. A queued
# or running build is lost when the process stops; startup queues those again
# (start_pending_indexing), and a build whose heartbeat stopped for
# public_link_stale_indexing_minutes can be taken over by a Sync.

_index_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="public-link-index")
# Set on shutdown: the running build stops at its next file instead of holding
# the process open until it finishes; startup picks it up again.
_stopping = threading.Event()

UNEXPECTED_FAILURE = "Indexing stopped unexpectedly. Try Sync."
UNREACHABLE = "Couldn't open the link. Check that it is still shared publicly."


class _Stopped(Exception):
    """The process is shutting down; the build is left for the next startup."""


def _start_index_job(job, *args) -> None:
    """Runs `job(*args)` on the indexing worker. Tests replace this to run nothing."""
    _index_executor.submit(_run_job, job, *args)


def _run_job(job, link_id: str, *args) -> None:
    """A build that raises is marked failed, so it can't sit at "indexing" for good."""
    try:
        job(link_id, *args)
    except _Stopped:
        logger.info("public_links: build of %s stopped for shutdown", link_id)
    except Exception:
        logger.exception("public_links: %s failed for %s", job.__name__, link_id)
        try:
            _fail_indexing(link_id, UNEXPECTED_FAILURE)
        except Exception:
            logger.exception("public_links: could not mark %s as failed", link_id)


def stop_indexing() -> None:
    """On shutdown: drop the queue and stop the running build at its next file."""
    _stopping.set()
    _index_executor.shutdown(wait=False, cancel_futures=True)


def _drive_item_key(url: str) -> str:
    """A Drive file by its file ID (stable across URL shapes), anything else by URL."""
    return _extract_google_drive_file_id(url) or url


def fingerprint_items(items: List[Dict[str, str]]) -> str:
    """What the link holds, as one hash: its files' IDs and names. Adding,
    removing or renaming a file changes it; editing a file in place doesn't."""
    entries = sorted(
        {(_drive_item_key(item["url"]), item.get("name") or "") for item in items if item.get("item_type") != "folder"}
    )
    return hashlib.sha256(json.dumps(entries).encode("utf-8")).hexdigest()


def _is_folder_link(link: Dict[str, Any]) -> bool:
    return bool(_extract_google_drive_folder_id(link["url"]))


async def _list_link_items(link: Dict[str, Any]) -> List[Dict[str, str]]:
    """The files a link holds right now, without downloading any. A Drive folder
    is listed again, sub-folders included, and raises if Drive can't be read (an
    unreachable folder must not read as an emptied one). Other links keep the
    items stored when they were added."""
    if _is_folder_link(link):
        return await _resolve_runtime_items(link["url"], strict=True)
    stored = [item for item in link.get("items", []) if item.get("item_type") != "folder"]
    return stored or [{"name": link["title"] or _name_from_url(link["url"]), "url": link["url"], "item_type": "file"}]


def _run_sql(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    conn = db.get_conn("public_links")
    if not conn:
        raise RuntimeError("Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall() if cur.description else []
    finally:
        conn.close()


def _get_link(link_id: str) -> Optional[Dict[str, Any]]:
    conn = db.get_conn("public_links")
    if not conn:
        raise RuntimeError("Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {_LINK_COLUMNS} FROM public_links WHERE link_id = %s", (link_id,))
            row = cur.fetchone()
            return _with_items(cur, row) if row else None
    finally:
        conn.close()


def _queue_indexing(link_id: str, takeover: bool = False) -> bool:
    """Puts the link in the indexing queue unless it is queued or being built
    already. A build whose heartbeat stopped long ago was cut off and is taken
    over; `takeover` (startup) also takes queued and running ones, since a new
    process has none of its own yet."""
    rows = _run_sql(
        """
        UPDATE public_links
        SET index_status = 'queued', index_stage = NULL, index_progress = NULL, index_error = NULL
        WHERE link_id = %s
          AND (%s
               OR index_status NOT IN ('queued', 'indexing')
               OR (index_status = 'indexing'
                   AND (index_heartbeat_at IS NULL
                        OR index_heartbeat_at < now() - make_interval(mins => %s))))
        RETURNING link_id
        """,
        (link_id, takeover, config.public_link_stale_indexing_minutes),
    )
    return bool(rows)


def start_indexing(link_id: str, items: Optional[List[Dict[str, str]]] = None, takeover: bool = False) -> bool:
    """Builds the link's index in the background, unless it is queued or being built already."""
    if not _queue_indexing(link_id, takeover):
        return False
    _start_index_job(_build_index, link_id, items)
    return True


def _beat(link_id: str, stage: str, progress: Optional[int]) -> bool:
    """Records where the build is, which is also its heartbeat. False when the
    row isn't this build's any more (deleted, or failed by someone else)."""
    if _stopping.is_set():
        raise _Stopped()
    rows = _run_sql(
        """
        UPDATE public_links
        SET index_status = 'indexing', index_stage = %s, index_progress = %s, index_heartbeat_at = now()
        WHERE link_id = %s AND index_status IN ('queued', 'indexing')
        RETURNING link_id
        """,
        (stage, progress, link_id),
    )
    return bool(rows)


def _fail_indexing(link_id: str, reason: str, skipped: Optional[List[Dict[str, str]]] = None) -> None:
    """The build stops here. The index built before, and its file list, stay in use."""
    _run_sql(
        """
        UPDATE public_links
        SET index_status = 'failed', index_error = %s, index_stage = NULL, index_progress = NULL,
            skipped_items = %s::jsonb
        WHERE link_id = %s
        """,
        (reason, json.dumps(skipped or []), link_id),
    )


def _build_index(link_id: str, items: Optional[List[Dict[str, str]]] = None) -> None:
    from services.rag.processor import processor
    from services.storage import supabase as supabase_storage

    if not _beat(link_id, "listing", 0):
        return
    link = _get_link(link_id)
    if not link:
        return
    try:
        if items is None:
            items = asyncio.run(_list_link_items(link))
    except Exception as exc:
        logger.warning("public_links: listing %s failed: %s", link_id, exc)
        _fail_indexing(link_id, UNREACHABLE)
        return

    vector_store, skipped = processor.build_public_link_index(
        {"link_id": link_id, "title": link["title"], "url": link["url"], "items": items},
        on_progress=lambda stage, progress: _beat(link_id, stage, progress),
    )
    if vector_store is None:
        _fail_indexing(link_id, "No file in this link could be read.", skipped)
        return

    _beat(link_id, "saving", 95)
    version = uuid.uuid4().hex[:12]
    index_path = processor.save_public_link_index(link_id, version, vector_store)
    if not index_path:
        _fail_indexing(link_id, "The index couldn't be saved.", skipped)
        return
    if supabase_storage.is_enabled() and not supabase_storage.upload_public_link_index(link_id, version, index_path):
        _fail_indexing(link_id, "The index couldn't be uploaded to storage.", skipped)
        return

    if not _finish_build(link, version, fingerprint_items(items), skipped, items):
        # Deleted while it was being built.
        processor.drop_public_link_index(link_id)
        supabase_storage.delete_public_link_index(link_id)
        return
    # Only now that the database points at the new version are the old ones
    # unused, here (local disk) and in Storage.
    processor.prune_public_link_versions(link_id, keep_version=version)
    if supabase_storage.is_enabled():
        supabase_storage.delete_public_link_index(link_id, keep_version=version)
    logger.info("public_links: %s indexed (version %s, %d skipped)", link_id, version, len(skipped))


def _finish_build(link: Dict[str, Any], version: str, fingerprint: str,
                  skipped: List[Dict[str, str]], items: List[Dict[str, str]]) -> bool:
    """Switches the link to the new index and, for a folder, its file list as
    listed for this build, together. A link's first index also activates it
    (it is created inactive); later builds leave the user's choice alone.
    False when the link was deleted meanwhile."""
    conn = db.get_conn("public_links")
    if not conn:
        raise RuntimeError("Database unavailable")
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public_links
                SET index_status = 'ready', index_stage = NULL, index_progress = 100, index_error = NULL,
                    status = CASE WHEN index_version IS NULL THEN 'active' ELSE status END,
                    index_version = %s, fingerprint = %s, skipped_items = %s::jsonb,
                    last_synced_at = now()
                WHERE link_id = %s
                RETURNING link_id
                """,
                (version, fingerprint, json.dumps(skipped), link["link_id"]),
            )
            if not cur.fetchone():
                return False
            if _is_folder_link(link):
                # Sources then lists the files that were indexed.
                _replace_items(cur, link["link_id"], items)
        conn.commit()
        return True
    finally:
        conn.close()


def start_pending_indexing() -> int:
    """At startup: queue every link never indexed (added before MS-907), and
    every build the previous process had queued or running, which it lost.
    A failed build waits for a Sync. Returns how many were queued."""
    try:
        rows = _run_sql(
            """
            SELECT link_id FROM public_links
            WHERE index_status IN ('pending', 'queued', 'indexing')
            ORDER BY created_at
            """
        )
    except Exception as exc:
        logger.warning("public_links: pending index scan failed: %s", exc)
        return 0
    started = sum(1 for row in rows if start_indexing(row["link_id"], takeover=True))
    if started:
        logger.info("public_links: queued %d link(s) for indexing", started)
    return started


async def sync_public_link(link_id: str, user: UserRecord) -> PublicLinkSyncResponse:
    """Lists the link again and rebuilds its index only if its files changed. A
    link that isn't a folder (one file, one page) has no file list that could
    change, so its Sync always rebuilds it, which is one download."""
    link = await asyncio.to_thread(_get_owned_link, link_id, user)
    if link["index_status"] in ("queued", "indexing") and not _is_stale(link):
        return PublicLinkSyncResponse(result="indexing", link=_as_public_link_source(link))
    try:
        items = await _list_link_items(link)
    except Exception as exc:
        logger.warning("public_links: sync listing %s failed: %s", link_id, exc)
        raise HTTPException(status_code=502, detail=UNREACHABLE)

    unchanged = (
        _is_folder_link(link)
        and link.get("index_version")
        and link.get("index_status") == "ready"
        and fingerprint_items(items) == link.get("fingerprint")
    )
    if unchanged:
        await asyncio.to_thread(
            _run_sql,
            "UPDATE public_links SET last_synced_at = now() WHERE link_id = %s",
            (link_id,),
        )
        result = "up_to_date"
    else:
        await asyncio.to_thread(start_indexing, link_id, items)
        result = "indexing"
    return PublicLinkSyncResponse(result=result, link=_as_public_link_source(await asyncio.to_thread(_get_link, link_id)))


def _is_stale(link: Dict[str, Any]) -> bool:
    """A build that stopped reporting: cut off by a restart or another instance
    going away. A queued one is stale only once taken over at startup."""
    if link.get("index_status") != "indexing":
        return False
    beat = link.get("index_heartbeat_at")
    if not beat:
        return True
    if isinstance(beat, str):
        beat = datetime.fromisoformat(beat)
    return datetime.now(timezone.utc) - beat > timedelta(minutes=config.public_link_stale_indexing_minutes)


def _get_owned_link(link_id: str, user: UserRecord) -> Dict[str, Any]:
    link = _get_link(link_id)
    if not link or (user.role != "admin" and link["user_id"] != user.user_id):
        raise HTTPException(status_code=404, detail="Public link not found")
    return link
