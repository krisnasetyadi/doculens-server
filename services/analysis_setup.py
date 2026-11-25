"""
MS-1114: starting an analysis from one sentence. The logic behind the
"New analysis" drawer's first steps, before services/compliance.py runs the
check.

The story:
- suggest: from "What do you want to check?", the AI picks the rule file
  (the document holding the requirements, such as an SoA) and the folders
  holding the documents to check, out of what the user can open. When the
  AI can't be reached, word overlap picks them instead.
- draft: the requirement list the check will use, for the user to read
  first. A spreadsheet is read row by row once its columns are mapped (the
  AI proposes the mapping, the user can change it), so nothing in it is
  guessed; any other document is read by the AI. Every requirement points
  at where it came from ("SoA · row 12", "ISO 27001.pdf · p. 4").
- reusable_analyses: rule sets an earlier analysis already holds, so a new
  check can reuse one instead of drafting again.
"""
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from models import (
    FolderOption,
    ReusableAnalysis,
    RuleFileOption,
    SetupDraft,
    SetupDraftRequest,
    SetupSuggestion,
    SheetPreview,
    TableMapping,
    UserRecord,
)
from services.billing.enforcement import enforce_gap_check_plan, enforce_member_allocation, enforce_plan_limit
from services.billing.inflight import begin_in_flight, end_in_flight, get_workspace_lock, in_flight_tokens, resolve_workspace_id
from services.billing.ledger import enforce_rate_limit, log_token_usage
from services.ingest.requirement_table import apply_mapping, guess_mapping, read_sheets, sheet_preview, valid_mapping
from services.lens_guide.scope import Scope, load_scope
from services.rag.conversation.language import detect_reply_language
from services.rag.gap_check import _read_document_blocks
from services.rag.processor import processor
from services.rag.shared import _extract_total_tokens
from services.storage import supabase as supabase_storage

logger = logging.getLogger(__name__)

# Rule files and folders listed for the AI: enough for a real workspace,
# bounded so the prompt stays small.
MAX_RULE_FILES = 60
MAX_FOLDERS = 80
RULE_FILE_TYPES = (".xlsx", ".pdf", ".docx", ".doc", ".csv", ".txt", ".pptx")
TABLE_TYPES = (".xlsx",)


# ---------------------------------------------------------------------------
# Suggesting the rule file and the folders
# ---------------------------------------------------------------------------

def suggest(goal: str, user: UserRecord) -> Tuple[SetupSuggestion, int]:
    """(suggestion, tokens used). Blocking: call through asyncio.to_thread.
    Switched-off documents count too: a rule file is often kept out of chat."""
    scope = load_scope(user, active_only=False)
    rule_files = rule_file_options(scope)
    folders = folder_options(scope)
    picked, tokens = _ask_ai_to_pick(goal, rule_files, folders)
    suggested_by = "ai" if picked else "guess"
    picked = picked or guess_pick(goal, rule_files, folders)

    rule_file = next((f for f in rule_files if f.collection_id == picked.get("rule_file")), None)
    chosen = set(picked.get("folders") or [])
    for folder in folders:
        folder.suggested = (folder.folder_id or "") in chosen
    return SetupSuggestion(
        rule_file=rule_file,
        rule_file_why=picked.get("rule_file_why") if rule_file else None,
        rule_files=rule_files,
        folders=folders,
        folders_why=picked.get("folders_why") if chosen else None,
        suggested_by=suggested_by,
    ), tokens


def rule_file_options(scope: Scope) -> List[RuleFileOption]:
    """Documents that could hold a requirement list: one file of a type the
    check can read each, the ones marked as a reference framework in Sources
    first."""
    flags = supabase_storage.collection_reference_flags([c.collection_id for c in scope.collections])
    options = []
    for c in scope.collections:
        row = flags.get(c.collection_id) or {}
        file_name = next(iter(row.get("file_names") or []), c.title)
        if not file_name.lower().endswith(RULE_FILE_TYPES):
            continue
        options.append(RuleFileOption(
            collection_id=c.collection_id, file_name=file_name, title=c.title,
            flagged=bool(row.get("is_reference_framework")), folder_path=c.folder_path,
        ))
    options.sort(key=lambda o: not o.flagged)  # stable: newest first within each group
    return options[:MAX_RULE_FILES]


def folder_options(scope: Scope) -> List[FolderOption]:
    """Every folder holding documents, with the documents in it and its
    subfolders, plus one entry for documents that are in no folder."""
    children: Dict[str, List[str]] = {}
    for f in scope.folders:
        if f.parent_folder_id:
            children.setdefault(f.parent_folder_id, []).append(f.folder_id)
    direct: Dict[Optional[str], List[str]] = {}
    for c in scope.collections:
        direct.setdefault(c.folder_id, []).append(c.collection_id)

    def documents_under(folder_id: str) -> List[str]:
        ids, stack, seen = [], [folder_id], set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            ids.extend(direct.get(current, []))
            stack.extend(children.get(current, []))
        return ids

    options = []
    for f in sorted(scope.folders, key=lambda f: f.path):
        ids = documents_under(f.folder_id)
        if ids:
            options.append(FolderOption(folder_id=f.folder_id, name=f.name, path=f.path,
                                        document_count=len(ids), collection_ids=ids))
    loose = direct.get(None, [])
    if loose:
        options.append(FolderOption(folder_id=None, name="Not in a folder", path=[],
                                    document_count=len(loose), collection_ids=loose))
    return options[:MAX_FOLDERS]


def _ask_ai_to_pick(goal: str, rule_files: List[RuleFileOption], folders: List[FolderOption]) -> Tuple[Optional[Dict[str, Any]], int]:
    """The AI's pick as {"rule_file": collection id, "folders": [folder ids,
    "" for no folder], and a one-line why for each}, or None when it gave no
    usable answer. Ids it was not shown are dropped."""
    if not rule_files and not folders:
        return None, 0
    file_ids = {f"R{i + 1}": f.collection_id for i, f in enumerate(rule_files)}
    folder_ids = {f"F{i + 1}": f.folder_id or "" for i, f in enumerate(folders)}
    files_text = "\n".join(
        f"[R{i + 1}] {'* ' if f.flagged else ''}{f.file_name}" + (f" (folder: {' / '.join(f.folder_path)})" if f.folder_path else "")
        for i, f in enumerate(rule_files)
    ) or "(none)"
    folders_text = "\n".join(
        f"[F{i + 1}] {' / '.join(f.path) or f.name} ({f.document_count} documents)" for i, f in enumerate(folders)
    ) or "(none)"
    # The goal's language, decided here: left to the model, folder and file names in another
    # language pull its sentences into that one.
    language_name = "Indonesian" if detect_reply_language(goal) == "id" else "English"
    prompt = f"""A user wants to check their company documents. In their words: "{goal}"

RULE FILES (documents that may hold the list of requirements to check against; * = marked as a rule file by the user):
{files_text}

FOLDERS (where the company documents are):
{folders_text}

Pick the ONE rule file the documents should be compared with, and the folders holding the documents to check.
Leave out the folder that only holds rule files. Write each "why" as one short sentence in {language_name}.
Reply with ONLY JSON, no markdown:
{{"rule_file": "R1" or null, "rule_file_why": "...", "folders": ["F1"], "folders_why": "..."}}"""
    try:
        llm, _ = processor.get_llm()
        result = llm.invoke(prompt)
        tokens = _extract_total_tokens(result)
        answer = parse_json_object(result.content if hasattr(result, "content") else str(result))
    except Exception as e:
        logger.warning("analysis setup: suggestion failed: %s", e)
        return None, 0
    if not answer:
        return None, tokens
    return {
        "rule_file": file_ids.get(str(answer.get("rule_file") or "")),
        "rule_file_why": _sentence(answer.get("rule_file_why")),
        "folders": [folder_ids[f] for f in answer.get("folders") or [] if f in folder_ids],
        "folders_why": _sentence(answer.get("folders_why")),
    }, tokens


def guess_pick(goal: str, rule_files: List[RuleFileOption], folders: List[FolderOption]) -> Dict[str, Any]:
    """Without the AI: the rule file sharing the most words with the goal (a
    marked one wins a tie, and is taken when nothing matches), and the
    folders sharing any word with it."""
    words = _words(goal)

    def overlap(text: str) -> int:
        return len(words & _words(text))

    rule_file = max(rule_files, key=lambda f: (overlap(f.file_name + " " + f.title), f.flagged), default=None)
    if rule_file and not overlap(rule_file.file_name + " " + rule_file.title) and not rule_file.flagged:
        rule_file = None
    picked_folders = [f.folder_id or "" for f in folders if overlap(" ".join(f.path) or f.name)]
    return {
        "rule_file": rule_file.collection_id if rule_file else None,
        "rule_file_why": "Its name matches what you want to check." if rule_file else None,
        "folders": picked_folders,
        "folders_why": "Their names match what you want to check." if picked_folders else None,
    }


_STOP_WORDS = {"the", "and", "for", "our", "are", "with", "what", "which", "that", "this", "documents", "document",
               "yang", "dan", "untuk", "apakah", "sudah", "dengan", "kami", "dokumen", "check", "cek"}


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2 and w not in _STOP_WORDS}


def _sentence(value: Any) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:240] or None


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """The first JSON object in a model reply (a markdown fence or text around
    it is ignored), or None."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# Drafting the requirement list
# ---------------------------------------------------------------------------

def draft(body: SetupDraftRequest, user: UserRecord) -> Tuple[SetupDraft, int]:
    """(draft, tokens used). Blocking: call through asyncio.to_thread. The
    rule file must be one the user may open."""
    allowed = supabase_storage.list_collection_ids_for_user(user.user_id, user.role == "admin")
    if body.collection_id not in allowed:
        raise HTTPException(status_code=403, detail="That document is not accessible to you")
    collection = supabase_storage.get_collection(body.collection_id)
    file_names = (collection or {}).get("file_names") or []
    file_name = body.file_name or next(iter(file_names), None)
    if not collection or not file_name or file_name not in file_names:
        raise HTTPException(status_code=404, detail="That file is not in the chosen document")

    framework_name = collection.get("title") or os.path.splitext(file_name)[0]
    if file_name.lower().endswith(TABLE_TYPES):
        return _draft_from_table(body, file_name, framework_name)
    return _draft_from_document(body, file_name, framework_name)


def _draft_from_table(body: SetupDraftRequest, file_name: str, framework_name: str) -> Tuple[SetupDraft, int]:
    sheets = _read_document_blocks(body.collection_id, file_name, read_sheets)
    previews = [SheetPreview(**p) for p in sheet_preview(sheets)]
    base = {"collection_id": body.collection_id, "file_name": file_name, "framework_name": framework_name,
            "kind": "table", "sheets": previews}
    if not sheets:
        raise HTTPException(status_code=422, detail="No rows could be read from this spreadsheet")

    tokens = 0
    if body.mapping:
        mapping, mapped_by = body.mapping.model_dump(), "user"
        if not valid_mapping(sheets, mapping):
            raise HTTPException(status_code=400, detail="Choose a sheet, header row and columns that exist in this file")
    else:
        mapping, tokens = _ask_ai_to_map(body.goal, sheets)
        mapped_by = "ai" if mapping else None
        if not mapping:
            mapping = guess_mapping(sheets)
            mapped_by = "guess" if mapping else None
    if not mapping:
        return SetupDraft(**base), tokens  # the user maps the columns

    requirements = apply_mapping(sheets, mapping, os.path.splitext(file_name)[0])
    return SetupDraft(**base, mapping=TableMapping(**mapping), mapped_by=mapped_by, requirements=requirements), tokens


def _ask_ai_to_map(goal: str, sheets: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int]:
    """The AI's column mapping from each sheet's first rows, kept only when it
    points at a sheet, row and columns that exist."""
    lines = []
    for p in sheet_preview(sheets):
        lines.append(f'Sheet "{p["name"]}" ({p["row_count"]} rows):')
        for row in p["rows"]:
            cells = " ".join(f'{p["columns"][i]}="{c[:60]}"' for i, c in enumerate(row["cells"]) if c)
            lines.append(f'  row {row["row"]}: {cells}')
    prompt = f"""The user wants to check: "{goal or 'their documents against a list of requirements'}".
This spreadsheet should hold that list of requirements. Its sheets, with their first rows:
{chr(10).join(lines)}

Which sheet holds the requirements, which row holds its column headers, and which columns hold each
requirement's number or code (if any), its name, and whether it applies (if any)?
Reply with ONLY JSON, no markdown, columns as letters:
{{"sheet": "...", "header_row": 3, "code": "A" or null, "name": "B", "applicable": "C" or null}}"""
    try:
        llm, _ = processor.get_llm()
        result = llm.invoke(prompt)
        tokens = _extract_total_tokens(result)
        answer = parse_json_object(result.content if hasattr(result, "content") else str(result))
    except Exception as e:
        logger.warning("analysis setup: column mapping failed: %s", e)
        return None, 0
    if not answer:
        return None, tokens
    try:
        mapping = TableMapping(
            sheet=str(answer.get("sheet") or ""), header_row=int(answer.get("header_row") or 0),
            code=_letter(answer.get("code")), name=_letter(answer.get("name")) or "",
            applicable=_letter(answer.get("applicable")),
        ).model_dump()
    except (ValueError, TypeError):
        return None, tokens
    return (mapping if valid_mapping(sheets, mapping) else None), tokens


def _letter(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper()
    return text if re.fullmatch(r"[A-Z]{1,3}", text) else None


def _draft_from_document(body: SetupDraftRequest, file_name: str, framework_name: str) -> Tuple[SetupDraft, int]:
    """A PDF, Word or other document: the AI lists its requirements (as a
    first run always did), and each is pointed at the page that names it."""
    language = detect_reply_language(body.goal or framework_name)
    requirements, tokens = processor.extract_framework_items([body.collection_id], framework_name, language)
    pages = _indexed_pages(body.collection_id)
    stem = os.path.splitext(file_name)[0]
    for requirement in requirements:
        page = find_page(pages, requirement.get("code"), requirement["label"])
        requirement["source"] = {"label": f"{stem} · p. {page}", "page": page} if page else {"label": stem}
    return SetupDraft(collection_id=body.collection_id, file_name=file_name, framework_name=framework_name,
                      kind="document", requirements=requirements), tokens


def _indexed_pages(collection_id: str) -> List[Tuple[int, str]]:
    """(page, lower-case text) of the collection's indexed chunks, in page order."""
    store = processor.get_vector_store(collection_id)
    if not store:
        return []
    try:
        docs = list(store.docstore._dict.values())
    except Exception:
        return []
    pages = []
    for doc in docs:
        try:
            page = int(doc.metadata.get("page"))
        except (TypeError, ValueError):
            continue
        pages.append((page, (doc.page_content or "").lower()))
    return sorted(pages, key=lambda p: p[0])


def find_page(pages: List[Tuple[int, str]], code: Optional[str], label: str) -> Optional[int]:
    """The first page naming the requirement: by its code when it has one,
    else by its name."""
    needles = [code.lower()] if code else []
    needles.append((label or "").lower()[:80])
    for needle in needles:
        if len(needle) < 3:
            continue
        for page, text in pages:
            if needle in text:
                return page
    return None


# ---------------------------------------------------------------------------
# Reusing a rule set
# ---------------------------------------------------------------------------

def reusable_analyses(user: UserRecord) -> List[ReusableAnalysis]:
    """Blocking: call through asyncio.to_thread."""
    rows = supabase_storage.list_reusable_analyses(user.user_id, user.role == "admin")
    return [ReusableAnalysis(**{k: v for k, v in row.items() if v is not None}) for row in rows]


# ---------------------------------------------------------------------------
# The endpoints' story: a paid-plan feature metered like the check itself
# (services/compliance._run_metered_gap_check): the rate limit, the workspace
# quota and the member's allocation are checked, and an estimate is held as
# in-flight while the AI works, so repeated clicks can't run past them.
# ---------------------------------------------------------------------------

# Tokens held while each call runs: a suggestion reads file and folder names;
# a draft may read a whole reference document (extract_framework_items).
SUGGEST_TOKEN_RESERVE = 5_000
DRAFT_TOKEN_RESERVE = 60_000


async def suggest_for(goal: str, user: UserRecord) -> SetupSuggestion:
    await asyncio.to_thread(enforce_gap_check_plan, user)
    return await _metered(user, SUGGEST_TOKEN_RESERVE, suggest, goal.strip(), user)


async def draft_for(body: SetupDraftRequest, user: UserRecord) -> SetupDraft:
    await asyncio.to_thread(enforce_gap_check_plan, user)
    return await _metered(user, DRAFT_TOKEN_RESERVE, draft, body, user)


async def reusable_for(user: UserRecord) -> List[ReusableAnalysis]:
    return await asyncio.to_thread(reusable_analyses, user)


async def _metered(user: UserRecord, reserve: int, work, *args):
    """Run `work(*args)` (returning (result, tokens)) off the event loop once
    the limits allow it, then log the tokens it used."""
    workspace_id = await asyncio.to_thread(resolve_workspace_id, user)
    async with get_workspace_lock(workspace_id):
        pending_workspace, pending_user = in_flight_tokens(workspace_id, user.user_id)
        await asyncio.to_thread(enforce_rate_limit, user.user_id, pending_user)
        workspace_cap = await asyncio.to_thread(enforce_plan_limit, user, pending_workspace, reserve)
        user_cap = await asyncio.to_thread(enforce_member_allocation, user, pending_user, reserve)
        reservation = begin_in_flight(workspace_id, user.user_id, reserve, workspace_cap, user_cap)
    try:
        result, tokens = await asyncio.to_thread(work, *args)
        if tokens > 0:
            try:
                await asyncio.to_thread(log_token_usage, user.user_id, tokens)
            except Exception:
                logger.warning("analysis setup: failed to log token usage", exc_info=True)
        return result
    finally:
        end_in_flight(reservation)
