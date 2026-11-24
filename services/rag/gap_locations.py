"""MS-1112: the parts of a gap check that need no LLM: stable requirement
keys, structured location pointers into the checked documents, and how an
item's status moved since the previous run. Pure functions, so they are
tested directly (tests/test_gap_locations.py)."""
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# The five statuses a user sees, plus not_checked: a requirement whose kind of
# document was not among the selected ones. It is stored but never counted in
# the summary and never shown as missing.
STATUSES = ("done", "incomplete", "missing", "needs_decision", "not_applicable")
NOT_CHECKED = "not_checked"
ALL_STATUSES = STATUSES + (NOT_CHECKED,)

# Runs stored before MS-1112 used four statuses.
LEGACY_STATUS = {"met": "done", "partial": "incomplete", "not_met": "missing", "unknown": "needs_decision"}

FINDING_VERDICTS = ("met", "partial", "missing")
NEXT_STEP_ACTIONS = ("upload_document", "revise_document", "add_evidence", "decide_status")

# How far an item is from done, for the trend. needs_decision sits with
# incomplete: something is there, but it is not settled yet.
_PROGRESS = {"missing": 0, "incomplete": 1, "needs_decision": 1, "done": 2, "not_applicable": 2}

# "A.5.1 Policies", "4.2.1 Risk treatment", "CC6.1 Logical access", "PS-01 Backup"
_CODE = re.compile(r"^\s*((?:[A-Z]{1,4}[.\-]?)?\d+(?:\.\d+)*[a-z]?)\s*[-–:.)]?\s+(\S.*)$")


def normalize_status(status: Any) -> str:
    value = str(status or "").strip().lower().replace(" ", "_")
    value = LEGACY_STATUS.get(value, value)
    return value if value in ALL_STATUSES else "needs_decision"


def split_code(label: str) -> Tuple[Optional[str], str]:
    """("A.5.1", "Policies") from "A.5.1 Policies"; (None, label) when the
    requirement has no code of its own."""
    match = _CODE.match(label or "")
    if not match:
        return None, (label or "").strip()
    return match.group(1), match.group(2).strip()


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:60].rstrip("-") or "requirement"


def build_requirements(raw: Iterable[Any]) -> List[Dict[str, Any]]:
    """The requirement list of a rule set, from what the LLM extracted: each
    a string or {"label", "expected_document"}. Each gets a key that stays the
    same across runs: its code when it has one ("A.5.1"), else a slug of its
    name, with "-2", "-3" added when two would collide. Duplicates are dropped."""
    requirements: List[Dict[str, Any]] = []
    taken = set()
    seen_labels = set()
    for entry in raw:
        if isinstance(entry, dict):
            label = str(entry.get("label") or entry.get("name") or "").strip()
            expected = str(entry.get("expected_document") or "").strip() or None
        else:
            label, expected = str(entry).strip(), None
        if not label or label.lower() in seen_labels:
            continue
        seen_labels.add(label.lower())
        code, title = split_code(label)
        base = code.upper() if code else _slug(title)
        key, n = base, 1
        while key in taken:
            n += 1
            key = f"{base}-{n}"
        taken.add(key)
        requirements.append({"key": key, "code": code, "label": title, "expected_document": expected})
    return requirements


def trend(previous_status: Optional[str], status: str) -> Optional[str]:
    """"improved" / "worse" / "unchanged" against the previous run, or None
    when there is nothing to compare (first run, or either side not checked)."""
    if previous_status is None:
        return None
    before, after = _PROGRESS.get(normalize_status(previous_status)), _PROGRESS.get(status)
    if before is None or after is None:
        return None
    if after > before:
        return "improved"
    if after < before:
        return "worse"
    return "unchanged"


# ---------------------------------------------------------------------------
# Location pointers
# ---------------------------------------------------------------------------

def file_kind(file_name: str) -> str:
    ext = os.path.splitext(file_name or "")[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext in (".docx", ".doc"):
        return "docx"
    if ext in (".xlsx", ".xls", ".csv"):
        return "xlsx"
    if ext == ".pptx":
        return "slide"
    return "file"


def pointer_label(pointer: Dict[str, Any]) -> str:
    """How a pointer reads to a person: "Backup.docx · Section 4. Restore, paragraph 2"."""
    name = pointer.get("file_name") or ""
    kind = pointer.get("kind")
    if kind == "pdf" and pointer.get("page"):
        return f"{name} · p. {pointer['page']}"
    if kind == "slide" and pointer.get("page"):
        return f"{name} · slide {pointer['page']}"
    if kind == "docx" and pointer.get("paragraph"):
        section = pointer.get("section")
        where = f"Section {section}, paragraph" if section else "Paragraph"
        return f"{name} · {where} {pointer['paragraph']}"
    if kind == "xlsx" and pointer.get("row"):
        sheet = f"{pointer['sheet']}, row" if pointer.get("sheet") else "Row"
        return f"{name} · {sheet} {pointer['row']}"
    return name


def _words(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _norm(text: str) -> str:
    return " ".join(_words(text))


# A quote shorter than this ("backup", "IT") appears in too many paragraphs to
# say which one it came from; the passage the AI read decides instead.
_MIN_QUOTE_WORDS = 3
# Paragraphs shorter than this ("Yes", "N/A") sit inside almost any passage.
_MIN_BLOCK_CHARS = 20


def find_block(blocks: List[Dict[str, Any]], quote: str, passage: str = "") -> Optional[Dict[str, Any]]:
    """The structured block (DOCX paragraph or XLSX row) a finding came from,
    or None (the pointer then names the file only). In order of trust:

    1. a block holding the whole quote; when several do, the one inside the
       passage the AI read;
    2. a block sharing at least half of the quote's words;
    3. the block the passage itself comes from: the passage sits inside one
       long block, or the first sizeable block sits inside the passage.

    Steps 1 and 2 need a quote of _MIN_QUOTE_WORDS words or more."""
    texts = [_norm(block.get("text", "")) for block in blocks]
    quote_words = _words(quote)
    passage_text = _norm(passage)
    in_passage = [bool(passage_text) and len(t) > _MIN_BLOCK_CHARS and t in passage_text for t in texts]

    if len(quote_words) >= _MIN_QUOTE_WORDS:
        needle = " ".join(quote_words)
        holding = [i for i, t in enumerate(texts) if needle in t]
        if holding:
            return blocks[next((i for i in holding if in_passage[i]), holding[0])]
        wanted = set(quote_words)
        overlap = [len(wanted & set(t.split())) / len(wanted) for t in texts]
        best = max(range(len(blocks)), key=overlap.__getitem__, default=None)
        if best is not None and overlap[best] >= 0.5:
            return blocks[best]

    if passage_text:
        around = next((i for i, t in enumerate(texts) if passage_text in t), None)
        if around is not None:
            return blocks[around]
        inside = next((i for i, flag in enumerate(in_passage) if flag), None)
        if inside is not None:
            return blocks[inside]
    return None


def make_pointer(
    file_name: str,
    collection_id: Optional[str],
    page: Any = None,
    blocks: Optional[List[Dict[str, Any]]] = None,
    quote: str = "",
    passage: str = "",
) -> Dict[str, Any]:
    """A structured pointer into one document. PDF and slide decks take the
    page the retrieved chunk was indexed with; Word and Excel are re-read
    (`blocks`, from extract_docx_blocks / extract_xlsx_blocks) and the quote
    is matched to its paragraph or row."""
    kind = file_kind(file_name)
    pointer: Dict[str, Any] = {
        "file_name": file_name, "collection_id": collection_id, "kind": kind,
        "page": None, "section": None, "section_index": None, "paragraph": None, "sheet": None, "row": None,
    }
    if kind in ("pdf", "slide"):
        try:
            pointer["page"] = int(page) if page not in (None, "") else None
        except (TypeError, ValueError):
            pointer["page"] = None
    elif kind in ("docx", "xlsx") and blocks:
        block = find_block(blocks, quote, passage)
        if block:
            for field in ("section", "section_index", "paragraph", "sheet", "row"):
                if field in block:
                    pointer[field] = block[field]
    pointer["label"] = pointer_label(pointer)
    return pointer


def searched_label(searched: Dict[str, Any]) -> str:
    """"Searched 2 folders (5 documents)" for a missing item."""
    folders = searched.get("folders") or []
    count = searched.get("document_count") or 0
    docs = f"{count} document{'s' if count != 1 else ''}"
    if folders:
        return f"Searched {', '.join(folders)} ({docs})"
    return f"Searched {docs}"


# ---------------------------------------------------------------------------
# One checked item, from the LLM's answer to what the API stores
# ---------------------------------------------------------------------------

_DEFAULT_REASON = {
    "done": "The documents cover this requirement.",
    "incomplete": "The documents cover part of this requirement.",
    "missing": "No document covers this requirement yet.",
    "needs_decision": "The AI could not settle this from the documents.",
    "not_applicable": "The documents say this requirement does not apply.",
    NOT_CHECKED: "None of the selected documents is the kind this requirement needs.",
}

_DEFAULT_ACTION = {
    "missing": "upload_document",
    "incomplete": "revise_document",
    "needs_decision": "decide_status",
}


def _first_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    match = re.match(r"(.+?[.!?])(\s|$)", text)
    return (match.group(1) if match else text)[:240]


def finalize_item(
    requirement: Dict[str, Any],
    verdict: Optional[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    searched: Dict[str, Any],
    load_blocks: Callable[[Optional[str], str], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Turn the LLM's verdict on one requirement into a stored item, keeping
    the rules the AI can't be trusted to keep on its own:

    - every status except missing has at least one finding with a pointer;
      a verdict with nothing retrieved to point at becomes missing;
    - every missing item says where the AI searched;
    - not_checked items carry no findings and no next step.

    `verdict` is None when the LLM gave no answer for the requirement (it
    skipped it, or the call failed): see unanswered_item. `sources` are the
    passages it was shown ({"id": "S1", "file_name", "collection_id", "page",
    "text"}); `load_blocks(collection_id, file_name)` gives a Word or Excel
    file's paragraphs or rows, for exact pointers."""
    if not isinstance(verdict, dict):
        return unanswered_item(requirement)
    status = normalize_status(verdict.get("status"))
    pointer_to = _pointer_finder(sources, load_blocks)
    findings = [] if status == NOT_CHECKED else _findings(verdict, pointer_to)

    if status in STATUSES and status != "missing" and not any(f["location"] for f in findings):
        if sources:
            _point_at_first_source(findings, status, pointer_to(sources[0]["id"], ""))
        else:
            # Nothing was retrieved, so whatever the AI said rests on nothing.
            status, verdict = "missing", None

    reason = _first_sentence(str((verdict or {}).get("reason") or "")) or _DEFAULT_REASON[status]
    summary = re.sub(r"\s+", " ", str((verdict or {}).get("summary") or "")).strip() or reason
    if status == "missing":
        _say_where_searched(findings, reason, searched_label(searched))
    next_step = _next_step(status, verdict)

    first_location = next((f["location"] for f in findings if f["location"]), None)
    return {
        "requirement_key": requirement["key"],
        "code": requirement.get("code"),
        "label": requirement["label"],
        "status": status,
        "answered": True,
        "reason": reason,
        "summary": summary,
        "next_step": next_step,
        "findings": findings,
        "searched": dict(searched) if status == "missing" else None,
        # Pre-MS-1112 fields, still read by the downloaded report.
        "evidence": next((f["quote"] for f in findings if f["quote"]), None),
        "source_citation": first_location["label"] if first_location else None,
        "recommendation": next_step["explain"] if next_step else None,
        "target_collection_id": first_location["collection_id"] if first_location else None,
    }


_LEGACY_FINDING_VERDICT = {"done": "met", "not_applicable": "met", "missing": "missing"}


def with_legacy_details(item: Dict[str, Any]) -> Dict[str, Any]:
    """An item stored before MS-1112 has only evidence, a citation string and
    a recommendation. Shown as a finding (its citation as the text, since it
    names no exact place), a reason and a next step, so old runs keep their
    details on the new results page. Items with findings are returned as is."""
    if item.get("findings") or not (item.get("evidence") or item.get("source_citation") or item.get("recommendation")):
        return item
    item = dict(item)
    status = item.get("status")
    evidence, citation = item.get("evidence"), item.get("source_citation")
    if evidence or citation:
        text = f"{evidence} ({citation})" if evidence and citation else (evidence or citation)
        item["findings"] = [_finding(_LEGACY_FINDING_VERDICT.get(status, "partial"), text, quote=evidence)]
    item["reason"] = item.get("reason") or _first_sentence(evidence or item.get("recommendation") or "") or None
    item["summary"] = item.get("summary") or evidence
    if not item.get("next_step") and item.get("recommendation") and status in _DEFAULT_ACTION:
        item["next_step"] = {"action": _DEFAULT_ACTION[status], "explain": _first_sentence(item["recommendation"])}
    return item


UNANSWERED_REASON ="The AI gave no answer for this requirement in this run. Check again to get one."


def unanswered_item(requirement: Dict[str, Any]) -> Dict[str, Any]:
    """A requirement the AI gave no answer for (its batch failed, or the reply
    left it out). It is not a finding about the documents, so it gets no
    status of its own and no made-up pointer: it waits for a person or the
    next run as needs_decision, and `answered: False` keeps it out of trends,
    both this run's and the next run's baseline."""
    return {
        "requirement_key": requirement["key"],
        "code": requirement.get("code"),
        "label": requirement["label"],
        "status": "needs_decision",
        "answered": False,
        "reason": UNANSWERED_REASON,
        "summary": UNANSWERED_REASON,
        "next_step": None,
        "findings": [],
        "searched": None,
        "evidence": None,
        "source_citation": None,
        "recommendation": None,
        "target_collection_id": None,
    }


def _finding(verdict: str, text: str, quote=None, location=None, suggested_place=None, searched=None) -> Dict[str, Any]:
    return {"verdict": verdict, "text": text, "quote": quote, "location": location,
            "suggested_place": suggested_place, "searched": searched}


def _pointer_finder(sources: List[Dict[str, Any]], load_blocks):
    """pointer_to(source_id, quote): the pointer for a passage the LLM cited,
    or None when it cited a passage it was not shown."""
    by_id = {s["id"]: s for s in sources}

    def pointer_to(source_id: str, quote: str) -> Optional[Dict[str, Any]]:
        source = by_id.get(source_id)
        if not source:
            return None
        name, collection_id = source["file_name"], source.get("collection_id")
        blocks = load_blocks(collection_id, name) if file_kind(name) in ("docx", "xlsx") else None
        return make_pointer(name, collection_id, source.get("page"), blocks, quote, source.get("text", ""))

    return pointer_to


def _findings(verdict: Optional[Dict[str, Any]], pointer_to) -> List[Dict[str, Any]]:
    """The LLM's findings, each pointed at the passage it cites ("S2", "[S2]")."""
    findings = []
    for raw in (verdict or {}).get("findings") or []:
        text = str(raw.get("text") or "").strip() if isinstance(raw, dict) else ""
        if not text:
            continue
        kind = str(raw.get("verdict") or "").strip().lower()
        quote = str(raw.get("quote") or "").strip()
        source_id = str(raw.get("source") or "").strip().strip("[]").upper()
        findings.append(_finding(
            verdict=kind if kind in FINDING_VERDICTS else "partial",
            text=text,
            quote=quote or None,
            location=pointer_to(source_id, quote),
            suggested_place=str(raw.get("suggested_place") or "").strip() or None,
        ))
    return findings


def _point_at_first_source(findings: List[Dict[str, Any]], status: str, pointer: Dict[str, Any]) -> None:
    """A status other than missing whose findings name no passage points at
    the best passage the AI was shown: its first finding that is not about
    something missing gets it, else a finding is added for it."""
    target = next((f for f in findings if f["verdict"] != "missing"), None)
    if target:
        target["location"] = pointer
    else:
        verdict = "met" if status in ("done", "not_applicable") else "partial"
        findings.insert(0, _finding(verdict, "Where the AI read this", location=pointer))


def _say_where_searched(findings: List[Dict[str, Any]], reason: str, where: str) -> None:
    """Each missing finding with no place to show says where the AI looked; a
    missing item with no missing finding gets one."""
    for f in findings:
        if f["verdict"] == "missing" and not f["location"] and not f["suggested_place"]:
            f["searched"] = where
    if not any(f["verdict"] == "missing" for f in findings):
        findings.append(_finding("missing", reason, searched=where))


def default_next_step(status: str) -> Optional[Dict[str, Any]]:
    """The usual next step for a status (upload / revise / decide), or None
    for one that needs nothing done: what a corrected item shows (MS-1115)."""
    action = _DEFAULT_ACTION.get(status)
    return {"action": action, "explain": None} if action else None


def _next_step(status: str, verdict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """What to do about a missing, incomplete or undecided item: the LLM's
    suggestion when it is one of the known actions, else the usual one."""
    if status not in _DEFAULT_ACTION:
        return None
    step = (verdict or {}).get("next_step")
    step = step if isinstance(step, dict) else {}
    action = str(step.get("action") or "").strip().lower()
    return {
        "action": action if action in NEXT_STEP_ACTIONS else _DEFAULT_ACTION[status],
        "explain": _first_sentence(str(step.get("explain") or "")) or None,
    }
