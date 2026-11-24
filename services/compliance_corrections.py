"""
MS-1115: a person correcting a status the AI got wrong, for the gap analysis
in services/compliance.py.

The story of a correction:

- Someone says what was wrong (the AI missed a document, the status is not
  right, or the requirement does not apply to them), the right status and
  why. build_correction turns that into what is stored on the item; the
  item's own `status` keeps the AI's answer.
- Whoever reads the item afterwards sees the corrected status
  (present_item), with the AI's status beside it; summaries, trends and the
  export count the corrected one.
- "Remember for next checks" adds a direction to the analysis's rule set
  (remembered_direction; storage adds it as a new rule-set version).
- A later run of the same analysis keeps the correction while the AI repeats
  the answer that was corrected and no document it relied on has changed
  since (carry_corrections).
- ai_accuracy says how many of the AI's statuses a person changed.
"""
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from models import GapAnalysisCorrectionRequest, UserRecord
from services.rag.gap_locations import NOT_CHECKED, default_next_step, make_pointer
from services.storage import supabase as supabase_storage

# The statuses a person can set. needs_decision is what a correction settles,
# so it is never one.
CORRECTABLE_STATUSES = ("done", "incomplete", "missing", "not_applicable")

# The verdict of the finding a missed document adds, by the corrected status.
_ADDED_FINDING_VERDICT = {"done": "met", "not_applicable": "met", "incomplete": "partial", "missing": "missing"}


# ---------------------------------------------------------------------------
# Making a correction
# ---------------------------------------------------------------------------

def correction_status(body: GapAnalysisCorrectionRequest) -> Optional[str]:
    """The status a correction sets, or None when it is not one a person may
    set. "Not applicable to us" always means not_applicable."""
    if body.kind == "not_applicable":
        return "not_applicable"
    return body.status if body.status in CORRECTABLE_STATUSES else None


def build_correction(
    body: GapAnalysisCorrectionRequest,
    item: Dict[str, Any],
    user: UserRecord,
    direction: Optional[str] = None,
) -> Dict[str, Any]:
    """What is stored for a correction. `basis` records the documents it rests
    on (the missed document, and those the AI's findings point at) as they
    are now, so a later run can tell whether any of them changed."""
    document = None
    if body.kind == "missed_document" and body.document:
        document = make_pointer(body.document.file_name, body.document.collection_id)
        if body.document.section:
            document["section"] = body.document.section
            document["label"] = f"{body.document.file_name} · {body.document.section}"
    return {
        "kind": body.kind,
        "status": correction_status(body),
        # What the AI said when it was corrected: a later run keeps the
        # correction only while the AI still says this (carry_corrections).
        "ai_status": item.get("status"),
        "reason": body.reason,
        "document": document,
        "corrected_by": user.user_id,
        "corrected_by_name": user.name or user.email,
        "corrected_at": datetime.now(timezone.utc).isoformat(),
        "direction": direction,
        "basis": document_basis(_relied_on(item, document)),
    }


def _relied_on(item: Dict[str, Any], document: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    places = [document] if document else []
    places += [f["location"] for f in item.get("findings") or [] if isinstance(f, dict) and f.get("location")]
    return places


def content_version(collection: Dict[str, Any]) -> str:
    """What changes when a collection's content changes: its files, where
    they are stored and how many chunks they were indexed into (an upload
    rewrites all three). Not updated_at, which also moves when a document is
    only renamed into a folder, switched on or off, or flagged as a
    reference."""
    return "|".join([
        ",".join(sorted(collection.get("file_names") or [])),
        ",".join(sorted(collection.get("storage_paths") or [])),
        str(collection.get("chunk_count") or 0),
    ])


def document_basis(places: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Each document named in `places`, once, with its collection's content
    version now. A collection that is gone is left out."""
    basis, seen = [], set()
    for place in places:
        key = (place.get("collection_id"), place.get("file_name"))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        collection = supabase_storage.get_collection(key[0])
        if collection:
            basis.append({"collection_id": key[0], "file_name": key[1], "version": content_version(collection)})
    return basis


def basis_unchanged(basis: List[Dict[str, Any]]) -> bool:
    """True when every document a correction rests on is still there, with
    the content it had when the correction was made."""
    for doc in basis:
        collection = supabase_storage.get_collection(doc["collection_id"])
        if not collection or doc["file_name"] not in (collection.get("file_names") or []):
            return False
        if content_version(collection) != doc.get("version"):
            return False
    return True


# ---------------------------------------------------------------------------
# Remember for next checks
# ---------------------------------------------------------------------------

def remembered_direction(item: Dict[str, Any], body: GapAnalysisCorrectionRequest) -> str:
    """The direction "Remember for next checks" adds, e.g. "A.8.26: the
    requirements are written in MLT-PO-P03 (Section 2), not the PO guide." """
    name = item.get("code") or item.get("label") or "This requirement"
    where = ""
    if body.kind == "missed_document" and body.document:
        section = f" ({body.document.section})" if body.document.section else ""
        where = f" Evidence: {body.document.file_name}{section}."
    if body.kind == "not_applicable":
        return f"{name} does not apply to us: {body.reason}"
    return f"{name}: {body.reason}{where}"


# ---------------------------------------------------------------------------
# A later run
# ---------------------------------------------------------------------------

def carry_corrections(
    items: List[Dict[str, Any]],
    previous: Dict[str, Dict[str, Any]],
    unchanged: Callable[[List[Dict[str, Any]]], bool] = basis_unchanged,
) -> None:
    """Give each new item the correction its requirement had in the previous
    run, as long as the AI repeats the answer that was corrected and no
    document the correction rests on changed. A new answer from the AI (say,
    done after the evidence was uploaded) or a changed document means the
    situation moved on: the AI's new answer stands."""
    for item in items:
        correction = previous.get(item.get("requirement_key"))
        if not correction or item.get("status") == NOT_CHECKED:
            continue
        if item.get("status") != correction.get("ai_status"):
            continue
        if unchanged(correction.get("basis") or []):
            item["correction"] = dict(correction)  # names the run it was made on (get_latest_corrections)


# ---------------------------------------------------------------------------
# Reading an item
# ---------------------------------------------------------------------------

def effective_status(item: Dict[str, Any]) -> str:
    return (item.get("correction") or {}).get("status") or item.get("status")


def present_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """An item as the API returns it: `ai_status` is the AI's answer and
    `status` the one in use. A corrected item reads as its correction: its
    reason, the next step its status calls for, and the missed document
    first among the findings, marked as added by its corrector."""
    item = dict(item)
    item["ai_status"] = item.get("status")
    correction = item.get("correction")
    if not correction:
        return item
    status = correction["status"]
    item["status"] = status
    item["reason"] = item["summary"] = correction.get("reason")
    item["next_step"] = default_next_step(status)
    if correction.get("document"):
        added = {
            "verdict": _ADDED_FINDING_VERDICT.get(status, "partial"),
            "text": correction.get("reason"),
            "quote": None,
            "location": correction["document"],
            "suggested_place": None,
            "searched": None,
            "added_by": correction.get("corrected_by_name"),
        }
        item["findings"] = [added] + list(item.get("findings") or [])
    return item


def ai_accuracy(items: List[Dict[str, Any]]) -> Dict[str, int]:
    """{"checked", "corrected"}: how many items the AI gave a status (not
    not_checked, not left undecided or unanswered), and how many of those a
    person changed to another status. Deciding a needs_decision item is not
    counted against the AI: it never claimed a status."""
    checked = corrected = 0
    for item in items:
        ai = item.get("ai_status") or item.get("status")
        if ai in (NOT_CHECKED, "needs_decision") or item.get("answered") is False:
            continue
        checked += 1
        correction = item.get("correction")
        if correction and correction.get("status") != ai:
            corrected += 1
    return {"checked": checked, "corrected": corrected}
