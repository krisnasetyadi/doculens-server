# services/compliance.py
"""
Generic "Reference Framework Gap Analysis" — the logic behind router/compliance.py.

Skill-agnostic by design: `skill_id` selects the behavior/output schema.
"compliance_gap_check" (Skill 1) is validated first via ISO 27001/9001 —
ISO is just the first `framework_name` a caller happens to use, nothing
here is ISO-specific. "scenario_regulatory_impact" (Skill 2) is scaffolded
so the endpoint/schema shape exists, but is NOT validated for real
decisions yet (see run_gap_analysis below).
"""
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException
from fastapi.responses import Response

from services.storage import supabase as supabase_storage
from config import config
from models import (
    GapAnalysisItem,
    GapAnalysisRequest,
    GapAnalysisResponse,
    GapAnalysisRun,
    GapAnalysisRunListItem,
    UserRecord,
)
from services.rag.processor import processor
from services.rag.gap_locations import NOT_CHECKED, STATUSES, normalize_status, trend
from services.billing.enforcement import enforce_gap_check_plan, enforce_member_allocation, enforce_plan_limit
from services.billing.inflight import begin_in_flight, end_in_flight, get_workspace_lock, in_flight_tokens, resolve_workspace_id
from services.billing.ledger import enforce_rate_limit, log_token_usage

logger = logging.getLogger(__name__)

SUPPORTED_SKILLS = {"compliance_gap_check", "scenario_regulatory_impact"}

SCENARIO_SCAFFOLD_DISCLAIMER = (
    "Skill scenario_regulatory_impact masih scaffold (belum divalidasi untuk keputusan nyata). "
    "Ini BUKAN nasihat pajak/hukum resmi — konsultasikan ke konsultan pajak/akuntan bersertifikat "
    "sebelum mengambil keputusan berdasarkan hasil ini."
)


def _normalize_item(item: dict) -> dict:
    """A stored item as the API returns it: runs stored before MS-1112 carry
    met / partial / not_met / unknown, which read as their new statuses."""
    item = dict(item)
    item["status"] = normalize_status(item.get("status"))
    item["findings"] = item.get("findings") or []
    return item


def _summarize(items: List[dict]) -> dict:
    """Counts per status. not_checked is left out: a requirement whose kind of
    document was not selected says nothing about the documents (MS-1112)."""
    summary = {status: 0 for status in STATUSES}
    for item in items:
        status = item.get("status")
        if status in summary:
            summary[status] += 1
    return summary


def _response(run_row: dict, items: List[dict], disclaimer: Optional[str]) -> GapAnalysisResponse:
    return GapAnalysisResponse(
        run=GapAnalysisRun(**run_row),
        items=[GapAnalysisItem(**it) for it in items],
        summary=_summarize(items),
        not_checked=sum(1 for it in items if it.get("status") == NOT_CHECKED),
        total_requirements=len(items),
        disclaimer=disclaimer,
    )


def _can_access_run(row: Optional[dict], user: UserRecord) -> bool:
    """Admins can access everything; everyone else only their own runs (or analyses)."""
    if user.role == "admin":
        return True
    if not row:
        return False
    return row.get("owner_id") == user.user_id


def _searched(folder_ids: List[str], document_count: int, user: UserRecord) -> dict:
    """Where the AI looked, as a missing item states it: the names of the
    folders the documents were picked from (only ones the user may open), and
    how many documents were checked."""
    names = []
    for folder_id in folder_ids:
        folder = supabase_storage.get_folder(folder_id)
        if folder and (user.role == "admin" or folder.get("owner_id") == user.user_id):
            names.append(folder.get("name") or folder_id)
    return {"folders": names, "document_count": document_count}


# Auditor wording for the downloaded report only; the API keeps the status keys.
EXPORT_STATUS_LABELS = {
    "done": "comply",
    "incomplete": "partially comply",
    "missing": "not comply",
    "needs_decision": "needs decision",
    "not_applicable": "not applicable",
    NOT_CHECKED: "not checked",
}


def _export_status(status: str) -> str:
    status = normalize_status(status)
    return EXPORT_STATUS_LABELS.get(status, status)


def _item_name(item: dict) -> str:
    """"A.8.13 Information backup": MS-1112 keeps the code apart from the name."""
    code, label = item.get("code"), item.get("label", "")
    return f"{code} {label}" if code and not label.startswith(code) else label


def _render_markdown(run_row: dict, items: List[dict]) -> str:
    lines = [
        f"# Laporan Gap Analysis — {run_row.get('framework_name', '')}",
        "",
        f"- Skill: {run_row.get('skill_id', '')}",
        f"- Dibuat: {run_row.get('created_at', '')}",
        f"- Reference collection: {', '.join(run_row.get('reference_collection_ids') or [])}",
        f"- Target collection(s): {', '.join(run_row.get('target_collection_ids') or []) or '-'}",
        "",
        "> Catatan: laporan ini dihasilkan AI berdasarkan dokumen yang diupload user — "
        "bukan audit/sertifikasi resmi maupun nasihat pajak/hukum resmi.",
        "",
        "| Item | File | Status | Evidence | Rekomendasi |",
        "|---|---|---|---|---|",
    ]
    for item in items:
        source = (item.get("source_citation") or item.get("target_collection_id") or "").replace("|", "/")
        evidence = (item.get("evidence") or "").replace("\n", " ").replace("|", "/")
        recommendation = (item.get("recommendation") or "").replace("\n", " ").replace("|", "/")
        lines.append(f"| {_item_name(item)} | {source} | {_export_status(item.get('status', ''))} | {evidence} | {recommendation} |")
    return "\n".join(lines)


def _render_pdf(run_row: dict, items: List[dict], markdown_fallback: str) -> bytes:
    """Render the run as a PDF. Falls back to returning the markdown as raw
    bytes if reportlab isn't installed — library choice is an implementation
    detail, not a blocker for shipping this endpoint (per plan).

    Landscape + Paragraph cells (not raw strings) so Evidence/Rekomendasi —
    which can be several sentences of free-text from the LLM — wrap inside
    their column instead of forcing the table wider than the page or
    overlapping neighboring columns, which is what a plain reportlab Table
    does with unwrapped strings and no column widths."""
    try:
        import io
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from xml.sax.saxutils import escape

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape(A4),
            leftMargin=15 * mm,
            rightMargin=15 * mm,
            topMargin=15 * mm,
            bottomMargin=15 * mm,
        )
        styles = getSampleStyleSheet()
        cell_style = ParagraphStyle("GapCell", parent=styles["Normal"], fontSize=8, leading=10)
        header_style = ParagraphStyle(
            "GapHeader", parent=cell_style, textColor=colors.white, fontName="Helvetica-Bold",
        )

        def cell(text: str, style: ParagraphStyle = cell_style) -> Paragraph:
            # Paragraph markup treats bare "&"/"<"/">" as XML — escape first,
            # then restore line breaks as <br/> so multi-line evidence/
            # recommendation text doesn't collapse onto one line.
            safe = escape(text or "—").replace("\n", "<br/>")
            return Paragraph(safe, style)

        has_multiple_targets = len(run_row.get("target_collection_ids") or []) > 1

        story = [
            Paragraph(f"Laporan Gap Analysis — {run_row.get('framework_name', '')}", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"Dibuat: {run_row.get('created_at', '')}", styles["Normal"]),
            Paragraph(
                "Catatan: laporan ini dihasilkan AI berdasarkan dokumen yang diupload user — "
                "bukan audit/sertifikasi resmi maupun nasihat pajak/hukum resmi.",
                styles["Italic"],
            ),
            Spacer(1, 12),
        ]

        headers = ["Item", "File", "Status", "Evidence", "Rekomendasi"]
        col_widths = [100, 100, 60, 245, 245]  # sums to ~750pt, fits landscape A4 minus margins
        if not has_multiple_targets:
            headers.pop(1)
            col_widths.pop(1)
            col_widths[-2] += 50  # File's freed width goes to the two free-text columns,
            col_widths[-1] += 50  # not Item/Status which don't need it

        table_data = [[cell(h, header_style) for h in headers]]
        for item in items:
            row = [cell(_item_name(item))]
            if has_multiple_targets:
                source = item.get("source_citation") or item.get("target_collection_id") or ""
                row.append(cell(source))
            row.append(cell(_export_status(item.get("status", ""))))
            row.append(cell(item.get("evidence", "")))
            row.append(cell(item.get("recommendation", "")))
            table_data.append(row)

        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E6B4C")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(table)
        doc.build(story)
        return buffer.getvalue()
    except ImportError:
        logger.warning("reportlab not installed — export falling back to markdown bytes for PDF request")
        return markdown_fallback.encode("utf-8")


async def run_gap_analysis(body: GapAnalysisRequest, user: UserRecord):
    """Run a skill and store the result as a run of an analysis (MS-1112).

    The story: open the analysis being run again (if any), settle which rule
    set this run checks, make sure every document is the user's to read, run
    the metered check, file the run under its analysis, record how each
    requirement moved since the last run, and store it."""
    _check_request(body)
    analysis = await _open_analysis(body.analysis_id, user)
    rule_set = _rule_set_for(body, analysis)
    await _check_collection_access(
        user, rule_set.reference_collection_ids + body.target_collection_ids, checking_again=analysis is not None
    )

    if body.skill_id == "scenario_regulatory_impact":
        # Scaffold only: the endpoint works end to end (request validated, run
        # stored) so the structure exists, but it deliberately returns no
        # analysis: this skill is not validated for real tax/legal decisions.
        return await _store_run(body, user, rule_set, [], SCENARIO_SCAFFOLD_DISCLAIMER)

    # Paid-plan feature (a run costs 130k-260k tokens), checked before any LLM
    # work. See services.billing.plans.GAP_CHECK_PLAN_IDS.
    await asyncio.to_thread(enforce_gap_check_plan, user)
    if not body.target_collection_ids:
        raise HTTPException(status_code=400, detail="target_collection_ids is required for compliance_gap_check")

    folder_ids = body.target_folder_ids or list((analysis or {}).get("target_folder_ids") or [])
    searched = await asyncio.to_thread(_searched, folder_ids, len(body.target_collection_ids), user)
    items, disclaimer, requirements = await _run_metered_gap_check(user, rule_set, body.target_collection_ids, searched)

    filed = await _file_under_analysis(body, user, analysis, rule_set, requirements, folder_ids)
    _record_trends(items, filed.previous_statuses)
    return await _store_run(body, user, rule_set, items, disclaimer, filed, searched)


def _check_request(body: GapAnalysisRequest) -> None:
    if body.skill_id not in SUPPORTED_SKILLS:
        raise HTTPException(status_code=400, detail=f"Unknown skill_id. Supported: {sorted(SUPPORTED_SKILLS)}")
    if not body.reference_collection_ids:
        raise HTTPException(status_code=400, detail="reference_collection_ids is required")


async def _open_analysis(analysis_id: Optional[str], user: UserRecord) -> Optional[dict]:
    """The analysis this run belongs to, with its latest rule set; None for a new one."""
    if not analysis_id:
        return None
    analysis = await asyncio.to_thread(supabase_storage.get_analysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if not _can_access_run(analysis, user):
        raise HTTPException(status_code=403, detail="Not allowed to access this analysis")
    return analysis


@dataclass
class RuleSet:
    """What one run checks against: the reference, its requirement list (None
    until it has been read from the reference once) and the user's directions."""
    framework_name: str
    reference_collection_ids: List[str]
    requirements: Optional[List[dict]]
    directions: str


def _rule_set_for(body: GapAnalysisRequest, analysis: Optional[dict]) -> RuleSet:
    """A new analysis takes everything from the request. A run of an existing
    one keeps its reference and requirement list, so items compare by key with
    the last run; directions sent with the request replace the stored ones (and
    become a new rule-set version, see _file_under_analysis)."""
    if not analysis:
        return RuleSet(body.framework_name, list(body.reference_collection_ids), None, body.directions or "")
    return RuleSet(
        framework_name=analysis.get("framework_name") or body.framework_name,
        reference_collection_ids=list(analysis.get("reference_collection_ids") or body.reference_collection_ids),
        requirements=analysis.get("requirements") or None,
        directions=body.directions if body.directions is not None else (analysis.get("directions") or ""),
    )


async def _check_collection_access(user: UserRecord, collection_ids: List[str], checking_again: bool) -> None:
    """Admins can reach every PDF collection; everyone else only their own (the
    rule services/agnostic.py enforces for hybrid search). Without this, a
    non-admin could point reference/target ids at another user's private
    collection and read its content back out through the findings' quotes.

    When an analysis is checked again, a document refused because it no
    longer exists was deleted since its last run: that is said plainly, as
    it is not an access problem. A new run never says it, so a refusal does
    not reveal whether someone else's document exists."""
    allowed = set(await asyncio.to_thread(
        supabase_storage.list_collection_ids_for_user, user.user_id, user.role == "admin"
    ))
    refused = set(collection_ids) - allowed
    if not refused:
        return
    if not checking_again:
        raise HTTPException(status_code=403, detail="One or more collections are not accessible to you")
    deleted = [cid for cid in refused if await asyncio.to_thread(supabase_storage.get_collection, cid) is None]
    if len(deleted) == len(refused):
        raise HTTPException(
            status_code=404,
            detail="A document this analysis checks has been deleted. Start a new analysis with the documents you have now.",
        )
    raise HTTPException(status_code=403, detail="One or more collections are not accessible to you")


async def _run_metered_gap_check(
    user: UserRecord, rule_set: RuleSet, target_collection_ids: List[str], searched: dict
) -> Tuple[List[dict], str, List[dict]]:
    """The check itself, metered like /agnostic/query (MS-248): otherwise the
    rate limit and per-member allocation would be opt-in by which feature you
    use. The workspace lock is held only to reserve
    config.gap_check_token_reserve tokens as in-flight; the multi-minute run
    happens outside it, so it does not freeze every other chat in the
    workspace (see services.billing.inflight.begin_in_flight).
    Returns (items, disclaimer, the requirement list the run used)."""
    workspace_id = await asyncio.to_thread(resolve_workspace_id, user)
    reserve = config.gap_check_token_reserve
    async with get_workspace_lock(workspace_id):
        pending_workspace, pending_user = in_flight_tokens(workspace_id, user.user_id)
        await asyncio.to_thread(enforce_rate_limit, user.user_id, pending_user)
        workspace_cap = await asyncio.to_thread(enforce_plan_limit, user, pending_workspace, reserve)
        user_cap = await asyncio.to_thread(enforce_member_allocation, user, pending_user, reserve)
        reservation = begin_in_flight(workspace_id, user.user_id, reserve, workspace_cap, user_cap)

    try:
        # Several blocking LLM calls: off the event loop, or one run freezes
        # every other request (health checks included) until it finishes.
        items, disclaimer, tokens_consumed, requirements = await asyncio.to_thread(
            processor.run_compliance_gap_check,
            reference_collection_ids=rule_set.reference_collection_ids,
            target_collection_ids=target_collection_ids,
            framework_name=rule_set.framework_name,
            requirements=rule_set.requirements,
            directions=rule_set.directions,
            searched=searched,
        )
        if tokens_consumed:
            try:
                await asyncio.to_thread(log_token_usage, user.user_id, tokens_consumed)
            except Exception:
                logger.warning("run_gap_analysis: failed to log token usage", exc_info=True)
    finally:
        end_in_flight(reservation)
    return items, disclaimer, requirements


@dataclass
class FiledRun:
    """Where a run is filed: its analysis, the rule-set version it checked,
    and the statuses of the analysis's previous run, by requirement key."""
    analysis_id: Optional[str] = None
    rule_set_version: Optional[int] = None
    previous_statuses: Dict[str, str] = field(default_factory=dict)


async def _file_under_analysis(
    body: GapAnalysisRequest,
    user: UserRecord,
    analysis: Optional[dict],
    rule_set: RuleSet,
    requirements: List[dict],
    folder_ids: List[str],
) -> FiledRun:
    """A first run creates its analysis with rule set version 1. A later run
    reads the previous run's statuses first (before this run is stored), and
    saves a new rule-set version whenever the requirement list or directions
    it used differ from the stored ones: a rule set is never changed in place,
    so every run names exactly what it was checked against."""
    if not analysis:
        analysis_id = str(uuid.uuid4())
        created = await asyncio.to_thread(
            supabase_storage.create_analysis,
            analysis_id, user.user_id, body.goal or rule_set.framework_name, folder_ids,
            rule_set.framework_name, rule_set.reference_collection_ids, requirements, rule_set.directions,
        )
        return FiledRun(analysis_id, 1) if created else FiledRun()

    analysis_id = analysis["analysis_id"]
    previous_statuses = await asyncio.to_thread(supabase_storage.get_latest_run_statuses, analysis_id)
    version = analysis.get("rule_set_version")
    if requirements != (analysis.get("requirements") or []) or rule_set.directions != (analysis.get("directions") or ""):
        version = await asyncio.to_thread(
            supabase_storage.add_rule_set_version,
            analysis_id, rule_set.framework_name, rule_set.reference_collection_ids, requirements, rule_set.directions,
        ) or version
    if body.target_folder_ids and list(body.target_folder_ids) != list(analysis.get("target_folder_ids") or []):
        await asyncio.to_thread(supabase_storage.update_analysis_targets, analysis_id, body.target_folder_ids)
    return FiledRun(analysis_id, version, previous_statuses)


def _record_trends(items: List[dict], previous_statuses: Dict[str, str]) -> None:
    """Each item's move since the previous run: improved, worse or unchanged
    (None on a first run, and for requirements that were not checked)."""
    for item in items:
        previous = previous_statuses.get(item.get("requirement_key"))
        item["trend"] = None if item["status"] == NOT_CHECKED else trend(previous, item["status"])


async def _store_run(
    body: GapAnalysisRequest,
    user: UserRecord,
    rule_set: RuleSet,
    items: List[dict],
    disclaimer: str,
    filed: Optional[FiledRun] = None,
    searched: Optional[dict] = None,
) -> GapAnalysisResponse:
    filed = filed or FiledRun()
    run_id = str(uuid.uuid4())
    run = {
        "run_id": run_id,
        "skill_id": body.skill_id,
        "framework_name": rule_set.framework_name,
        "reference_collection_ids": rule_set.reference_collection_ids,
        "target_collection_ids": body.target_collection_ids,
        "scenario_input": body.scenario_input,
        "status": "completed",
        "analysis_id": filed.analysis_id,
        "rule_set_version": filed.rule_set_version,
        "searched": searched,
    }
    await asyncio.to_thread(supabase_storage.create_gap_analysis_run, owner_id=user.user_id, **run)
    await asyncio.to_thread(supabase_storage.save_gap_analysis_items, run_id, items)
    stored = await asyncio.to_thread(supabase_storage.get_gap_analysis_run, run_id)
    return _response(stored or {**run, "created_at": ""}, items, disclaimer)


async def list_runs(user: UserRecord):
    is_admin = user.role == "admin"
    rows = supabase_storage.list_gap_analysis_runs(owner_id=user.user_id, is_admin=is_admin)
    return [
        GapAnalysisRunListItem(**row, run_by=row.get("owner_email") if is_admin else None)
        for row in rows
    ]


async def get_run(run_id: str, user: UserRecord):
    run_row = supabase_storage.get_gap_analysis_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")
    if not _can_access_run(run_row, user):
        raise HTTPException(status_code=403, detail="Not allowed to access this run")

    items = [_normalize_item(it) for it in supabase_storage.get_gap_analysis_items(run_id)]
    return _response(run_row, items, None)


async def delete_run(run_id: str, user: UserRecord):
    run_row = supabase_storage.get_gap_analysis_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")
    if not _can_access_run(run_row, user):
        raise HTTPException(status_code=403, detail="Not allowed to delete this run")

    ok = await asyncio.to_thread(supabase_storage.delete_gap_analysis_run, run_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to delete run")
    return {"run_id": run_id, "deleted": True}


async def export_run(run_id: str, format: str, user: UserRecord):
    run_row = supabase_storage.get_gap_analysis_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")
    if not _can_access_run(run_row, user):
        raise HTTPException(status_code=403, detail="Not allowed to access this run")

    items = supabase_storage.get_gap_analysis_items(run_id)
    markdown = _render_markdown(run_row, items)

    if format == "markdown":
        return Response(
            content=markdown,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="gap-analysis-{run_id}.md"'},
        )

    pdf_bytes = _render_pdf(run_row, items, markdown)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="gap-analysis-{run_id}.pdf"'},
    )
