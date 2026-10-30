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
from typing import List, Optional

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


def _summarize(items: List[dict]) -> dict:
    summary = {"met": 0, "partial": 0, "not_met": 0, "unknown": 0}
    for item in items:
        status = item.get("status", "unknown")
        summary[status] = summary.get(status, 0) + 1
    return summary


def _can_access_run(row: Optional[dict], user: UserRecord) -> bool:
    """Admins can access everything; everyone else only their own runs."""
    if user.role == "admin":
        return True
    if not row:
        return False
    return row.get("owner_id") == user.user_id


# Auditor wording for the downloaded report only; stored statuses and the API keep met / not_met.
EXPORT_STATUS_LABELS = {"met": "comply", "not_met": "not comply"}


def _export_status(status: str) -> str:
    return EXPORT_STATUS_LABELS.get(status, status)


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
        lines.append(f"| {item.get('label', '')} | {source} | {_export_status(item.get('status', ''))} | {evidence} | {recommendation} |")
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
            row = [cell(item.get("label", ""))]
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
    if body.skill_id not in SUPPORTED_SKILLS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown skill_id. Supported: {sorted(SUPPORTED_SKILLS)}",
        )
    if not body.reference_collection_ids:
        raise HTTPException(status_code=400, detail="reference_collection_ids is required")

    # Scope collection access — admins can reach every PDF collection;
    # everyone else only their own (same rule services/agnostic.py enforces
    # for hybrid search). Without this, a non-admin could point
    # reference/target ids at another user's private collection and read
    # its content back out via the "evidence" quotes in the response.
    is_admin = user.role == "admin"
    allowed_pdf_ids = set(await asyncio.to_thread(
        supabase_storage.list_collection_ids_for_user, user.user_id, is_admin
    ))
    requested_ids = set(body.reference_collection_ids) | set(body.target_collection_ids)
    if not requested_ids.issubset(allowed_pdf_ids):
        raise HTTPException(status_code=403, detail="One or more collections are not accessible to you")

    run_id = str(uuid.uuid4())
    items: List[dict] = []

    if body.skill_id == "compliance_gap_check":
        # Paid-plan feature (a run costs 130k-260k tokens) — checked first,
        # before any validation or LLM work. See services.billing.plans.GAP_CHECK_PLAN_IDS.
        await asyncio.to_thread(enforce_gap_check_plan, user)
        if not body.target_collection_ids:
            raise HTTPException(
                status_code=400,
                detail="target_collection_ids is required for compliance_gap_check",
            )
        # Gap Check is a second, separate path to the LLM alongside
        # /agnostic/query (MS-248) — it must be metered/enforced the same
        # way, or the flat rate limit and per-member allocation are just
        # opt-in via which feature you happen to use. Same workspace-keyed
        # lock as agnostic.py — see get_workspace_lock's docstring for why
        # it's keyed by workspace rather than by this caller alone.
        # The lock is held only for the check: the run then reserves
        # config.gap_check_token_reserve tokens as in-flight and executes
        # outside the lock, so a multi-minute gap check no longer freezes
        # every other chat in the workspace (see services.billing.inflight.begin_in_flight).
        workspace_id = await asyncio.to_thread(resolve_workspace_id, user)
        reserve = config.gap_check_token_reserve
        async with get_workspace_lock(workspace_id):
            pending_workspace, pending_user = in_flight_tokens(workspace_id, user.user_id)
            await asyncio.to_thread(enforce_rate_limit, user.user_id, pending_user)
            workspace_cap = await asyncio.to_thread(enforce_plan_limit, user, pending_workspace, reserve)
            user_cap = await asyncio.to_thread(enforce_member_allocation, user, pending_user, reserve)
            reservation = begin_in_flight(workspace_id, user.user_id, reserve, workspace_cap, user_cap)

        try:
            # run_compliance_gap_check makes several sequential blocking LLM
            # calls — off the event loop, or one gap-analysis run freezes every
            # other concurrent request (health checks included) until it finishes.
            items, disclaimer, tokens_consumed = await asyncio.to_thread(
                processor.run_compliance_gap_check,
                reference_collection_ids=body.reference_collection_ids,
                target_collection_ids=body.target_collection_ids,
                framework_name=body.framework_name,
            )

            if tokens_consumed:
                try:
                    await asyncio.to_thread(log_token_usage, user.user_id, tokens_consumed)
                except Exception:
                    logger.warning("run_gap_analysis: failed to log token usage", exc_info=True)
        finally:
            end_in_flight(reservation)
    else:
        # scenario_regulatory_impact — scaffold only. The endpoint works
        # end-to-end (request validated, run persisted) so the structure
        # exists for later, but it deliberately returns no analysis yet:
        # per plan, this skill is not validated for real tax/legal decisions.
        disclaimer = SCENARIO_SCAFFOLD_DISCLAIMER

    supabase_storage.create_gap_analysis_run(
        run_id=run_id,
        skill_id=body.skill_id,
        reference_collection_ids=body.reference_collection_ids,
        framework_name=body.framework_name,
        target_collection_ids=body.target_collection_ids,
        scenario_input=body.scenario_input,
        owner_id=user.user_id,
        status="completed",
    )
    supabase_storage.save_gap_analysis_items(run_id, items)

    run_row = supabase_storage.get_gap_analysis_run(run_id) or {
        "run_id": run_id,
        "skill_id": body.skill_id,
        "framework_name": body.framework_name,
        "reference_collection_ids": body.reference_collection_ids,
        "target_collection_ids": body.target_collection_ids,
        "scenario_input": body.scenario_input,
        "status": "completed",
        "created_at": "",
    }

    return GapAnalysisResponse(
        run=GapAnalysisRun(**run_row),
        items=[GapAnalysisItem(**it) for it in items],
        summary=_summarize(items),
        disclaimer=disclaimer,
    )


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

    items = supabase_storage.get_gap_analysis_items(run_id)
    return GapAnalysisResponse(
        run=GapAnalysisRun(**run_row),
        items=[GapAnalysisItem(**it) for it in items],
        summary=_summarize(items),
        disclaimer=None,
    )


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
