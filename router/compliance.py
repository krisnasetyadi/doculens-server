# router/compliance.py
"""Gap analysis endpoints. Each route reads the request and calls
services/compliance.py, where runs are checked, metered, stored and exported."""
from typing import List

from fastapi import APIRouter, Depends, Query

from models import (
    GapAnalysisRequest,
    ReusableAnalysis,
    SetupDraft,
    SetupDraftRequest,
    SetupSuggestRequest,
    SetupSuggestion,
    GapAnalysisResponse,
    GapAnalysisRunListItem,
    RunDeletedResponse,
    UserRecord,
)
from services import analysis_setup
from services import compliance as compliance_service
from services.auth import get_current_user

router = APIRouter()


@router.post("/analysis/gap-analysis", response_model=GapAnalysisResponse)
async def run_gap_analysis(
    body: GapAnalysisRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Run a Skill against reference (+ optional target) collections and
    persist the result as a run. Generic across skills — not ISO-specific."""
    return await compliance_service.run_gap_analysis(body, user)


@router.get("/analysis/gap-analysis/runs", response_model=List[GapAnalysisRunListItem])
async def list_gap_analysis_runs(user: UserRecord = Depends(get_current_user)):
    """History of runs visible to the current user (own runs, or all for admins)."""
    return await compliance_service.list_runs(user)


@router.post("/analysis/gap-analysis/setup/suggest", response_model=SetupSuggestion)
async def suggest_analysis_setup(body: SetupSuggestRequest, user: UserRecord = Depends(get_current_user)):
    """From "What do you want to check?": the rule file and the folders the AI
    suggests (MS-1114)."""
    return await analysis_setup.suggest_for(body.goal, user)


@router.post("/analysis/gap-analysis/setup/draft", response_model=SetupDraft)
async def draft_analysis_requirements(body: SetupDraftRequest, user: UserRecord = Depends(get_current_user)):
    """The requirement list a new analysis would check, for the user to read
    and approve first (MS-1114)."""
    return await analysis_setup.draft_for(body, user)


@router.get("/analysis/gap-analysis/analyses", response_model=List[ReusableAnalysis])
async def list_reusable_analyses(user: UserRecord = Depends(get_current_user)):
    """Analyses whose rule set a new check can reuse (MS-1114). Declared before
    /{run_id} so "analyses" is not read as a run id."""
    return await analysis_setup.reusable_for(user)


@router.get("/analysis/gap-analysis/{run_id}", response_model=GapAnalysisResponse)
async def get_gap_analysis_run(run_id: str, user: UserRecord = Depends(get_current_user)):
    """Full result of a stored run — this is the 'view everything in-app'
    path (not just download): the frontend's full-table dialog reads from
    this endpoint, no re-analysis needed."""
    return await compliance_service.get_run(run_id, user)


@router.delete("/analysis/gap-analysis/{run_id}", response_model=RunDeletedResponse)
async def delete_gap_analysis_run(run_id: str, user: UserRecord = Depends(get_current_user)):
    """Delete a stored run — runs never overwrite each other (each gets a
    fresh run_id), so this is the only way to remove one from history."""
    return await compliance_service.delete_run(run_id, user)


@router.get("/analysis/gap-analysis/{run_id}/export")
async def export_gap_analysis_run(
    run_id: str,
    format: str = Query("markdown", pattern="^(markdown|pdf)$"),
    user: UserRecord = Depends(get_current_user),
):
    """Render a stored run into a downloadable report. Pure render of
    already-persisted data — no LLM call. Per plan, this is part of Skill 1's
    MVP (not deferred): download is an additional way to take the report
    outside the app, not the only way to see the full result."""
    return await compliance_service.export_run(run_id, format, user)
