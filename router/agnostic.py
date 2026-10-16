"""
router/agnostic.py
------------------
POST /api/v1/agnostic/query  --  Delegates to the existing processor
(same FAISS-indexed collections as hybrid.py) so queries actually work
against uploaded documents.

The "agnostic" label is preserved for UI compatibility; internally this
is the same hybrid-search path with a HybridResponse-compatible schema.
"""

from fastapi import APIRouter, Depends, Request

from models import AgnosticQueryRequest, AgnosticQueryResponse, UserRecord
from services import agnostic as agnostic_service
from services.auth import get_current_user

router = APIRouter(tags=["agnostic"])


@router.post("/agnostic/query", response_model=AgnosticQueryResponse)
async def agnostic_query(
    req: AgnosticQueryRequest,
    request: Request,
    user: UserRecord = Depends(get_current_user),
):
    base_url = str(request.base_url).rstrip('/')
    # request.is_disconnected lets the service stop working on a question
    # the user stopped (MS-667: the browser aborts the request).
    return await agnostic_service.answer_query(req, base_url, user, is_disconnected=request.is_disconnected)
