# router/skills.py
"""Skill endpoints (MS-251). Each route reads the request and calls
services/skills.py, where the visibility and ownership rules live."""
from typing import List

from fastapi import APIRouter, Depends

from models import SkillCreate, SkillResponse, SkillUpdate, UserRecord
from services import skills as skill_service
from services.auth import get_current_user

router = APIRouter()


@router.post("/skills", response_model=SkillResponse, status_code=201)
async def create_skill(body: SkillCreate, user: UserRecord = Depends(get_current_user)):
    """Upload a skill. `scope="team"` is admin-only — checked server-side, since
    hiding the toggle in the UI would still leave the endpoint open."""
    return await skill_service.create_skill(body, user)


@router.get("/skills", response_model=List[SkillResponse])
async def list_skills(user: UserRecord = Depends(get_current_user)):
    """Skills this account may use: its own, plus its admin's team skills."""
    return await skill_service.list_skills(user)


@router.get("/skills/{skill_id}", response_model=SkillResponse)
async def get_skill(skill_id: str, user: UserRecord = Depends(get_current_user)):
    return await skill_service.get_skill(skill_id, user)


# PUT rather than PATCH even though the body is partial: that is the shape
# already used for partial updates in this codebase (router/sessions.py's
# rename_session) and the one the frontend RequestHandler speaks.
@router.put("/skills/{skill_id}", response_model=SkillResponse)
async def update_skill(
    skill_id: str,
    body: SkillUpdate,
    user: UserRecord = Depends(get_current_user),
):
    """Owner-only edit. A member can see an admin's team skill but not change it."""
    return await skill_service.update_skill(skill_id, body, user)


@router.delete("/skills/{skill_id}")
async def delete_skill(skill_id: str, user: UserRecord = Depends(get_current_user)):
    """Owner-only hard delete, matching how collections and runs are removed."""
    return await skill_service.delete_skill(skill_id, user)
