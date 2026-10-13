# services/skills.py
"""
Skills CRUD (MS-251).

A skill is an uploaded instruction file (.md): the frontmatter becomes name /
slash_command / description, the body becomes `instruction`. This module only
stores and scopes them — running a skill is MS-252 and lives elsewhere.

Two visibility rules, both enforced here and not only in the UI:
  - scope "team"     — admin uploads, every member that admin created can use it
  - scope "personal" — anyone uploads, only the uploader can use it
A non-admin cannot create a team skill, and nobody can edit or delete a skill
they do not own.
"""
import asyncio
import uuid

from fastapi import HTTPException

from services.storage import supabase as supabase_storage
from models import SkillCreate, SkillResponse, SkillUpdate, UserRecord


async def create_skill(body: SkillCreate, user: UserRecord):
    if body.scope == "team" and user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only an admin can share a skill with the whole team",
        )

    skill_id = str(uuid.uuid4())
    ok = await asyncio.to_thread(
        supabase_storage.create_skill,
        skill_id=skill_id,
        name=body.name,
        slash_command=body.slash_command,
        instruction=body.instruction,
        owner_id=user.user_id,
        description=body.description,
        scope=body.scope,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to create skill")

    row = await asyncio.to_thread(
        supabase_storage.get_skill_for_user, skill_id, user.user_id, user.role == "admin"
    )
    if not row:
        raise HTTPException(status_code=500, detail="Skill created but could not be read back")
    return SkillResponse(**row)


async def list_skills(user: UserRecord):
    rows = await asyncio.to_thread(
        supabase_storage.list_skills_for_user, user.user_id, user.role == "admin"
    )
    return [SkillResponse(**row) for row in rows]


async def get_skill(skill_id: str, user: UserRecord):
    row = await asyncio.to_thread(
        supabase_storage.get_skill_for_user, skill_id, user.user_id, user.role == "admin"
    )
    if not row:
        raise HTTPException(status_code=404, detail="Skill not found")
    return SkillResponse(**row)


async def update_skill(skill_id: str, body: SkillUpdate, user: UserRecord):
    row = await asyncio.to_thread(
        supabase_storage.get_skill_for_user, skill_id, user.user_id, user.role == "admin"
    )
    if not row:
        raise HTTPException(status_code=404, detail="Skill not found")
    if row["owner_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to edit this skill")
    if body.scope == "team" and user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only an admin can share a skill with the whole team",
        )

    fields = body.model_dump(exclude_none=True)
    if not fields:
        return SkillResponse(**row)

    updated = await asyncio.to_thread(
        supabase_storage.update_skill, skill_id, user.user_id, fields
    )
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update skill")
    return SkillResponse(**updated)


async def delete_skill(skill_id: str, user: UserRecord):
    row = await asyncio.to_thread(
        supabase_storage.get_skill_for_user, skill_id, user.user_id, user.role == "admin"
    )
    if not row:
        raise HTTPException(status_code=404, detail="Skill not found")
    if row["owner_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to delete this skill")

    ok = await asyncio.to_thread(supabase_storage.delete_skill, skill_id, user.user_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to delete skill")
    return {"skill_id": skill_id, "deleted": True}
