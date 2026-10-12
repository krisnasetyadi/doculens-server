# router/public_links.py
"""Public link source endpoints. Each route reads the request and calls
services/public_links.py, where the SQL, Drive parsing and URL checks live."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from models import (
    CreatePublicLinkRequest,
    PublicLinkSource,
    PublicLinksResponse,
    SetPublicLinkActiveRequest,
    UserRecord,
)
from services import public_links
from services.auth import get_current_user

router = APIRouter()


@router.get("/public-links", response_model=PublicLinksResponse)
async def list_public_links(user: UserRecord = Depends(get_current_user)):
    return await public_links.list_public_links(user)


@router.post("/public-links", response_model=PublicLinkSource, status_code=status.HTTP_201_CREATED)
async def create_public_link(
    body: CreatePublicLinkRequest,
    user: UserRecord = Depends(get_current_user),
):
    return await public_links.create_public_link(body, user)


@router.post("/public-links/activate")
async def set_public_link_active(
    body: SetPublicLinkActiveRequest,
    user: UserRecord = Depends(get_current_user),
):
    return await public_links.set_public_link_active(body, user)


@router.delete("/public-links/{link_id}")
async def delete_public_link(link_id: str, user: UserRecord = Depends(get_current_user)):
    return await public_links.delete_public_link(link_id, user)
