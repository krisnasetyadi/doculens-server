# router/database_connections.py
"""Database connection endpoints (admin-only). Each route reads the request
and calls services/database_connections.py."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from models import (
    CreateDatabaseConnectionRequest,
    DatabaseConnectionSource,
    DatabaseConnectionsResponse,
    SetDatabaseConnectionActiveRequest,
    UserRecord,
)
from services import database_connections as connection_service
from services.auth import require_role

router = APIRouter()


@router.get("/database-connections", response_model=DatabaseConnectionsResponse)
async def list_database_connections(_: UserRecord = Depends(require_role("admin"))):
    return await connection_service.list_connections()


@router.post("/database-connections", response_model=DatabaseConnectionSource, status_code=status.HTTP_201_CREATED)
async def create_database_connection(
    body: CreateDatabaseConnectionRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    return await connection_service.create_connection(body)


@router.get("/database-connections/{connection_id}/tables", response_model=DatabaseConnectionSource)
async def refresh_database_connection_tables(
    connection_id: str,
    _: UserRecord = Depends(require_role("admin")),
):
    return await connection_service.refresh_tables(connection_id)


@router.post("/database-connections/activate")
async def set_database_connection_active(
    body: SetDatabaseConnectionActiveRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    return await connection_service.set_connection_active(body)


@router.delete("/database-connections/{connection_id}")
async def delete_database_connection(
    connection_id: str,
    _: UserRecord = Depends(require_role("admin")),
):
    return await connection_service.delete_connection(connection_id)
