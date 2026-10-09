# schema/database_connections.py
"""Tables owned by router/database_connections.py (moved here from it by MS-657)."""

import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)

_tables_ensured = False


def ensure(conn) -> None:
    global _tables_ensured
    if _tables_ensured:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS database_connections (
                    id            BIGSERIAL PRIMARY KEY,
                    connection_id TEXT        NOT NULL UNIQUE,
                    user_id       TEXT        NOT NULL,
                    workspace_id  TEXT,
                    label         TEXT        NOT NULL,
                    url           TEXT        NOT NULL,
                    status        TEXT        NOT NULL DEFAULT 'active'
                                              CHECK (status IN ('active', 'inactive')),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE INDEX IF NOT EXISTS idx_database_connections_user_created
                    ON database_connections (user_id, created_at DESC);

                CREATE OR REPLACE FUNCTION _set_database_connections_updated_at()
                RETURNS TRIGGER LANGUAGE plpgsql AS $$
                BEGIN
                    NEW.updated_at = now();
                    RETURN NEW;
                END;
                $$;

                DROP TRIGGER IF EXISTS trg_database_connections_updated_at ON database_connections;
                CREATE TRIGGER trg_database_connections_updated_at
                    BEFORE UPDATE ON database_connections
                    FOR EACH ROW EXECUTE FUNCTION _set_database_connections_updated_at();
                """
            )
        _tables_ensured = True
    except Exception as exc:
        logger.error("database_connections: ensure table failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to initialize database-connections schema")
