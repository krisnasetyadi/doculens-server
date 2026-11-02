# schema/public_links.py
"""Tables owned by router/public_links.py (moved here from it by MS-657)."""

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
                CREATE TABLE IF NOT EXISTS public_links (
                    id            BIGSERIAL PRIMARY KEY,
                    link_id       TEXT        NOT NULL UNIQUE,
                    user_id       TEXT        NOT NULL,
                    workspace_id  TEXT,
                    title         TEXT        NOT NULL,
                    url           TEXT        NOT NULL,
                    status        TEXT        NOT NULL DEFAULT 'inactive'
                                              CHECK (status IN ('active', 'inactive')),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE TABLE IF NOT EXISTS public_link_items (
                    id            BIGSERIAL PRIMARY KEY,
                    item_id       TEXT        NOT NULL UNIQUE,
                    link_id       TEXT        NOT NULL REFERENCES public_links(link_id) ON DELETE CASCADE,
                    name          TEXT        NOT NULL,
                    url           TEXT        NOT NULL,
                    item_type     TEXT        NOT NULL DEFAULT 'file'
                                              CHECK (item_type IN ('file', 'folder')),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                -- MS-907: a link is indexed once, in the background, and its
                -- index kept in Supabase Storage under
                -- public-links/{link_id}/{index_version}/ (services/public_links.py).
                ALTER TABLE public_links
                    ADD COLUMN IF NOT EXISTS index_status     TEXT NOT NULL DEFAULT 'pending',
                    ADD COLUMN IF NOT EXISTS index_stage      TEXT,
                    ADD COLUMN IF NOT EXISTS index_progress   INTEGER,
                    ADD COLUMN IF NOT EXISTS index_error      TEXT,
                    ADD COLUMN IF NOT EXISTS index_version    TEXT,
                    ADD COLUMN IF NOT EXISTS fingerprint      TEXT,
                    ADD COLUMN IF NOT EXISTS skipped_items    JSONB NOT NULL DEFAULT '[]'::jsonb,
                    ADD COLUMN IF NOT EXISTS index_heartbeat_at TIMESTAMPTZ,
                    ADD COLUMN IF NOT EXISTS last_synced_at   TIMESTAMPTZ;

                CREATE INDEX IF NOT EXISTS idx_public_links_user_created
                    ON public_links (user_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_public_link_items_link
                    ON public_link_items (link_id);

                CREATE OR REPLACE FUNCTION _set_public_links_updated_at()
                RETURNS TRIGGER LANGUAGE plpgsql AS $$
                BEGIN
                    NEW.updated_at = now();
                    RETURN NEW;
                END;
                $$;

                DROP TRIGGER IF EXISTS trg_public_links_updated_at ON public_links;
                CREATE TRIGGER trg_public_links_updated_at
                    BEFORE UPDATE ON public_links
                    FOR EACH ROW EXECUTE FUNCTION _set_public_links_updated_at();
                """
            )
        _tables_ensured = True
    except Exception as exc:
        logger.error("public_links: ensure table failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to initialize public links schema")
