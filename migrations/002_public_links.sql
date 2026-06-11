-- migrations/002_public_links.sql
-- Public link sources for user-scoped source-of-truth storage.

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
