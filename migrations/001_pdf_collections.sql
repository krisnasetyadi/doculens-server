-- migrations/001_pdf_collections.sql
-- Run once in your Supabase SQL Editor.
--
-- Creates:
--   1. pdf_collections table  — tracks every uploaded collection
--   2. Storage buckets        — pdf-uploads, pdf-indices  (created via dashboard
--                                or uncomment the INSERT statements below if your
--                                Supabase project exposes storage.buckets)
--
-- After running this migration:
--   • Set SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env / HF Space secrets.
--   • New uploads will persist in Supabase Storage across container restarts.


-- ─────────────────────────────────────────────────────────────────────────────
-- 1. pdf_collections metadata table
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS pdf_collections (
    id             BIGSERIAL PRIMARY KEY,
    collection_id  UUID        NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    file_names     TEXT[]      NOT NULL DEFAULT '{}',
    chunk_count    INTEGER     NOT NULL DEFAULT 0,
    storage_paths  TEXT[]      NOT NULL DEFAULT '{}',  -- paths in pdf-uploads bucket
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Keep updated_at current automatically
CREATE OR REPLACE FUNCTION _set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_pdf_collections_updated_at ON pdf_collections;
CREATE TRIGGER trg_pdf_collections_updated_at
    BEFORE UPDATE ON pdf_collections
    FOR EACH ROW EXECUTE FUNCTION _set_updated_at();

-- Index for fast UUID lookups
CREATE INDEX IF NOT EXISTS idx_pdf_collections_collection_id
    ON pdf_collections (collection_id);

-- Index for ordering / pagination
CREATE INDEX IF NOT EXISTS idx_pdf_collections_created_at
    ON pdf_collections (created_at DESC);


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Row Level Security  (optional but recommended)
--    Service-role key bypasses RLS — only anon/authenticated are restricted.
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE pdf_collections ENABLE ROW LEVEL SECURITY;

-- Allow service-role full access (used by the API server)
CREATE POLICY "service_role_all"
    ON pdf_collections
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Deny all for anonymous users (tighten as needed)
-- To allow authenticated users to read:
-- CREATE POLICY "authenticated_read"
--   ON pdf_collections FOR SELECT TO authenticated USING (true);


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Storage buckets
--    Create these via: Supabase Dashboard → Storage → New bucket
--    OR uncomment below if your project allows SQL bucket creation:
-- ─────────────────────────────────────────────────────────────────────────────

-- INSERT INTO storage.buckets (id, name, public, file_size_limit)
-- VALUES
--   ('pdf-uploads', 'pdf-uploads', false, 52428800),   -- 50 MB per file
--   ('pdf-indices', 'pdf-indices', false, 104857600)   -- 100 MB per file
-- ON CONFLICT (id) DO NOTHING;


-- ─────────────────────────────────────────────────────────────────────────────
-- Done.  Verify with:
--   SELECT * FROM pdf_collections LIMIT 5;
-- ─────────────────────────────────────────────────────────────────────────────
