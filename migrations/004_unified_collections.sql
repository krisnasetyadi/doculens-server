-- migrations/004_unified_collections.sql
-- Unify pdf_collections + chat_collections into one `collections` table,
-- discriminated by `kind` (MS-274 — Folder Management for Sources).
--
-- Mirrors the DDL in storage.ensure_schema(), which is what actually runs at
-- startup — this file is the readable record, like 001, 002 and 003.
--
-- Why: the Files tab already merges PDF/Word/CSV/Excel uploads and WhatsApp
-- exports into one list in the UI, so a Source needs a single folder_id
-- regardless of kind. Two FK columns (one per legacy table) would have been
-- needless duplication once folders (a later migration) land on top of this.
--
-- Column reconciliation:
--   file_names TEXT[] (pdf) vs file_name TEXT singular (chat)
--     -> one array column; chat writes a 1-element array, reads reconstruct
--        file_name = file_names[0] in storage.py (_chat_row_from_unified).
--   chunk_count (pdf) vs message_count (chat)
--     -> one `item_count` column, same reconstruction trick.
--   chat-only platform / participants / date_range / keywords
--     -> folded into `metadata JSONB` (always '{}' for kind='pdf') rather
--        than four more nullable columns for a single kind's data.
--
-- pdf_collections and chat_collections (001) are NOT dropped by this
-- migration — they're left in place, unread and unwritten by the app from
-- this point on, as a rollback snapshot. Dropping them is a separate,
-- later follow-up once the team is confident in the cutover.
--
-- Backfill: idempotent (ON CONFLICT DO NOTHING) INSERT...SELECT statements
-- run inline in storage.ensure_schema() on every startup, not a one-off
-- script — cheap once caught up, and removes the risk of someone forgetting
-- to run a separate migration script against production.

CREATE TABLE IF NOT EXISTS collections (
    id            BIGSERIAL   PRIMARY KEY,
    collection_id TEXT        NOT NULL UNIQUE,
    kind          TEXT        NOT NULL CHECK (kind IN ('pdf', 'chat')),
    title         TEXT        NOT NULL DEFAULT '',
    file_names    TEXT[]      NOT NULL DEFAULT '{}',
    item_count    INTEGER     NOT NULL DEFAULT 0,
    storage_paths TEXT[]      NOT NULL DEFAULT '{}',
    status        TEXT        NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'inactive')),
    owner_id      TEXT,                             -- set for kind='pdf'; always NULL for kind='chat'
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_collections_cid          ON collections (collection_id);
CREATE INDEX IF NOT EXISTS idx_collections_owner        ON collections (owner_id);
CREATE INDEX IF NOT EXISTS idx_collections_kind_created ON collections (kind, created_at DESC);

-- One-time backfill (safe to re-run — ON CONFLICT DO NOTHING).
INSERT INTO collections
    (collection_id, kind, title, file_names, item_count, storage_paths, status, owner_id, created_at, updated_at)
SELECT collection_id, 'pdf', title, file_names, chunk_count, storage_paths, status, owner_id, created_at, updated_at
FROM pdf_collections
ON CONFLICT (collection_id) DO NOTHING;

INSERT INTO collections
    (collection_id, kind, title, file_names, item_count, storage_paths, status, metadata, created_at, updated_at)
SELECT collection_id, 'chat', '', ARRAY[file_name], message_count, storage_paths, status,
       jsonb_build_object(
           'platform', platform,
           'participants', to_jsonb(participants),
           'date_range', date_range,
           'keywords', to_jsonb(keywords)
       ),
       created_at, updated_at
FROM chat_collections
ON CONFLICT (collection_id) DO NOTHING;

-- IMPORTANT — authorization: list_collection_ids_for_user (and every other
-- query below this point in storage.py) MUST filter by kind explicitly.
-- Before this migration, pdf_collections and chat_collections were separate
-- tables, so a query against "the pdf table" was safe by construction. Now
-- that both kinds share one table, omitting `AND kind = 'pdf'` would leak
-- kind='chat' collection_ids into owner-gated, non-admin-visible code paths
-- (router/compliance.py, router/agnostic.py) that were never meant to see
-- them — chat/WhatsApp collections stay admin-only via require_role("admin")
-- at the route layer (router/chat.py), never via owner_id.
