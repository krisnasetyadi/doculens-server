-- migrations/006_collection_size.sql
-- MS-504: track the size of every source's original file(s) so a workspace's
-- storage can be summed and held to its plan's quota.
--
-- Mirrors the DDL in storage.ensure_schema(), which is what actually runs at
-- startup -- this file is the readable record, like 001 to 005.
--
-- size_bytes is the total of the original uploaded files only. FAISS indices
-- are derived data and are not counted, so the number matches what the user
-- uploaded.
--
-- Rows that existed before this column get 0 until
-- scripts/backfill_collection_sizes.py reads their real sizes from storage.
-- Until then those workspaces look emptier than they are; nothing is blocked
-- by mistake.

ALTER TABLE collections
    ADD COLUMN IF NOT EXISTS size_bytes BIGINT NOT NULL DEFAULT 0;
