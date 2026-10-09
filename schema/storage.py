# schema/storage.py
"""Tables owned by storage.py (moved here from it by MS-657)."""

import logging

import db

logger = logging.getLogger(__name__)

_migration_done = False


def ensure():
    """Create pdf_collections + chat_collections tables if they don't exist."""
    global _migration_done
    if _migration_done:
        return
    database_url = db.database_url()
    if not database_url:
        logger.warning("ensure_schema: DATABASE_URL not set — skipping auto-migration")
        _migration_done = True
        return
    logger.info("ensure_schema: connecting to DB to create tables...")
    try:
        import psycopg2
        # Embed sslmode in URL to avoid kwarg conflict with pooler DSN
        conn = psycopg2.connect(database_url, connect_timeout=10)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_collections (
                    id            BIGSERIAL PRIMARY KEY,
                    collection_id TEXT        NOT NULL UNIQUE,
                    title         TEXT        NOT NULL DEFAULT '',
                    file_names    TEXT[]      NOT NULL DEFAULT '{}',
                    chunk_count   INTEGER     NOT NULL DEFAULT 0,
                    storage_paths TEXT[]      NOT NULL DEFAULT '{}',
                    status        TEXT        NOT NULL DEFAULT 'active'
                                  CHECK (status IN ('active', 'inactive')),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                ALTER TABLE pdf_collections
                    ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT '';
                ALTER TABLE pdf_collections
                    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
                ALTER TABLE pdf_collections
                    ADD COLUMN IF NOT EXISTS owner_id TEXT;
                CREATE INDEX IF NOT EXISTS idx_pdf_collections_cid
                    ON pdf_collections (collection_id);
                CREATE INDEX IF NOT EXISTS idx_pdf_collections_owner
                    ON pdf_collections (owner_id);
                CREATE INDEX IF NOT EXISTS idx_pdf_collections_created
                    ON pdf_collections (created_at DESC);

                CREATE TABLE IF NOT EXISTS chat_collections (
                    id              BIGSERIAL PRIMARY KEY,
                    collection_id   TEXT        NOT NULL UNIQUE,
                    file_name       TEXT        NOT NULL DEFAULT '',
                    platform        TEXT        NOT NULL DEFAULT 'whatsapp',
                    message_count   INTEGER     NOT NULL DEFAULT 0,
                    participants    TEXT[]      NOT NULL DEFAULT '{}',
                    date_range      JSONB,
                    keywords        TEXT[]      NOT NULL DEFAULT '{}',
                    storage_paths   TEXT[]      NOT NULL DEFAULT '{}',
                    status          TEXT        NOT NULL DEFAULT 'active'
                                    CHECK (status IN ('active', 'inactive')),
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                ALTER TABLE chat_collections
                    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
                CREATE INDEX IF NOT EXISTS idx_chat_collections_cid
                    ON chat_collections (collection_id);
                CREATE INDEX IF NOT EXISTS idx_chat_collections_created
                    ON chat_collections (created_at DESC);

                -- MS-274: pdf_collections + chat_collections unified into one
                -- table so a Source can carry a single folder_id regardless of
                -- kind. pdf_collections/chat_collections above are left in
                -- place (unread, unwritten) as a rollback snapshot — not
                -- dropped here. See migrations/004_unified_collections.sql.
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
                    owner_id      TEXT,
                    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_collections_cid
                    ON collections (collection_id);
                CREATE INDEX IF NOT EXISTS idx_collections_owner
                    ON collections (owner_id);
                CREATE INDEX IF NOT EXISTS idx_collections_kind_created
                    ON collections (kind, created_at DESC);
                -- MS-504: bytes of the original uploaded file(s), summed per
                -- workspace to enforce the storage quota. 0 for rows created
                -- before this column existed until
                -- scripts/backfill_collection_sizes.py fills them in.
                ALTER TABLE collections
                    ADD COLUMN IF NOT EXISTS size_bytes BIGINT NOT NULL DEFAULT 0;
                -- One-time backfill from the legacy tables. Idempotent via
                -- ON CONFLICT DO NOTHING, so it's safe (and cheap once caught
                -- up) to leave running on every startup rather than requiring
                -- a separate manual script — same idiom as the
                -- gap_analysis_runs.target_collection_ids backfill above.
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

                -- Folders group `collections` rows (Files tab only).
                -- Before deleting a folder, storage.delete_folder moves its
                -- direct sources and child folders to the deleted folder's parent.
                CREATE TABLE IF NOT EXISTS folders (
                    id          BIGSERIAL   PRIMARY KEY,
                    folder_id   TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    name        TEXT        NOT NULL,
                    owner_id    TEXT        NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                ALTER TABLE folders ADD COLUMN IF NOT EXISTS parent_folder_id TEXT
                    REFERENCES folders(folder_id) ON DELETE RESTRICT;
                CREATE INDEX IF NOT EXISTS idx_folders_owner ON folders (owner_id);
                CREATE INDEX IF NOT EXISTS idx_folders_parent ON folders (parent_folder_id);

                ALTER TABLE collections
                    ADD COLUMN IF NOT EXISTS folder_id TEXT
                        REFERENCES folders(folder_id) ON DELETE SET NULL;
                CREATE INDEX IF NOT EXISTS idx_collections_folder ON collections (folder_id);

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id               BIGSERIAL PRIMARY KEY,
                    session_id       TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    title            TEXT        NOT NULL DEFAULT '',
                    pdf_collections  TEXT[]      NOT NULL DEFAULT '{}',
                    chat_collections TEXT[]      NOT NULL DEFAULT '{}',
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                ALTER TABLE chat_sessions
                    ADD COLUMN IF NOT EXISTS owner_id TEXT;
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_sid
                    ON chat_sessions (session_id);
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated
                    ON chat_sessions (updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_owner
                    ON chat_sessions (owner_id);

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id          BIGSERIAL   PRIMARY KEY,
                    message_id  TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    session_id  TEXT        NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                    role        TEXT        NOT NULL,
                    content     TEXT        NOT NULL DEFAULT '',
                    model_used  TEXT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                    ON chat_messages (session_id, created_at ASC);

                CREATE TABLE IF NOT EXISTS gap_analysis_runs (
                    id                       BIGSERIAL   PRIMARY KEY,
                    run_id                   TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    skill_id                 TEXT        NOT NULL,
                    framework_name           TEXT        NOT NULL DEFAULT '',
                    reference_collection_ids TEXT[]      NOT NULL DEFAULT '{}',
                    target_collection_id     TEXT,
                    scenario_input           TEXT,
                    owner_id                 TEXT,
                    status                   TEXT        NOT NULL DEFAULT 'completed'
                                             CHECK (status IN ('running', 'completed', 'failed')),
                    created_at               TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                -- target_collection_id (singular) is kept only so old rows stay
                -- readable; new rows are written to target_collection_ids below,
                -- which supports comparing one guideline against several files.
                ALTER TABLE gap_analysis_runs
                    ADD COLUMN IF NOT EXISTS target_collection_ids TEXT[] NOT NULL DEFAULT '{}';
                -- Backfill pre-existing rows once: every read path now selects only
                -- the plural column, so without this, old runs would silently show
                -- "no target" the moment this migration lands. Idempotent — the
                -- WHERE clause only matches rows that haven't been backfilled yet.
                UPDATE gap_analysis_runs
                   SET target_collection_ids = ARRAY[target_collection_id]
                 WHERE target_collection_id IS NOT NULL
                   AND (target_collection_ids IS NULL OR target_collection_ids = '{}');
                CREATE INDEX IF NOT EXISTS idx_gap_analysis_runs_run_id
                    ON gap_analysis_runs (run_id);
                CREATE INDEX IF NOT EXISTS idx_gap_analysis_runs_owner
                    ON gap_analysis_runs (owner_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS gap_analysis_items (
                    id              BIGSERIAL   PRIMARY KEY,
                    run_id          TEXT        NOT NULL REFERENCES gap_analysis_runs(run_id) ON DELETE CASCADE,
                    label           TEXT        NOT NULL,
                    status          TEXT        NOT NULL DEFAULT 'unknown'
                                    CHECK (status IN ('met', 'partial', 'not_met', 'unknown')),
                    evidence        TEXT,
                    source_citation TEXT,
                    recommendation  TEXT,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                ALTER TABLE gap_analysis_items
                    ADD COLUMN IF NOT EXISTS target_collection_id TEXT;
                CREATE INDEX IF NOT EXISTS idx_gap_analysis_items_run
                    ON gap_analysis_items (run_id);

                CREATE TABLE IF NOT EXISTS users (
                    id            BIGSERIAL    PRIMARY KEY,
                    user_id       TEXT         NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    email         TEXT         NOT NULL UNIQUE,
                    password_hash TEXT         NOT NULL,
                    role          TEXT         NOT NULL DEFAULT 'user'
                                  CHECK (role IN ('user', 'admin')),
                    is_active     BOOLEAN      NOT NULL DEFAULT true,
                    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_users_email   ON users (email);
                CREATE INDEX IF NOT EXISTS idx_users_user_id ON users (user_id);

                CREATE TABLE IF NOT EXISTS skills (
                    id            BIGSERIAL   PRIMARY KEY,
                    skill_id      TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    name          TEXT        NOT NULL,
                    slash_command TEXT        NOT NULL,
                    description   TEXT        NOT NULL DEFAULT '',
                    instruction   TEXT        NOT NULL,
                    scope         TEXT        NOT NULL DEFAULT 'personal'
                                  CHECK (scope IN ('personal', 'team')),
                    owner_id      TEXT        NOT NULL,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                -- owner_id leads the index: both visibility branches filter it by
                -- equality, and scope only has two values so it barely narrows
                -- anything on its own. Same shape as idx_gap_analysis_runs_owner.
                CREATE INDEX IF NOT EXISTS idx_skills_owner_scope
                    ON skills (owner_id, scope);
            """)
        conn.close()
        logger.info("Schema ensured: collections (pdf+chat, unified), folders, chat_sessions, chat_messages, gap_analysis_runs, gap_analysis_items, skills.")
    except Exception as e:
        logger.warning("Auto-migration skipped (disk fallback): %s", e)
    _migration_done = True
