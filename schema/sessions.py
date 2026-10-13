# schema/sessions.py
"""Tables owned by router/sessions.py (moved here from it by MS-657)."""

import logging

logger = logging.getLogger(__name__)

_tables_ensured = False


def ensure(conn):
    """Create chat_sessions + chat_messages tables if they do not exist."""
    global _tables_ensured
    if _tables_ensured:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
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
            """)
        _tables_ensured = True
        logger.info("sessions: Schema ensured successfully.")
    except Exception as e:
        logger.warning("sessions: ensure tables failed: %s", e)
