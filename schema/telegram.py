# schema/telegram.py
"""Tables owned by router/telegram.py (moved here from it by MS-657)."""

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
                CREATE TABLE IF NOT EXISTS telegram_connections (
                    id             BIGSERIAL   PRIMARY KEY,
                    connection_id  TEXT        NOT NULL UNIQUE,
                    user_id        TEXT        NOT NULL,
                    api_id         BIGINT      NOT NULL,
                    api_hash_enc   TEXT        NOT NULL,
                    phone          TEXT        NOT NULL,
                    label          TEXT        NOT NULL,
                    session_enc    TEXT        NOT NULL,
                    status         TEXT        NOT NULL DEFAULT 'active'
                                   CHECK (status IN ('active', 'inactive')),
                    last_synced_at TIMESTAMPTZ,
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_telegram_connections_user
                    ON telegram_connections (user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS telegram_selected_chats (
                    id                  BIGSERIAL   PRIMARY KEY,
                    connection_id       TEXT        NOT NULL REFERENCES telegram_connections(connection_id) ON DELETE CASCADE,
                    dialog_id           TEXT        NOT NULL,
                    dialog_title        TEXT        NOT NULL,
                    dialog_type         TEXT        NOT NULL,
                    chat_collection_id  TEXT,
                    message_count       INTEGER,
                    status              TEXT        NOT NULL DEFAULT 'active'
                                        CHECK (status IN ('active', 'inactive')),
                    last_synced_at      TIMESTAMPTZ,
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (connection_id, dialog_id)
                );
                CREATE INDEX IF NOT EXISTS idx_telegram_selected_chats_connection
                    ON telegram_selected_chats (connection_id);
                ALTER TABLE telegram_selected_chats
                    ADD COLUMN IF NOT EXISTS message_count INTEGER;
                """
            )
        _tables_ensured = True
    except Exception as exc:
        logger.error("telegram: ensure tables failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to initialize Telegram schema")
