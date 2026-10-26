# schema/telegram.py
"""Tables used by services/telegram.py (moved here from router/telegram.py by MS-657)."""

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

                -- MS-912: how far back a chat is kept. Chats synced before it
                -- stored their whole history, so existing rows read 'all'; the
                -- 3 month default for a newly picked chat is chosen by the client.
                ALTER TABLE telegram_selected_chats
                    ADD COLUMN IF NOT EXISTS history_range TEXT NOT NULL DEFAULT 'all'
                    CHECK (history_range IN ('1m', '3m', '1y', 'all'));

                -- MS-911: a phone number is only kept masked, the way
                -- services/telegram.py _mask_phone writes it; mask rows saved
                -- before that. Masking an already masked number changes
                -- nothing, so this is safe at every startup.
                UPDATE telegram_connections
                   SET phone = CASE
                       WHEN length(btrim(phone)) <= 6
                           THEN left(btrim(phone), 1) || repeat('*', greatest(length(btrim(phone)) - 1, 0))
                       ELSE left(btrim(phone), 4)
                            || repeat('*', greatest(length(btrim(phone)) - 7, 1))
                            || right(btrim(phone), 3)
                   END
                 WHERE btrim(phone) <> '' AND position('*' in phone) = 0;
                """
            )
        _tables_ensured = True
    except Exception as exc:
        logger.error("telegram: ensure tables failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to initialize Telegram schema")
