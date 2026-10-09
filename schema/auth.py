# schema/auth.py
"""Tables owned by router/auth.py (moved here from it by MS-657)."""

import logging

logger = logging.getLogger(__name__)


def ensure(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("""
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
                ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS created_by TEXT;
                ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS max_sub_users INTEGER NOT NULL DEFAULT 5;
                ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS name TEXT;
                ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS avatar_url TEXT;
                CREATE INDEX IF NOT EXISTS idx_users_email      ON users (email);
                CREATE INDEX IF NOT EXISTS idx_users_user_id    ON users (user_id);
                CREATE INDEX IF NOT EXISTS idx_users_created_by ON users (created_by);
            """)
    except Exception as e:
        logger.warning("auth: ensure users table failed: %s", e)
