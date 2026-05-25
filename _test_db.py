import os
import psycopg2
from psycopg2.extras import RealDictCursor

URL = os.environ["DATABASE_URL"]

conn = psycopg2.connect(URL, cursor_factory=RealDictCursor, connect_timeout=10)
conn.autocommit = True
print("✓ Connected")

with conn.cursor() as cur:
    # Create all tables
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
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_sid ON chat_sessions (session_id);

        CREATE TABLE IF NOT EXISTS chat_messages (
            id          BIGSERIAL   PRIMARY KEY,
            message_id  TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
            session_id  TEXT        NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
            role        TEXT        NOT NULL,
            content     TEXT        NOT NULL DEFAULT '',
            model_used  TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages (session_id, created_at ASC);
    """)
    print("✓ Tables created / already exist")

    # List all public tables
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")
    tables = [r["table_name"] for r in cur.fetchall()]
    print(f"✓ Tables in DB: {tables}")

    # Quick sanity insert
    cur.execute("INSERT INTO chat_sessions (session_id, title) VALUES ('_test_', 'Test') ON CONFLICT DO NOTHING")
    cur.execute("SELECT COUNT(*) AS c FROM chat_sessions")
    print(f"✓ chat_sessions row count: {cur.fetchone()['c']}")
    cur.execute("DELETE FROM chat_sessions WHERE session_id = '_test_'")

conn.close()
print("\n✅ All good — update HF Space secret DATABASE_URL to the aws-1 URL above")
