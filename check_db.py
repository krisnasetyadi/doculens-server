# TO_REMOVE
# This file is a one-off script / dev utility and is no longer needed.
# Safe to delete after confirming no active references.
# -------------------------------------------------------------------
import os
import psycopg2

DB_URL = os.environ["DATABASE_URL"]

conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
tables = [r[0] for r in cur.fetchall()]
print('Tables:', tables)

for t in tables:
    try:
        cur.execute(f'SELECT COUNT(*) FROM "{t}"')
        count = cur.fetchone()[0]
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name=%s ORDER BY ordinal_position", (t,))
        cols = cur.fetchall()
        print(f"\n=== {t} ({count} rows) ===")
        for c in cols:
            print(f"  {c[0]} ({c[1]})")
        if count > 0:
            cur.execute(f'SELECT * FROM "{t}" LIMIT 1')
            rows = cur.fetchall()
            print(f"  Sample: {rows[0]}")
    except Exception as e:
        print(f"  ERROR {t}: {e}")
        conn.rollback()

conn.close()

