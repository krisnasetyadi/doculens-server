# TO_REMOVE
# This file is a one-off script / dev utility and is no longer needed.
# Safe to delete after confirming no active references.
# -------------------------------------------------------------------
import psycopg2

conn = psycopg2.connect(
    "postgresql://neondb_owner:npg_oL9hyFqOPEB3"
    "@ep-broad-glitter-a45az27j-pooler.us-east-1.aws.neon.tech"
    "/neondb?sslmode=require"
)
cur = conn.cursor()

cur.execute("SELECT ticker, company_name, report_period, net_profit_usd_k FROM company_watchlist")
rows = cur.fetchall()
print("=== company_watchlist ===")
for r in rows:
    print(" ", r)

cur.execute("SELECT ticker, recommendation, risk_level FROM analyst_notes")
rows = cur.fetchall()
print("\n=== analyst_notes ===")
for r in rows:
    print(" ", r)

conn.close()
