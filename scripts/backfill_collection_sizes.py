"""One-time backfill (MS-504): fill collections.size_bytes for rows created
before the column existed, from the real object sizes in Supabase Storage.

Run manually once after deploying the size_bytes column, BEFORE relying on
the storage quota (until then old workspaces look emptier than they are):
    python scripts/backfill_collection_sizes.py --dry-run
    python scripts/backfill_collection_sizes.py

Safe to re-run: only rows with size_bytes = 0 are touched. A row whose
objects cannot be found stays at 0 and is reported at the end.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2.extras import RealDictCursor

import storage
from schema import storage as storage_schema

BUCKET_BY_KIND = {"pdf": "pdf-uploads", "chat": "chat-uploads"}


def _database_url() -> str:
    # storage._database_url() also reads the value from .env through config, so
    # the script works without DATABASE_URL being exported in the shell.
    url = storage._database_url()
    if not url:
        raise SystemExit("DATABASE_URL is not set")
    return url


def _object_bytes(s3, bucket: str, collection_id: str) -> int:
    total = 0
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": f"{collection_id}/"}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        total += sum(obj["Size"] for obj in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            return total
        token = resp["NextContinuationToken"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report sizes without writing them")
    args = parser.parse_args()

    s3 = storage._s3_client()
    if not s3:
        raise SystemExit("Supabase S3 credentials are not configured")

    storage_schema.ensure()
    conn = psycopg2.connect(_database_url(), cursor_factory=RealDictCursor, connect_timeout=10)
    conn.autocommit = True
    filled = 0
    missing = []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT collection_id, kind FROM collections WHERE size_bytes = 0")
            rows = cur.fetchall()
        print(f"{len(rows)} collection(s) with size_bytes = 0")

        for row in rows:
            size = _object_bytes(s3, BUCKET_BY_KIND[row["kind"]], row["collection_id"])
            if size == 0:
                missing.append(row["collection_id"])
                continue
            if not args.dry_run:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE collections SET size_bytes = %s WHERE collection_id = %s",
                        (size, row["collection_id"]),
                    )
            filled += 1
    finally:
        conn.close()

    verb = "would fill" if args.dry_run else "filled"
    print(f"{verb} {filled} row(s); {len(missing)} left at 0 (no objects found)")
    for cid in missing:
        print(f"  no objects: {cid}")


if __name__ == "__main__":
    main()
