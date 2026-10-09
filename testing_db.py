# testing_db.py
"""
A throwaway PostgreSQL schema for tests that need the app's real SQL to run
(MS-657: behaviour tests written before code is moved).

IsolatedSchemaTestCase creates a fresh schema in the database DATABASE_URL
points at, routes every app connection to it through search_path, creates
the app's tables there with schema.ensure_all(), and drops it again after
the test class. Rows in the real tables are never read or written. Without
a reachable database the tests are skipped, so they cost nothing in a
deployment that has none.
"""

import os
import unittest
import uuid
from urllib.parse import quote

from dotenv import load_dotenv

import db


def _base_url():
    load_dotenv()
    return db.database_url()


def database_reachable() -> bool:
    url = _base_url()
    if not url:
        return False
    try:
        import psycopg2
        psycopg2.connect(url, connect_timeout=3).close()
        return True
    except Exception:
        return False


def _reset_schema_flags():
    """schema/* remember that their tables exist; a new schema needs them again."""
    from schema import database_connections, payment, public_links, sessions, storage, telegram
    for module in (database_connections, payment, public_links, sessions, telegram):
        module._tables_ensured = False
    payment._usage_tables_ensured = False
    storage._migration_done = False


@unittest.skipUnless(database_reachable(), "app database not reachable")
class IsolatedSchemaTestCase(unittest.TestCase):
    schema_name: str

    @classmethod
    def setUpClass(cls):
        import psycopg2
        import schema

        base = _base_url()
        cls.schema_name = f"ms657_test_{uuid.uuid4().hex[:10]}"
        cls._admin = psycopg2.connect(base)
        cls._admin.autocommit = True
        with cls._admin.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {cls.schema_name}")
        cls._saved_url = os.environ.get("DATABASE_URL")
        sep = "&" if "?" in base else "?"
        os.environ["DATABASE_URL"] = f"{base}{sep}options={quote(f'-csearch_path={cls.schema_name}')}"
        db.close_pool()
        _reset_schema_flags()
        if not schema.ensure_all():
            raise RuntimeError("could not create the app tables in the test schema")

    @classmethod
    def tearDownClass(cls):
        db.close_pool()
        if cls._saved_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = cls._saved_url
        _reset_schema_flags()
        with cls._admin.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {cls.schema_name} CASCADE")
        cls._admin.close()

    def sql(self, query, params=None):
        """Run SQL in the test schema; returns rows as dicts for a SELECT."""
        conn = db.get_conn("test")
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall() if cur.description else None
        finally:
            conn.close()
