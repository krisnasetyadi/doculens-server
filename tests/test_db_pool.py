"""Tests for MS-657: the shared, pooled connection to the app's own database (db.py)."""

import os
import unittest
from unittest.mock import patch

from dotenv import load_dotenv

import db

FAKE_URL = "postgresql://user:pw@db.invalid:5432/app?sslmode=disable"


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if self.conn.dead:
            raise RuntimeError("server closed the connection unexpectedly")

    def fetchone(self):
        return {"?column?": 1}


class FakeConn:
    def __init__(self):
        self.closed = 0
        self.autocommit = False
        self.dead = False
        self.rollbacks = 0

    def cursor(self):
        return FakeCursor(self)

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = 1


class PoolTest(unittest.TestCase):
    def setUp(self):
        db.close_pool()
        self.opened = []

        def connect(*args, **kwargs):
            conn = FakeConn()
            self.opened.append(conn)
            return conn

        patches = [
            patch.dict(os.environ, {"DATABASE_URL": FAKE_URL}),
            patch("psycopg2.connect", side_effect=connect),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(db.close_pool)

    def test_connection_is_reused_instead_of_reopened(self):
        first = db.get_conn("test")
        raw = first._conn
        first.close()
        second = db.get_conn("test")
        self.assertIs(second._conn, raw)
        self.assertEqual(len(self.opened), 1)
        second.close()

    def test_connection_has_autocommit_on(self):
        conn = db.get_conn("test")
        self.assertTrue(conn.autocommit)
        conn.close()

    def test_closing_twice_is_harmless(self):
        conn = db.get_conn("test")
        conn.close()
        conn.close()
        self.assertEqual(self.opened[0].closed, 0)

    def test_open_transaction_is_rolled_back_before_reuse(self):
        conn = db.get_conn("test")
        conn.autocommit = False
        raw = self.opened[0]
        conn.close()
        self.assertEqual(raw.rollbacks, 1)
        self.assertTrue(raw.autocommit)

    def test_closed_connection_is_replaced(self):
        conn = db.get_conn("test")
        self.opened[0].closed = 1
        conn.close()
        again = db.get_conn("test")
        self.assertIsNot(again._conn, self.opened[0])
        self.assertEqual(len(self.opened), 2)
        again.close()

    def test_dead_idle_connection_is_replaced(self):
        conn = db.get_conn("test")
        conn.close()
        self.opened[0].dead = True
        with patch.object(db, "PING_AFTER_SECONDS", 0):
            again = db.get_conn("test")
        self.assertIsNot(again._conn, self.opened[0])
        again.close()

    def test_full_pool_falls_back_to_a_one_off_connection(self):
        with patch.object(db, "POOL_MAX", 1):
            db.close_pool()
            held = db.get_conn("test")
            extra = db.get_conn("test")
            self.assertIsNotNone(extra)
            self.assertEqual(len(self.opened), 2)
            extra.close()
            self.assertEqual(self.opened[1].closed, 1)  # not kept in the pool
            held.close()
            self.assertEqual(self.opened[0].closed, 0)  # kept in the pool

    def test_concurrent_use_never_holds_more_than_the_limit(self):
        import threading
        with patch.object(db, "POOL_MAX", 3):
            db.close_pool()

            def worker():
                for _ in range(50):
                    db.get_conn("test").close()

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            kept = [c for c in self.opened if not c.closed]
            self.assertLessEqual(len(kept), 3)
            self.assertEqual(len(db._idle), len(kept))
            self.assertEqual(db._open, len(kept))

    def test_unreachable_database_returns_none(self):
        with patch("psycopg2.connect", side_effect=RuntimeError("could not connect")):
            self.assertIsNone(db.get_conn("test"))

    def test_unconfigured_database_returns_none(self):
        with patch.dict(os.environ, {"DATABASE_URL": ""}), patch.object(db.config, "database_url", None):
            self.assertIsNone(db.get_conn("test"))

    def test_sslmode_required_when_url_does_not_say(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://u:p@h/db"}):
            self.assertEqual(db.database_url(), "postgresql://u:p@h/db?sslmode=require")


def _real_db_reachable() -> bool:
    load_dotenv()
    if not os.getenv("DATABASE_URL"):
        return False
    try:
        import psycopg2
        psycopg2.connect(db.database_url(), connect_timeout=3).close()
        return True
    except Exception:
        return False


@unittest.skipUnless(_real_db_reachable(), "app database not reachable")
class RealDatabaseTest(unittest.TestCase):
    def setUp(self):
        db.close_pool()
        self.addCleanup(db.close_pool)

    def backend_pid(self):
        conn = db.get_conn("test")
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_backend_pid() AS pid")
                return cur.fetchone()["pid"]
        finally:
            conn.close()

    def test_consecutive_requests_share_one_server_connection(self):
        self.assertEqual(self.backend_pid(), self.backend_pid())


if __name__ == "__main__":
    unittest.main()
