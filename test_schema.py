"""Tests for MS-657: app tables are ensured once at startup (schema.py), not per request."""

import os
import re
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

import schema
import storage
from router import auth, database_connections, payment, public_links, sessions, telegram

STEPS = [
    (auth, "_ensure_users_table"),
    (sessions, "_ensure_tables"),
    (public_links, "_ensure_tables"),
    (database_connections, "_ensure_tables"),
    (payment, "_ensure_tables"),
    (payment, "_ensure_usage_tables"),
    (telegram, "_ensure_tables"),
]


class EnsureAllTest(unittest.TestCase):
    def test_every_domain_is_ensured_on_one_connection_that_is_returned(self):
        conn = MagicMock()
        mocks = {}
        with patch.object(storage, "ensure_schema") as storage_schema, \
                patch.object(schema.db, "get_conn", return_value=conn):
            patches = [patch.object(module, name) for module, name in STEPS]
            for (module, name), p in zip(STEPS, patches):
                mocks[(module.__name__, name)] = p.start()
            try:
                self.assertTrue(schema.ensure_all())
            finally:
                for p in patches:
                    p.stop()
        storage_schema.assert_called_once_with()
        for (module_name, name), mock in mocks.items():
            with self.subTest(step=f"{module_name}.{name}"):
                mock.assert_called_once_with(conn)
        conn.close.assert_called_once_with()

    def test_unreachable_database_is_reported_not_raised(self):
        with patch.object(storage, "ensure_schema"), patch.object(schema.db, "get_conn", return_value=None), \
                patch.object(auth, "_ensure_users_table") as users:
            self.assertFalse(schema.ensure_all())
        users.assert_not_called()

    def test_a_failing_domain_does_not_stop_startup_or_the_other_domains(self):
        from fastapi import HTTPException
        conn = MagicMock()
        with patch.object(storage, "ensure_schema"), patch.object(schema.db, "get_conn", return_value=conn):
            patches = [patch.object(module, name) for module, name in STEPS]
            mocks = [p.start() for p in patches]
            try:
                # public_links' DDL fails the way its _ensure_tables reports it.
                mocks[2].side_effect = HTTPException(status_code=500, detail="Failed to initialize public links schema")
                self.assertFalse(schema.ensure_all())
            finally:
                for p in patches:
                    p.stop()
        for mock in mocks[3:]:
            mock.assert_called_once_with(conn)
        conn.close.assert_called_once_with()

    def test_handlers_no_longer_create_tables(self):
        """Table creation lives in the _ensure_* definitions and schema.py only."""
        root = os.path.dirname(os.path.abspath(__file__))
        call = re.compile(r"^\s+(\w+\.)?(_ensure_\w+|ensure_schema)\(")
        for rel in ("router/auth.py", "router/sessions.py", "router/public_links.py",
                    "router/database_connections.py", "router/payment.py", "router/telegram.py",
                    "storage.py", "storage_limits.py"):
            with open(os.path.join(root, rel), encoding="utf-8-sig") as f:
                for lineno, line in enumerate(f, 1):
                    if call.match(line):
                        self.fail(f"{rel}:{lineno} still calls table creation: {line.strip()}")


if __name__ == "__main__":
    unittest.main()
