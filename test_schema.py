"""Tests for MS-657: app tables are ensured once at startup (schema.py), not per request."""

import os
import re
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

import schema
from schema import auth, database_connections, payment, public_links, sessions, storage, telegram

STEPS = [
    (auth, "ensure"),
    (sessions, "ensure"),
    (public_links, "ensure"),
    (database_connections, "ensure"),
    (payment, "ensure"),
    (payment, "ensure_usage"),
    (telegram, "ensure"),
]


class EnsureAllTest(unittest.TestCase):
    def test_every_domain_is_ensured_on_one_connection_that_is_returned(self):
        conn = MagicMock()
        mocks = {}
        with patch.object(storage, "ensure") as storage_schema, \
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
        with patch.object(storage, "ensure"), patch.object(schema.db, "get_conn", return_value=None), \
                patch.object(auth, "ensure") as users:
            self.assertFalse(schema.ensure_all())
        users.assert_not_called()

    def test_a_failing_domain_does_not_stop_startup_or_the_other_domains(self):
        from fastapi import HTTPException
        conn = MagicMock()
        with patch.object(storage, "ensure"), patch.object(schema.db, "get_conn", return_value=conn):
            patches = [patch.object(module, name) for module, name in STEPS]
            mocks = [p.start() for p in patches]
            try:
                # public_links' DDL fails the way its ensure() reports it.
                mocks[2].side_effect = HTTPException(status_code=500, detail="Failed to initialize public links schema")
                self.assertFalse(schema.ensure_all())
            finally:
                for p in patches:
                    p.stop()
        for mock in mocks[3:]:
            mock.assert_called_once_with(conn)
        conn.close.assert_called_once_with()

    def test_handlers_no_longer_create_tables(self):
        """Table creation runs from schema.ensure_all() only, never from a handler."""
        root = os.path.dirname(os.path.abspath(__file__))
        call = re.compile(r"^\s+((\w+\.)?(_ensure_\w+|ensure_schema)|(schema\.)?\w+\.ensure(_usage)?)\(")
        for rel in ("router/auth.py", "router/sessions.py", "router/public_links.py",
                    "router/database_connections.py", "router/payment.py", "router/telegram.py",
                    "storage.py", "storage_limits.py"):
            with open(os.path.join(root, rel), encoding="utf-8-sig") as f:
                for lineno, line in enumerate(f, 1):
                    if call.match(line):
                        self.fail(f"{rel}:{lineno} still calls table creation: {line.strip()}")


if __name__ == "__main__":
    unittest.main()
