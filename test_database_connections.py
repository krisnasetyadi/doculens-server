"""MS-657: behaviour of database connections as a knowledge source
(router/database_connections.py) against real SQL, recorded before it is
moved: creating (connect + schema browse first), listing with the password
masked, refreshing tables, activating, deleting, the optional per-user scope
and the active list the RAG pipeline reads. The user's external database is a
fake answering the three schema queries; the app's own tables live in a
throwaway schema (testing_db); skipped when no database is reachable."""

import os
import unittest
import uuid
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

import asyncio

import psycopg2
from fastapi import FastAPI
from fastapi.testclient import TestClient

from models import UserRecord
from router import database_connections
from services.auth import get_current_user
from testing_db import IsolatedSchemaTestCase

# Built from parts: secret scanners flag any user:password@ URL literal, even a fake one.
FAKE_PASSWORD = "not-a-real-password"
EXTERNAL_URL = f"postgresql://hr_user:{FAKE_PASSWORD}@ext.example.com:5432/hr"
MASKED_URL = "postgresql://hr_user:" + "****" + "@ext.example.com:5432/hr"
_real_connect = psycopg2.connect


class FakeExternalDb:
    """Answers list_tables_with_columns' three queries for one small schema."""

    tables = [{"table_name": "employees", "approx_rows": 12}, {"table_name": "teams", "approx_rows": 0}]
    columns = [
        {"table_name": "employees", "column_name": "id", "data_type": "integer", "is_nullable": "NO"},
        {"table_name": "employees", "column_name": "name", "data_type": "text", "is_nullable": "YES"},
        {"table_name": "teams", "column_name": "code", "data_type": "text", "is_nullable": "NO"},
    ]
    primary_keys = [{"table_name": "employees", "column_name": "id"}]

    def __init__(self):
        self.closed = False
        self.autocommit = False
        self._result = []

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if "pg_class" in sql:
            self._result = self.tables
        elif "table_constraints" in sql:
            self._result = self.primary_keys
        else:
            self._result = self.columns

    def fetchall(self):
        return self._result

    def close(self):
        self.closed = True


class DatabaseConnectionsTestCase(IsolatedSchemaTestCase):
    def setUp(self):
        self.external = []

        def connect(dsn=None, *args, **kwargs):
            if dsn and "ext.example.com" in dsn:
                fake = FakeExternalDb()
                self.external.append(fake)
                return fake
            if dsn and "down.example.com" in dsn:
                raise psycopg2.OperationalError("could not translate host name")
            return _real_connect(dsn, *args, **kwargs)

        connect_patch = patch("psycopg2.connect", connect)
        connect_patch.start()
        self.addCleanup(connect_patch.stop)
        for name in ("DB_CONNECTIONS_ENFORCE_SCOPE", "DB_CONNECTIONS_USER_ID"):
            env = patch.dict(os.environ, {name: os.environ.get(name, "")})
            env.start()
            self.addCleanup(env.stop)
            os.environ.pop(name, None)
        self.sql("DELETE FROM database_connections")
        self.role = "admin"

    @property
    def http(self):
        app = FastAPI()
        app.include_router(database_connections.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="admin-1", email="admin@example.com", role=self.role, is_active=True)
        return TestClient(app)

    def create(self, url=EXTERNAL_URL, label=None):
        return self.http.post("/api/v1/database-connections", json={"url": url, **({"label": label} if label else {})})


class CreateTest(DatabaseConnectionsTestCase):
    def test_a_reachable_database_is_browsed_and_saved_with_the_password_masked(self):
        resp = self.create()
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        self.assertEqual((body["label"], body["url"], body["status"], body["table_count"]),
                         ("ext.example.com/hr", MASKED_URL, "active", 2))
        employees, teams = body["tables"]
        self.assertEqual((employees["name"], employees["row_count"]), ("employees", 12))
        self.assertEqual(employees["columns"], [
            {"name": "id", "type": "integer", "nullable": False, "primary_key": True},
            {"name": "name", "type": "text", "nullable": True, "primary_key": False},
        ])
        self.assertEqual([c["name"] for c in teams["columns"]], ["code"])
        self.assertTrue(self.external[0].closed and self.external[0].autocommit)
        row = self.sql("SELECT * FROM database_connections")[0]
        self.assertEqual((row["url"], row["label"], row["user_id"]), (EXTERNAL_URL, "ext.example.com/hr", "db-connections-local-user"))

    def test_invalid_and_unreachable_databases_are_not_saved(self):
        self.assertEqual(self.create("mysql://x@host/db").json()["detail"], "URL harus diawali postgresql:// atau postgres://")
        self.assertEqual(self.create("postgresql:///nohost").json()["detail"], "URL koneksi tidak valid")
        down = self.create(f"postgresql://u:{FAKE_PASSWORD}@down.example.com/db")
        self.assertEqual(down.status_code, 400)
        self.assertTrue(down.json()["detail"].startswith("Could not connect: could not translate host name"))
        self.assertEqual(self.sql("SELECT * FROM database_connections"), [])

    def test_admin_only(self):
        self.role = "user"
        self.assertEqual(self.create().status_code, 403)
        self.assertEqual(self.http.get("/api/v1/database-connections").status_code, 403)


class ManageTest(DatabaseConnectionsTestCase):
    def test_list_refresh_activate_and_delete(self):
        first = self.create(label="  HR  ").json()["connection_id"]
        second = self.create("postgres://ext.example.com/sales").json()["connection_id"]
        listed = self.http.get("/api/v1/database-connections").json()
        self.assertEqual(listed["count"], 2)
        self.assertEqual([(c["label"], c["url"], c["table_count"]) for c in listed["connections"]],
                         [("ext.example.com/sales", "postgres://ext.example.com/sales", 0),
                          ("HR", MASKED_URL, 0)])

        refreshed = self.http.get(f"/api/v1/database-connections/{first}/tables").json()
        self.assertEqual((refreshed["connection_id"], refreshed["table_count"]), (first, 2))
        self.assertEqual(self.http.get("/api/v1/database-connections/nope/tables").status_code, 404)

        resp = self.http.post("/api/v1/database-connections/activate", json={"connection_id": first, "active": False})
        self.assertEqual(resp.json(), {"status": "success", "connection_id": first, "active": False})
        self.assertEqual(self.http.post("/api/v1/database-connections/activate",
                                        json={"connection_id": "nope", "active": True}).status_code, 404)

        self.assertEqual(self.http.delete(f"/api/v1/database-connections/{second}").json(),
                         {"status": "success", "message": "Database connection deleted", "connection_id": second})
        self.assertEqual(self.http.delete(f"/api/v1/database-connections/{second}").status_code, 404)
        self.assertEqual([c["status"] for c in self.http.get("/api/v1/database-connections").json()["connections"]], ["inactive"])

    def test_scope_keeps_other_users_connections_out_of_reach(self):
        theirs = self.create().json()["connection_id"]
        with patch.dict(os.environ, {"DB_CONNECTIONS_ENFORCE_SCOPE": "true", "DB_CONNECTIONS_USER_ID": "someone-else"}):
            self.assertEqual(self.http.get("/api/v1/database-connections").json()["count"], 0)
            self.assertEqual(self.http.get(f"/api/v1/database-connections/{theirs}/tables").status_code, 404)
            self.assertEqual(self.http.post("/api/v1/database-connections/activate",
                                            json={"connection_id": theirs, "active": False}).status_code, 404)
            self.assertEqual(self.http.delete(f"/api/v1/database-connections/{theirs}").status_code, 404)
            mine = self.create().json()["connection_id"]
            self.assertEqual([c["connection_id"] for c in self.http.get("/api/v1/database-connections").json()["connections"]], [mine])
        self.assertEqual(self.http.get("/api/v1/database-connections").json()["count"], 2)  # scope off: everything


class RagSourcesTest(DatabaseConnectionsTestCase):
    def test_only_active_and_selected_connections_with_their_real_urls(self):
        on = self.create().json()["connection_id"]
        other = self.create("postgres://ext.example.com/sales").json()["connection_id"]
        off = self.create("postgres://ext.example.com/old").json()["connection_id"]
        self.http.post("/api/v1/database-connections/activate", json={"connection_id": off, "active": False})

        resolve = lambda ids=None: asyncio.run(database_connections.resolve_active_database_connections(ids))
        self.assertEqual(sorted(c["connection_id"] for c in resolve()), sorted([on, other]))
        self.assertEqual(resolve([on, off]), [{"connection_id": on, "label": "ext.example.com/hr", "url": EXTERNAL_URL}])


class HelperTest(unittest.TestCase):
    """Pure helpers; no database needed."""

    def test_redact_and_label(self):
        self.assertEqual(database_connections._redact_url(f"postgresql://u:{FAKE_PASSWORD}@h/db"), "postgresql://u:" + "****@h/db")
        self.assertEqual(database_connections._redact_url("postgresql://h/db"), "postgresql://h/db")
        self.assertEqual(database_connections._derive_label("postgresql://h:5432"), "h/database")


if __name__ == "__main__":
    unittest.main()
