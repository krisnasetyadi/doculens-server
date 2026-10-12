"""MS-657 part 5: unexpected errors never reach the client as raw exception
text. The central handler (services/errors.py) answers a generic 500 with an
error_id that is also in the log, and the routes that used to put str(e) in
their 500 detail now send a fixed message."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from models import UserRecord
from router import auth, collections
from services import collections as collection_service
from services import errors
from services.auth import get_current_user
from services.rag.processor import processor
from tests.test_agnostic import AgnosticTestCase

SECRET = "postgresql-password-hunter2"


class CentralHandlerTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        errors.register(app)

        @app.get("/boom")
        def boom():
            raise RuntimeError(f"could not connect with {SECRET}")

        @app.get("/refused")
        def refused():
            raise HTTPException(status_code=403, detail="Not allowed")

        self.http = TestClient(app, raise_server_exceptions=False)

    def test_unexpected_errors_get_a_generic_500_with_an_id_that_is_logged(self):
        with self.assertLogs("services.errors", level="ERROR") as logs:
            resp = self.http.get("/boom")
        self.assertEqual(resp.status_code, 500)
        body = resp.json()
        self.assertEqual(body["detail"], "Internal server error")
        self.assertNotIn(SECRET, resp.text)
        self.assertIn(body["error_id"], logs.output[0])
        self.assertIn(SECRET, logs.output[0])  # the details stay in the log
        self.assertIn("GET /boom", logs.output[0])

    def test_http_exceptions_pass_through_unchanged(self):
        resp = self.http.get("/refused")
        self.assertEqual((resp.status_code, resp.json()), (403, {"detail": "Not allowed"}))


class AgnosticLeakTest(AgnosticTestCase):
    def test_a_failed_query_does_not_return_the_exception_text(self):
        self.m["search"].side_effect = RuntimeError(SECRET)
        resp = self.ask()
        self.assertEqual(resp.status_code, 500)
        self.assertNotIn(SECRET, resp.text)


class FailingConn:
    def cursor(self):
        raise RuntimeError(SECRET)

    def close(self):
        pass


class AuthLeakTest(unittest.TestCase):
    def test_a_failed_login_does_not_return_the_exception_text(self):
        app = FastAPI()
        app.include_router(auth.router, prefix="/api/v1")
        with patch("db.get_conn", lambda source="": FailingConn()):
            resp = TestClient(app).post("/api/v1/auth/login", json={"email": "a@example.com", "password": "pw-123456"})
        self.assertEqual((resp.status_code, resp.json()), (500, {"detail": "Login failed"}))


class CollectionsLeakTest(unittest.TestCase):
    def test_failed_collection_routes_do_not_return_the_exception_text(self):
        app = FastAPI()
        app.include_router(collections.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="a1", email="a@example.com", role="admin", is_active=True)
        http = TestClient(app)
        boom = RuntimeError(SECRET)
        with patch.object(collection_service.supabase_storage, "has_database", side_effect=boom), \
                patch.object(collection_service.supabase_storage, "is_enabled", side_effect=boom), \
                patch.object(collection_service.supabase_storage, "get_collection", side_effect=boom), \
                patch.object(processor, "invalidate_cache"):
            responses = [http.get("/api/v1/pdf-collections"),
                         http.delete("/api/v1/pdf-collections/c1"),
                         http.get("/api/v1/files/c1/a.pdf"),
                         http.get("/api/v1/collection/c1/files")]
        for resp in responses:
            self.assertEqual(resp.status_code, 500, resp.request.url)
            self.assertNotIn(SECRET, resp.text)


class MainAppTest(unittest.TestCase):
    def test_the_app_registers_the_handler(self):
        import main
        self.assertIs(main.app.exception_handlers.get(Exception), errors.handle_unexpected)


if __name__ == "__main__":
    unittest.main()
