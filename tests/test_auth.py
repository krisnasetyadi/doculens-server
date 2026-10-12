"""MS-657: behaviour of authentication and team management (router/ +
services/auth.py) against real SQL, recorded before it is moved. Real tokens go through
get_current_user, no dependency overrides. Runs in a throwaway schema
(testing_db); skipped when no database is reachable."""

import os
import unittest
import uuid

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from router import auth
from services import auth as auth_service
from tests.testing_db import IsolatedSchemaTestCase

PASSWORD = "correct-horse-1"


class AuthTestCase(IsolatedSchemaTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        app = FastAPI()
        app.include_router(auth.router, prefix="/api/v1")
        cls.http = TestClient(app)

    def setUp(self):
        auth_service._failed_login_attempts.clear()

    def email(self):
        return f"u{uuid.uuid4().hex[:8]}@example.com"

    def login(self, email, password=PASSWORD):
        return self.http.post("/api/v1/auth/login", json={"email": email, "password": password})

    def bearer(self, token):
        return {"Authorization": f"Bearer {token}"}

    def make_admin(self):
        """A user promoted to admin directly (registration only ever creates one admin)."""
        email = self.email()
        resp = self.http.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD})
        self.sql("UPDATE users SET role = 'admin' WHERE email = %s", (email,))
        token = self.login(email).json()["access_token"]
        return resp.json()["user_id"], self.bearer(token)

    def add_member(self, admin_headers, email=None, **extra):
        return self.http.post("/api/v1/auth/admin/users", headers=admin_headers,
                              json={"email": email or self.email(), "password": PASSWORD, **extra})


class FirstAccountTest(AuthTestCase):
    """Own class, so its schema starts with no users."""

    def test_first_account_is_admin_and_later_ones_are_users(self):
        first = self.http.post("/api/v1/auth/register", json={"email": self.email(), "password": PASSWORD, "name": "  Ann "})
        self.assertEqual(first.status_code, 201)
        self.assertEqual((first.json()["role"], first.json()["name"]), ("admin", "Ann"))
        second = self.http.post("/api/v1/auth/register", json={"email": self.email(), "password": PASSWORD})
        self.assertEqual(second.json()["role"], "user")


class RegisterLoginTest(AuthTestCase):
    def test_duplicate_email_and_oversized_password(self):
        email = self.email()
        self.http.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD})
        self.assertEqual(self.http.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD}).status_code, 409)
        self.assertEqual(self.http.post("/api/v1/auth/register", json={"email": self.email(), "password": "é" * 37}).status_code, 422)

    def test_login_and_wrong_password(self):
        email = self.email()
        self.http.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD})
        ok = self.login(email)
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["email"], email)
        self.assertEqual(self.login(email, "wrong-password").status_code, 401)
        self.assertEqual(self.login(self.email()).status_code, 401)  # unknown email, same answer

    def test_failed_logins_are_rate_limited(self):
        email = self.email()
        self.http.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD})
        for _ in range(auth_service.LOGIN_ATTEMPT_LIMIT):
            self.assertEqual(self.login(email, "wrong-password").status_code, 401)
        self.assertEqual(self.login(email).status_code, 429)  # even the right password, until the window passes


class TokenTest(AuthTestCase):
    def test_me_with_a_real_token(self):
        email = self.email()
        token = self.http.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD}).json()["access_token"]
        me = self.http.get("/api/v1/auth/me", headers=self.bearer(token))
        self.assertEqual((me.status_code, me.json()["email"]), (200, email))

    def test_missing_invalid_and_revoked_tokens(self):
        self.assertEqual(self.http.get("/api/v1/auth/me").status_code, 401)
        self.assertEqual(self.http.get("/api/v1/auth/me", headers=self.bearer("not-a-jwt")).status_code, 401)
        email = self.email()
        body = self.http.post("/api/v1/auth/register", json={"email": email, "password": PASSWORD}).json()
        self.sql("DELETE FROM users WHERE user_id = %s", (body["user_id"],))
        self.assertEqual(self.http.get("/api/v1/auth/me", headers=self.bearer(body["access_token"])).status_code, 401)

    def test_admin_routes_need_the_admin_role(self):
        token = self.http.post("/api/v1/auth/register", json={"email": self.email(), "password": PASSWORD}).json()["access_token"]
        self.sql("UPDATE users SET role = 'user'")  # whichever account registered first is not under test here
        self.assertEqual(self.http.get("/api/v1/auth/admin/users", headers=self.bearer(token)).status_code, 403)


class ProfileTest(AuthTestCase):
    def setUp(self):
        super().setUp()
        self.email_addr = self.email()
        token = self.http.post("/api/v1/auth/register", json={"email": self.email_addr, "password": PASSWORD}).json()["access_token"]
        self.headers = self.bearer(token)

    def test_update_name_and_avatar_then_remove_avatar(self):
        updated = self.http.post("/api/v1/auth/me", headers=self.headers,
                                 json={"name": "Budi", "avatar_url": "data:image/png;base64,AAAA"}).json()
        self.assertEqual((updated["name"], updated["avatar_url"]), ("Budi", "data:image/png;base64,AAAA"))
        cleared = self.http.post("/api/v1/auth/me", headers=self.headers, json={"remove_avatar": True}).json()
        self.assertEqual((cleared["name"], cleared["avatar_url"]), ("Budi", None))

    def test_invalid_profile_updates(self):
        self.assertEqual(self.http.post("/api/v1/auth/me", headers=self.headers, json={}).status_code, 400)
        self.assertEqual(self.http.post("/api/v1/auth/me", headers=self.headers,
                                        json={"avatar_url": "https://example.com/a.png"}).status_code, 422)
        self.assertEqual(self.http.post("/api/v1/auth/me", headers=self.headers, json={"name": "   "}).status_code, 422)

    def test_change_password(self):
        self.assertEqual(self.http.post("/api/v1/auth/change-password", headers=self.headers,
                                        json={"current_password": "nope", "new_password": "new-pass-123"}).status_code, 401)
        self.assertEqual(self.http.post("/api/v1/auth/change-password", headers=self.headers,
                                        json={"current_password": PASSWORD, "new_password": "new-pass-123"}).status_code, 200)
        self.assertEqual(self.login(self.email_addr).status_code, 401)
        self.assertEqual(self.login(self.email_addr, "new-pass-123").status_code, 200)


class TeamTest(AuthTestCase):
    def test_add_and_list_members(self):
        admin_id, headers = self.make_admin()
        added = self.add_member(headers)
        self.assertEqual(added.status_code, 201)
        self.assertEqual(added.json()["role"], "user")
        self.assertIsNotNone(added.json()["allocated_tokens"])   # MS-402 initial allocation
        listed = self.http.get("/api/v1/auth/admin/users", headers=headers).json()
        self.assertEqual(listed["max_sub_users"], 5)
        self.assertEqual([m["user_id"] for m in listed["members"]], [added.json()["user_id"]])

    def test_member_validation_duplicates_and_limit(self):
        admin_id, headers = self.make_admin()
        self.assertEqual(self.http.post("/api/v1/auth/admin/users", headers=headers,
                                        json={"email": self.email(), "password": "short"}).status_code, 422)
        self.assertEqual(self.http.post("/api/v1/auth/admin/users", headers=headers,
                                        json={"email": "x" * 40 + "@example.com", "password": PASSWORD}).status_code, 422)
        email = self.email()
        self.add_member(headers, email)
        self.assertEqual(self.add_member(headers, email).status_code, 409)
        self.sql("UPDATE users SET max_sub_users = 1 WHERE user_id = %s", (admin_id,))
        self.assertEqual(self.add_member(headers).status_code, 403)

    def test_deactivated_member_cannot_log_in_or_use_a_token(self):
        admin_id, headers = self.make_admin()
        email = self.email()
        member_id = self.add_member(headers, email).json()["user_id"]
        member_token = self.login(email).json()["access_token"]
        resp = self.http.post("/api/v1/auth/admin/users/activate", headers=headers, json={"user_id": member_id, "active": False})
        self.assertEqual(resp.json(), {"status": "success", "user_id": member_id, "active": False})
        self.assertEqual(self.login(email).status_code, 403)
        self.assertEqual(self.http.get("/api/v1/auth/me", headers=self.bearer(member_token)).status_code, 401)

    def test_admins_only_manage_their_own_members(self):
        _, headers = self.make_admin()
        _, other_headers = self.make_admin()
        member_id = self.add_member(other_headers).json()["user_id"]
        for resp in (
            self.http.post("/api/v1/auth/admin/users/activate", headers=headers, json={"user_id": member_id, "active": False}),
            self.http.put(f"/api/v1/auth/admin/users/{member_id}", headers=headers, json={"name": "x"}),
            self.http.delete(f"/api/v1/auth/admin/users/{member_id}", headers=headers),
            self.http.post("/api/v1/auth/admin/reset-password", headers=headers, json={"user_id": member_id, "new_password": "new-pass-123"}),
        ):
            self.assertEqual(resp.status_code, 404)

    def test_rename_and_reset_password(self):
        _, headers = self.make_admin()
        email = self.email()
        member_id = self.add_member(headers, email).json()["user_id"]
        self.assertEqual(self.http.put(f"/api/v1/auth/admin/users/{member_id}", headers=headers, json={}).status_code, 400)
        self.assertEqual(self.http.put(f"/api/v1/auth/admin/users/{member_id}", headers=headers,
                                       json={"name": "Citra"}).json()["name"], "Citra")
        reset = self.http.post("/api/v1/auth/admin/reset-password", headers=headers,
                               json={"user_id": member_id, "new_password": "new-pass-123"})
        self.assertEqual(reset.json(), {"message": f"Password reset for {email}"})
        self.assertEqual(self.login(email, "new-pass-123").status_code, 200)

    def test_delete_member_removes_their_chats_and_allocation(self):
        _, headers = self.make_admin()
        member_id = self.add_member(headers).json()["user_id"]
        self.sql("INSERT INTO chat_sessions (session_id, owner_id) VALUES (%s, %s)", (f"s-{member_id}", member_id))
        self.assertEqual(self.http.delete(f"/api/v1/auth/admin/users/{member_id}", headers=headers).json(),
                         {"status": "success", "user_id": member_id})
        self.assertEqual(self.sql("SELECT 1 FROM users WHERE user_id = %s", (member_id,)), [])
        self.assertEqual(self.sql("SELECT 1 FROM chat_sessions WHERE owner_id = %s", (member_id,)), [])
        self.assertEqual(self.sql("SELECT 1 FROM token_allocations WHERE user_id = %s", (member_id,)), [])


if __name__ == "__main__":
    unittest.main()
