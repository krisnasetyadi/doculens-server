"""MS-657: behaviour of the payment/billing domain against real SQL, recorded
before router/payment.py is split. Runs in a throwaway schema (testing_db);
skipped when no database is reachable."""

import os
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

import stripe
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from config import config
from billing import allocation, enforcement, inflight, ledger, plans, quotas
from router import payment
from router.auth import UserRecord, get_current_user
from testing_db import IsolatedSchemaTestCase

FREE_LIMIT = config.free_plan_token_limit


class BillingTestCase(IsolatedSchemaTestCase):
    def setUp(self):
        inflight._in_flight_workspace_tokens.clear()
        inflight._in_flight_user_tokens.clear()

    # --- data -------------------------------------------------------------

    def make_admin(self) -> UserRecord:
        user_id = f"admin-{uuid.uuid4().hex[:8]}"
        self.sql("INSERT INTO users (user_id, email, password_hash, role) VALUES (%s, %s, 'x', 'admin')",
                 (user_id, f"{user_id}@example.com"))
        return UserRecord(user_id=user_id, email=f"{user_id}@example.com", role="admin", is_active=True)

    def make_member(self, admin: UserRecord) -> UserRecord:
        user_id = f"member-{uuid.uuid4().hex[:8]}"
        self.sql("INSERT INTO users (user_id, email, password_hash, role, created_by) VALUES (%s, %s, 'x', 'user', %s)",
                 (user_id, f"{user_id}@example.com", admin.user_id))
        return UserRecord(user_id=user_id, email=f"{user_id}@example.com", role="user", is_active=True)

    def pay(self, admin: UserRecord, plan_id="team", days_ago=0, cancelled=False):
        self.sql(
            """INSERT INTO payments (payment_id, user_id, plan_id, amount, currency, status, created_at, cancelled_at)
               VALUES (%s, %s, %s, 1, 'idr', 'succeeded', now() - %s * interval '1 day',
                       CASE WHEN %s THEN now() ELSE NULL END)""",
            (f"pay_{uuid.uuid4().hex}", admin.user_id, plan_id, days_ago, cancelled),
        )

    def use(self, user: UserRecord, admin: UserRecord, tokens: int, **extra):
        cols = ["user_id", "admin_user_id", "tokens", *extra]
        self.sql(f"INSERT INTO token_usage ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})",
                 (user.user_id, admin.user_id, tokens, *extra.values()))

    # --- http -------------------------------------------------------------

    def client(self, user: UserRecord) -> TestClient:
        app = FastAPI()
        app.include_router(payment.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: user
        # Checkout accepts guests: get_optional_user calls get_current_user
        # directly, so the override above doesn't reach it.
        app.dependency_overrides[payment.get_optional_user] = lambda: user
        return TestClient(app)


class PlanWindowTest(BillingTestCase):
    def test_new_workspace_is_on_the_free_plan(self):
        admin = self.make_admin()
        body = self.client(admin).get("/api/v1/payments/subscription/members").json()
        self.assertEqual((body["pool_plan_name"], body["pool_token_limit"]), ("Free", FREE_LIMIT))
        self.assertFalse(body["subscription"]["is_paid"])
        self.assertEqual(body["members"][0]["user_id"], admin.user_id)  # admin is the first row

    def test_paid_plan_sets_the_pool(self):
        admin = self.make_admin()
        self.pay(admin, "team")
        body = self.client(admin).get("/api/v1/payments/subscription/members").json()
        self.assertEqual((body["pool_plan_name"], body["pool_token_limit"]), ("Team", 10_000_000))
        self.assertEqual(body["subscription"]["subscription_status"], "active")
        self.assertTrue(body["subscription"]["is_paid"])

    def test_expired_paid_plan_falls_back_to_the_free_pool(self):
        admin = self.make_admin()
        self.pay(admin, "team", days_ago=31)
        body = self.client(admin).get("/api/v1/payments/subscription/members").json()
        self.assertEqual(body["subscription"]["subscription_status"], "expired")
        self.assertEqual(body["subscription"]["plan_name"], "Team")
        self.assertEqual(body["pool_plan_name"], "Free")

    def test_storage_limits_follow_the_workspace_plan(self):
        admin = self.make_admin()
        member = self.make_member(admin)
        limits, workspace = plans.resolve_storage_limits(member)
        self.assertEqual((limits.plan_name, workspace), ("Free", admin.user_id))


class UsageLedgerTest(BillingTestCase):
    def test_member_usage_is_logged_against_the_workspace(self):
        admin = self.make_admin()
        member = self.make_member(admin)
        ledger.log_token_usage(member.user_id, 1234)
        rows = self.sql("SELECT admin_user_id, tokens FROM token_usage WHERE user_id = %s", (member.user_id,))
        self.assertEqual([(r["admin_user_id"], r["tokens"]) for r in rows], [(admin.user_id, 1234)])

    def test_zero_tokens_are_not_logged_unless_efficient_mode(self):
        admin = self.make_admin()
        ledger.log_token_usage(admin.user_id, 0)
        ledger.log_token_usage(admin.user_id, 0, efficient_mode=True, raw_tokens_est=10, final_tokens_est=5)
        rows = self.sql("SELECT efficient_mode FROM token_usage WHERE user_id = %s", (admin.user_id,))
        self.assertEqual([r["efficient_mode"] for r in rows], [True])

    def test_rate_limit_counts_own_usage_and_blocks_at_the_cap(self):
        admin = self.make_admin()
        self.use(admin, admin, 700)
        with patch.object(config, "rate_limit_token_cap", 1000):
            body = self.client(admin).get("/api/v1/payments/rate-limit/me").json()
            self.assertEqual((body["used_tokens"], body["cap_tokens"], body["blocked"]), (700, 1000, False))
            ledger.enforce_rate_limit(admin.user_id)
            with self.assertRaises(HTTPException) as raised:
                ledger.enforce_rate_limit(admin.user_id, pending_tokens=300)
            self.assertEqual(raised.exception.status_code, 429)
            self.use(admin, admin, 300)
            body = self.client(admin).get("/api/v1/payments/rate-limit/me").json()
            self.assertTrue(body["blocked"])
            self.assertIsNotNone(body["reset_at"])

    def test_efficient_mode_stats(self):
        admin = self.make_admin()
        self.use(admin, admin, 10, efficient_mode=True, raw_tokens_est=1000, final_tokens_est=600)
        self.use(admin, admin, 10, efficient_mode=True, raw_tokens_est=1000, final_tokens_est=400)
        body = self.client(admin).get("/api/v1/payments/efficient-mode/stats").json()
        self.assertEqual(body["queries_tested"], 2)
        self.assertEqual(body["avg_reduction_pct"], 50.0)
        self.assertEqual(body["total_tokens_saved_est"], 1000)

    def test_usage_snapshot_shows_workspace_totals_to_admins_only(self):
        admin = self.make_admin()
        member = self.make_member(admin)
        self.use(member, admin, 500)
        member_view = allocation.get_usage_snapshot(member)
        admin_view = allocation.get_usage_snapshot(admin)
        self.assertEqual(member_view["my_token_used"], 500)
        self.assertNotIn("workspace_token_used", member_view)
        self.assertEqual(admin_view["workspace_token_used"], 500)


class EnforcementTest(BillingTestCase):
    def test_workspace_plan_limit(self):
        admin = self.make_admin()
        self.assertEqual(enforcement.enforce_plan_limit(admin, reserve=0), FREE_LIMIT)
        self.use(admin, admin, FREE_LIMIT)
        with self.assertRaises(HTTPException) as raised:
            enforcement.enforce_plan_limit(admin, reserve=0)
        self.assertEqual(raised.exception.status_code, 402)

    def test_gap_check_needs_a_paid_plan(self):
        admin = self.make_admin()
        with self.assertRaises(HTTPException) as raised:
            enforcement.enforce_gap_check_plan(admin)
        self.assertEqual(raised.exception.status_code, 403)
        self.pay(admin, "individual")
        enforcement.enforce_gap_check_plan(admin)
        self.assertTrue(self.client(admin).get("/api/v1/payments/subscription/me").json()["gap_check_available"])

    def test_member_without_a_row_is_capped_at_the_workspace_default(self):
        admin = self.make_admin()
        member = self.make_member(admin)
        default = config.default_member_token_allocation
        self.assertEqual(enforcement.enforce_member_allocation(member, reserve=0), default)
        self.use(member, admin, default)
        with self.assertRaises(HTTPException) as raised:
            enforcement.enforce_member_allocation(member, reserve=0)
        self.assertEqual(raised.exception.status_code, 403)

    def test_admin_without_a_row_is_uncapped(self):
        admin = self.make_admin()
        self.assertIsNone(enforcement.enforce_member_allocation(admin))


class AllocationTest(BillingTestCase):
    def allocate(self, admin, user_id, tokens, **quotas):
        return self.client(admin).post("/api/v1/payments/subscription/allocations",
                                       json={"user_id": user_id, "allocated_tokens": tokens, **quotas})

    def test_allocation_within_the_pool_is_saved(self):
        admin = self.make_admin()
        member = self.make_member(admin)
        resp = self.allocate(admin, member.user_id, 8000)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["member"]["allocated_tokens"], 8000)
        rows = self.client(admin).get("/api/v1/payments/subscription/members").json()["members"]
        self.assertEqual({r["user_id"]: r["allocated_tokens"] for r in rows}[member.user_id], 8000)

    def test_increase_beyond_the_pool_is_refused(self):
        admin = self.make_admin()
        member = self.make_member(admin)
        self.assertEqual(self.allocate(admin, member.user_id, FREE_LIMIT + 1).status_code, 400)

    def test_other_workspaces_members_are_not_found(self):
        admin, other_admin = self.make_admin(), self.make_admin()
        stranger = self.make_member(other_admin)
        self.assertEqual(self.allocate(admin, stranger.user_id, 100).status_code, 404)

    def test_quota_rules(self):
        admin = self.make_admin()
        member = self.make_member(admin)
        self.assertEqual(self.allocate(admin, member.user_id, 9000, daily_token_quota=100).status_code, 400)
        self.assertEqual(self.allocate(admin, member.user_id, 9000, daily_token_quota=600,
                                       weekly_token_quota=500).status_code, 400)
        self.assertEqual(self.allocate(admin, member.user_id, 9000, daily_token_quota=100,
                                       weekly_token_quota=9001).status_code, 400)

    def test_daily_quota_blocks_and_shows_three_tiers(self):
        admin = self.make_admin()
        member = self.make_member(admin)
        resp = self.allocate(admin, member.user_id, 9000, daily_token_quota=100, weekly_token_quota=1000)
        self.assertEqual(resp.status_code, 200)
        self.use(member, admin, 100)
        usage = self.client(member).get("/api/v1/payments/subscription/me").json()["usage"]
        tiers = {t["interval"]: t for t in usage["quota_tiers"]}
        self.assertEqual(set(tiers), {"daily", "weekly", "monthly"})
        self.assertTrue(tiers["daily"]["blocked"])
        self.assertFalse(tiers["weekly"]["blocked"])
        with self.assertRaises(HTTPException) as raised:
            enforcement.enforce_member_allocation(member)
        self.assertIn("Daily", raised.exception.detail)

    def test_default_allocation_setting(self):
        admin = self.make_admin()
        client = self.client(admin)
        self.assertEqual(client.get("/api/v1/payments/subscription/settings").json(),
                         {"default_member_allocation": config.default_member_token_allocation})
        self.assertEqual(client.put("/api/v1/payments/subscription/settings",
                                    json={"default_member_allocation": FREE_LIMIT + 1}).status_code, 400)
        client.put("/api/v1/payments/subscription/settings", json={"default_member_allocation": 7000})
        self.assertEqual(client.get("/api/v1/payments/subscription/settings").json()["default_member_allocation"], 7000)

    def test_initial_allocation_is_clamped_to_the_pool_and_released_on_delete(self):
        admin = self.make_admin()
        member = self.make_member(admin)
        self.assertEqual(allocation.assign_initial_allocation(admin.user_id, member.user_id, FREE_LIMIT * 2),
                         (FREE_LIMIT, True))
        row = self.sql("SELECT daily_token_quota, weekly_token_quota FROM token_allocations WHERE user_id = %s",
                       (member.user_id,))[0]
        self.assertEqual((row["daily_token_quota"], row["weekly_token_quota"]),
                         quotas.default_quota_limits(FREE_LIMIT))
        allocation.release_member_allocation(member.user_id)
        self.assertEqual(self.sql("SELECT * FROM token_allocations WHERE user_id = %s", (member.user_id,)), [])


class CancelResumeTest(BillingTestCase):
    def test_free_plan_cannot_be_cancelled(self):
        admin = self.make_admin()
        self.assertEqual(self.client(admin).post("/api/v1/payments/subscription/cancel").status_code, 400)

    def test_cancel_then_resume(self):
        admin = self.make_admin()
        self.pay(admin, "team")
        client = self.client(admin)
        cancelled = client.post("/api/v1/payments/subscription/cancel").json()
        self.assertTrue(cancelled["cancel_at_period_end"])
        self.assertIsNone(cancelled["next_reset_date"])
        self.assertEqual(client.post("/api/v1/payments/subscription/cancel").status_code, 400)
        resumed = client.post("/api/v1/payments/subscription/resume").json()
        self.assertFalse(resumed["cancel_at_period_end"])
        self.assertEqual(client.post("/api/v1/payments/subscription/resume").status_code, 400)


class TokenRequestTest(BillingTestCase):
    def test_request_list_dismiss(self):
        admin = self.make_admin()
        member = self.make_member(admin)
        member_client, admin_client = self.client(member), self.client(admin)

        created = member_client.post("/api/v1/payments/subscription/request-more", json={"message": "more please"})
        self.assertEqual(created.json()["request"]["status"], "pending")
        self.assertEqual(member_client.post("/api/v1/payments/subscription/request-more", json={}).status_code, 400)

        listed = admin_client.get("/api/v1/payments/subscription/requests").json()
        self.assertEqual(listed["pending_count"], 1)
        self.assertEqual(listed["requests"][0]["email"], member.email)

        request_id = created.json()["request"]["request_id"]
        dismissed = admin_client.post(f"/api/v1/payments/subscription/requests/{request_id}/dismiss").json()
        self.assertEqual(dismissed["request"]["status"], "resolved")
        self.assertEqual(admin_client.post("/api/v1/payments/subscription/requests/treq_nope/dismiss").status_code, 404)
        self.assertEqual(member_client.post("/api/v1/payments/subscription/request-more", json={}).status_code, 200)


class StripeCheckoutTest(BillingTestCase):
    def test_guest_checkout_is_attributed_to_guest(self):
        session_id = f"cs_test_{uuid.uuid4().hex[:8]}"
        fake_session = SimpleNamespace(id=session_id, url="https://checkout.example/x")
        app = FastAPI()
        app.include_router(payment.router, prefix="/api/v1")
        with patch.object(config, "stripe_secret_key", "sk_test_dummy"), \
                patch.object(stripe.checkout.Session, "create", return_value=fake_session):
            self.assertEqual(TestClient(app).post("/api/v1/payments/checkout-session",
                                                  json={"plan_id": "team"}).status_code, 201)
        rows = self.sql("SELECT user_id FROM payments WHERE stripe_checkout_session_id = %s", (session_id,))
        self.assertEqual([r["user_id"] for r in rows], ["guest"])

    def test_unknown_plan_is_refused(self):
        admin = self.make_admin()
        self.assertEqual(self.client(admin).post("/api/v1/payments/checkout-session",
                                                 json={"plan_id": "enterprise"}).status_code, 400)

    def checkout(self, admin, session_id):
        fake_session = SimpleNamespace(id=session_id, url=f"https://checkout.example/{session_id}")
        with patch.object(config, "stripe_secret_key", "sk_test_dummy"), \
                patch.object(stripe.checkout.Session, "create", return_value=fake_session) as create:
            resp = self.client(admin).post("/api/v1/payments/checkout-session", json={"plan_id": "team"})
        return resp, create

    def webhook(self, event_type, data):
        event = {"type": event_type, "data": {"object": SimpleNamespace(to_dict=lambda: data)}}
        with patch.object(config, "stripe_webhook_secret", "whsec_dummy"), \
                patch.object(stripe.Webhook, "construct_event", return_value=event):
            return TestClient(self.client(self.make_admin()).app).post(
                "/api/v1/payments/webhook", content=b"{}", headers={"stripe-signature": "t=1,v1=x"})

    def test_checkout_records_a_pending_payment_and_the_webhook_completes_it(self):
        admin = self.make_admin()
        session_id = f"cs_test_{uuid.uuid4().hex[:8]}"
        resp, create = self.checkout(admin, session_id)
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(create.call_args.kwargs["line_items"][0]["price_data"]["unit_amount"],
                         plans.PLAN_PRICES["team"]["amount"])
        record = self.client(admin).get(f"/api/v1/payments/session/{session_id}").json()["payment"]
        self.assertEqual(record["status"], "pending")

        self.assertEqual(self.webhook("checkout.session.completed",
                                      {"id": session_id, "payment_status": "paid", "payment_intent": "pi_1"}).json(),
                         {"received": True})
        record = self.client(admin).get(f"/api/v1/payments/session/{session_id}").json()["payment"]
        self.assertEqual(record["status"], "succeeded")
        members = self.client(admin).get("/api/v1/payments/subscription/members").json()
        self.assertEqual(members["pool_plan_name"], "Team")

    def test_expired_checkout_is_cancelled(self):
        admin = self.make_admin()
        session_id = f"cs_test_{uuid.uuid4().hex[:8]}"
        self.checkout(admin, session_id)
        self.webhook("checkout.session.expired", {"id": session_id})
        record = self.client(admin).get(f"/api/v1/payments/session/{session_id}").json()["payment"]
        self.assertEqual(record["status"], "cancelled")

    def test_unknown_session_is_not_found(self):
        admin = self.make_admin()
        self.assertEqual(self.client(admin).get("/api/v1/payments/session/cs_nope").status_code, 404)

    def test_webhook_without_a_secret_is_acknowledged(self):
        with patch.object(config, "stripe_webhook_secret", None):
            resp = self.client(self.make_admin()).post("/api/v1/payments/webhook", content=b"{}")
        self.assertEqual(resp.json(), {"received": True})


if __name__ == "__main__":
    unittest.main()
