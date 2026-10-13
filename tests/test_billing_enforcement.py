"""In-flight token reservations against plan and member caps, and the
Compliance Gap Check plan gate (subscription flag and endpoint refusal).

Split out of test_conversation_language.py by MS-657."""

import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET", "conversation-local-test-secret-not-for-production")

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from services.rag.processor import processor
import db
from services.billing import allocation, enforcement, inflight, plans
from router import payment
from models import UserRecord
from services.auth import get_current_user


class InFlightReservationTests(unittest.TestCase):
    """The workspace lock is held only for the quota check; the LLM call
    runs outside it, guarded by in-flight reservations (services/billing/inflight.py)."""

    def tearDown(self):
        inflight._in_flight_workspace_tokens.clear()
        inflight._in_flight_user_tokens.clear()

    def test_reservation_bookkeeping(self):
        first = inflight.begin_in_flight("w1", "u1", 2000)
        second = inflight.begin_in_flight("w1", "u2", 2000)
        self.assertEqual(inflight.in_flight_tokens("w1", "u1"), (4000, 2000))
        inflight.end_in_flight(first)
        inflight.end_in_flight(second)
        self.assertEqual(inflight.in_flight_tokens("w1", "u1"), (0, 0))
        self.assertEqual(inflight._in_flight_workspace_tokens, {})

    def test_reservation_clamped_to_half_of_each_cap(self):
        # A 100k gap check on a 60k Free pool / a member's 5k allocation.
        reservation = inflight.begin_in_flight("w1", "u1", 100_000, workspace_cap=60_000, user_cap=5_000)
        self.assertEqual((reservation.workspace_tokens, reservation.user_tokens), (30_000, 2_500))
        uncapped = inflight.begin_in_flight("w2", "u2", 5_000)
        self.assertEqual((uncapped.workspace_tokens, uncapped.user_tokens), (5_000, 5_000))

    def test_chat_still_allowed_while_gap_check_in_flight(self):
        window = MagicMock(plan={"token_limit": 60_000, "name": "Free"})
        user = UserRecord(user_id="u1", email="u1@example.com", role="admin", is_active=True)
        with patch.object(db, "get_conn", return_value=MagicMock()), \
                patch.object(enforcement, "resolve_admin_user_id", return_value="w1"), \
                patch.object(enforcement, "get_enforced_window", return_value=window), \
                patch.object(enforcement, "sum_tokens_for_admin", return_value=5_000):
            cap = enforcement.enforce_plan_limit(user, 0, 100_000)
            inflight.begin_in_flight("w1", "u1", 100_000, workspace_cap=cap)
            pending_workspace, _ = inflight.in_flight_tokens("w1", "u1")
            # 5k used + 30k (clamped) gap check + 2k chat headroom fits in 60k.
            self.assertEqual(enforcement.enforce_plan_limit(user, pending_workspace), 60_000)

    def test_plan_limit_counts_in_flight_reservations(self):
        window = MagicMock(plan={"token_limit": 10_000, "name": "Free"})
        with patch.object(db, "get_conn", return_value=MagicMock()), \
                patch.object(enforcement, "resolve_admin_user_id", return_value="w1"), \
                patch.object(enforcement, "get_enforced_window", return_value=window), \
                patch.object(enforcement, "sum_tokens_for_admin", return_value=5_000):
            user = UserRecord(user_id="u1", email="u1@example.com", role="admin", is_active=True)
            enforcement.enforce_plan_limit(user, 0, 2_000)  # 5k used + 2k reserve fits in 10k
            with self.assertRaises(HTTPException) as blocked:
                enforcement.enforce_plan_limit(user, 4_000, 2_000)  # + 4k in flight no longer fits
            self.assertEqual(blocked.exception.status_code, 402)


class GapCheckPlanTests(unittest.TestCase):
    def _gate(self, plan_id):
        window = MagicMock(plan=plans.PLAN_QUOTAS[plan_id])
        user = UserRecord(user_id="u1", email="u1@example.com", role="member", is_active=True)
        with patch.object(db, "get_conn", return_value=MagicMock()), \
                patch.object(enforcement, "resolve_admin_user_id", return_value="w1"), \
                patch.object(enforcement, "get_enforced_window", return_value=window):
            enforcement.enforce_gap_check_plan(user)

    def test_paid_plans_allowed(self):
        self._gate("individual")
        self._gate("team")

    def test_free_refused(self):
        with self.assertRaises(HTTPException) as refused:
            self._gate("free")
        self.assertEqual(refused.exception.status_code, 403)
        self.assertIn("plan Individual dan Team", refused.exception.detail)

    def test_usage_endpoint_reports_gap_check_availability(self):
        app = FastAPI()
        app.include_router(payment.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="u1", email="u1@example.com", role="admin", is_active=True
        )
        client = TestClient(app)
        for plan_id, expected in (("free", False), ("individual", True), ("team", True)):
            window = MagicMock(plan=plans.PLAN_QUOTAS[plan_id])
            with patch.object(db, "get_conn", return_value=MagicMock()), \
                    patch.object(allocation, "resolve_admin_user_id", return_value="u1"), \
                    patch.object(allocation, "get_enforced_window", return_value=window), \
                    patch.object(allocation, "get_user_allocation", return_value=None), \
                    patch.object(allocation, "sum_tokens_for_user", return_value=0):
                body = client.get("/api/v1/payments/subscription/me").json()
            self.assertEqual(body["gap_check_available"], expected, plan_id)

    def test_gap_check_endpoint_refuses_non_team_before_any_llm_work(self):
        from router import compliance
        from services import compliance as compliance_service

        app = FastAPI()
        app.include_router(compliance.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="u1", email="u1@example.com", role="admin", is_active=True
        )
        refusal = HTTPException(status_code=403, detail="Compliance Gap Check hanya tersedia untuk plan Individual dan Team")
        with patch.object(compliance_service.supabase_storage, "list_collection_ids_for_user", return_value=["r1", "t1"]), \
                patch.object(compliance_service, "enforce_gap_check_plan", side_effect=refusal), \
                patch.object(processor, "run_compliance_gap_check") as run:
            response = TestClient(app).post("/api/v1/analysis/gap-analysis", json={
                "skill_id": "compliance_gap_check", "reference_collection_ids": ["r1"],
                "target_collection_ids": ["t1"], "framework_name": "ISO 27001",
            })
        self.assertEqual(response.status_code, 403)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
