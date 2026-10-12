"""MS-657: behaviour of the gap analysis endpoints (router/ + services/compliance_service.py)
against real SQL, recorded before they are moved: request checks and
collection access, a metered compliance gap check (the LLM part is faked at
the processor), the scenario scaffold, run history, viewing, deleting and
exporting. Plan checks are passed through; their own tests live in
test_conversation_language.py and test_billing.py. Runs in a throwaway
schema (testing_db); skipped when no database is reachable."""

import os
import uuid
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.storage import supabase as storage
from config import config
from models import UserRecord
from processor import processor
from router import compliance
from services import compliance as compliance_service
from services.auth import get_current_user
from services.billing import inflight
from tests.testing_db import IsolatedSchemaTestCase

ITEMS = [
    {"label": "A.5.1 Policies", "status": "met", "evidence": "Policy | approved\nyearly",
     "source_citation": "policy.pdf", "recommendation": None, "target_collection_id": "t"},
    {"label": "A.8.2 Access", "status": "not_met", "evidence": None,
     "source_citation": None, "recommendation": "Add an access review", "target_collection_id": "t"},
    {"label": "A.9 Backup", "status": "partial", "evidence": "Weekly only",
     "source_citation": None, "recommendation": "Daily backups", "target_collection_id": "t"},
]


def user(role="user") -> UserRecord:
    uid = f"{role}-{uuid.uuid4().hex[:8]}"
    return UserRecord(user_id=uid, email=f"{uid}@example.com", role=role, is_active=True)


class ComplianceTestCase(IsolatedSchemaTestCase):
    def setUp(self):
        inflight._in_flight_workspace_tokens.clear()
        inflight._in_flight_user_tokens.clear()
        patches = {
            "plan_check": patch.object(compliance_service, "enforce_gap_check_plan"),
            "workspace": patch.object(compliance_service, "resolve_workspace_id", side_effect=lambda u: f"w-{u.user_id}"),
            "rate": patch.object(compliance_service, "enforce_rate_limit"),
            "plan": patch.object(compliance_service, "enforce_plan_limit", return_value=None),
            "member": patch.object(compliance_service, "enforce_member_allocation", return_value=None),
            "log": patch.object(compliance_service, "log_token_usage"),
            "run": patch.object(processor, "run_compliance_gap_check", return_value=(ITEMS, "AI generated", 1234)),
        }
        self.m = {name: p.start() for name, p in patches.items()}
        self.addCleanup(patch.stopall)
        self.owner = user()
        self.reference = self.collection(self.owner, "ISO 27001")
        self.target = self.collection(self.owner, "Our policies")

    def collection(self, owner, title):
        cid = str(uuid.uuid4())
        self.assertTrue(storage.register_collection(cid, [f"{title}.pdf"], 3, title=title, owner_id=owner.user_id))
        return cid

    def client(self, who):
        app = FastAPI()
        app.include_router(compliance.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: who
        return TestClient(app)

    def run_check(self, who=None, **body):
        payload = {"skill_id": "compliance_gap_check", "reference_collection_ids": [self.reference],
                   "target_collection_ids": [self.target], "framework_name": "ISO 27001", **body}
        return self.client(who or self.owner).post("/api/v1/analysis/gap-analysis", json=payload)


class RunTest(ComplianceTestCase):
    def test_a_gap_check_is_metered_stored_and_summarised(self):
        resp = self.run_check()
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["summary"], {"met": 1, "partial": 1, "not_met": 1, "unknown": 0})
        self.assertEqual((body["disclaimer"], len(body["items"])), ("AI generated", 3))
        run = body["run"]
        self.assertEqual((run["skill_id"], run["framework_name"], run["reference_collection_ids"], run["target_collection_ids"]),
                         ("compliance_gap_check", "ISO 27001", [self.reference], [self.target]))
        self.m["run"].assert_called_once_with(reference_collection_ids=[self.reference],
                                              target_collection_ids=[self.target], framework_name="ISO 27001")
        self.m["log"].assert_called_once_with(self.owner.user_id, 1234)
        reserve = config.gap_check_token_reserve
        self.m["plan"].assert_called_once_with(self.owner, 0, reserve)
        self.assertEqual(inflight.in_flight_tokens(f"w-{self.owner.user_id}", self.owner.user_id), (0, 0))

        stored = self.client(self.owner).get(f"/api/v1/analysis/gap-analysis/{run['run_id']}").json()
        self.assertEqual(([i["label"] for i in stored["items"]], stored["summary"], stored["disclaimer"]),
                         ([i["label"] for i in ITEMS], body["summary"], None))

    def test_request_checks(self):
        self.assertEqual(self.run_check(skill_id="other").status_code, 400)
        self.assertEqual(self.run_check(reference_collection_ids=[]).status_code, 400)
        self.assertEqual(self.run_check(target_collection_ids=[]).status_code, 400)
        stranger = user()
        self.assertEqual(self.run_check(stranger).status_code, 403)  # someone else's collections
        self.assertEqual(self.run_check(reference_collection_ids=["nope"]).status_code, 403)
        self.assertEqual(self.run_check(user("admin")).status_code, 200)  # admins reach every collection
        self.assertEqual(self.m["run"].call_count, 1)

    def test_a_failed_check_frees_the_reservation_and_stores_nothing(self):
        self.m["run"].side_effect = RuntimeError("llm down")
        with self.assertRaises(RuntimeError):
            self.run_check()
        self.assertEqual(inflight.in_flight_tokens(f"w-{self.owner.user_id}", self.owner.user_id), (0, 0))
        self.assertEqual(self.client(self.owner).get("/api/v1/analysis/gap-analysis/runs").json(), [])

    def test_scenario_skill_is_a_scaffold_without_llm_or_metering(self):
        resp = self.client(self.owner).post("/api/v1/analysis/gap-analysis", json={
            "skill_id": "scenario_regulatory_impact", "reference_collection_ids": [self.reference],
            "framework_name": "PPh", "scenario_input": "loan vs equity"})
        body = resp.json()
        self.assertEqual((body["items"], body["disclaimer"]), ([], compliance_service.SCENARIO_SCAFFOLD_DISCLAIMER))
        self.assertEqual(body["run"]["scenario_input"], "loan vs equity")
        self.m["run"].assert_not_called()
        self.m["plan_check"].assert_not_called()
        self.m["log"].assert_not_called()


class HistoryTest(ComplianceTestCase):
    def test_runs_are_listed_viewed_and_deleted_by_their_owner(self):
        mine = self.run_check().json()["run"]["run_id"]
        stranger, admin = user(), user("admin")
        self.assertEqual([r["run_id"] for r in self.client(self.owner).get("/api/v1/analysis/gap-analysis/runs").json()], [mine])
        self.assertEqual(self.client(stranger).get("/api/v1/analysis/gap-analysis/runs").json(), [])
        self.assertIn(mine, [r["run_id"] for r in self.client(admin).get("/api/v1/analysis/gap-analysis/runs").json()])

        for method in ("get", "delete"):
            self.assertEqual(getattr(self.client(stranger), method)(f"/api/v1/analysis/gap-analysis/{mine}").status_code, 403)
            self.assertEqual(getattr(self.client(self.owner), method)("/api/v1/analysis/gap-analysis/nope").status_code, 404)
        self.assertEqual(self.client(admin).get(f"/api/v1/analysis/gap-analysis/{mine}").status_code, 200)

        self.assertEqual(self.client(self.owner).delete(f"/api/v1/analysis/gap-analysis/{mine}").json(),
                         {"run_id": mine, "deleted": True})
        self.assertEqual(self.client(self.owner).get(f"/api/v1/analysis/gap-analysis/{mine}").status_code, 404)


class ExportTest(ComplianceTestCase):
    def test_markdown_and_pdf(self):
        run_id = self.run_check().json()["run"]["run_id"]
        url = f"/api/v1/analysis/gap-analysis/{run_id}/export"
        md = self.client(self.owner).get(url)
        self.assertEqual(md.headers["content-disposition"], f'attachment; filename="gap-analysis-{run_id}.md"')
        self.assertTrue(md.headers["content-type"].startswith("text/markdown"))
        lines = md.text.splitlines()
        self.assertEqual(lines[0], "# Laporan Gap Analysis — ISO 27001")
        self.assertIn(f"- Reference collection: {self.reference}", lines)
        self.assertIn("| A.5.1 Policies | policy.pdf | met | Policy / approved yearly |  |", lines)
        self.assertIn("| A.8.2 Access | t | not_met |  | Add an access review |", lines)

        pdf = self.client(self.owner).get(url, params={"format": "pdf"})
        self.assertEqual((pdf.headers["content-type"], pdf.content[:5]), ("application/pdf", b"%PDF-"))
        self.assertEqual(self.client(self.owner).get(url, params={"format": "docx"}).status_code, 422)
        self.assertEqual(self.client(user()).get(url).status_code, 403)
        self.assertEqual(self.client(self.owner).get("/api/v1/analysis/gap-analysis/nope/export").status_code, 404)


if __name__ == "__main__":
    import unittest
    unittest.main()
