"""MS-657: behaviour of the gap analysis endpoints (router/ + services/compliance_service.py)
against real SQL, recorded before they are moved: request checks and
collection access, a metered compliance gap check (the LLM part is faked at
the processor), the scenario scaffold, run history, viewing, deleting and
exporting. Plan checks are passed through; their own tests live in
test_billing_enforcement.py and test_billing.py. Runs in a throwaway
schema (testing_db); skipped when no database is reachable."""

import os
import uuid
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.storage import supabase as storage
from config import config
import db
from models import UserRecord
from services.rag.processor import processor
from router import compliance
from services import compliance as compliance_service
from services.auth import get_current_user
from services.billing import inflight
from tests.testing_db import IsolatedSchemaTestCase

POINTER = {"file_name": "policy.pdf", "collection_id": "t", "kind": "pdf", "page": 2,
           "section": None, "paragraph": None, "sheet": None, "row": None, "label": "policy.pdf · p. 2"}
ITEMS = [
    {"requirement_key": "A.5.1", "code": "A.5.1", "label": "Policies", "status": "done",
     "reason": "Approved yearly.", "summary": "Approved yearly.", "next_step": None,
     "findings": [{"verdict": "met", "text": "Approved", "quote": "approved", "location": POINTER,
                   "suggested_place": None, "searched": None}], "searched": None,
     "evidence": "Policy | approved\nyearly", "source_citation": "policy.pdf", "recommendation": None,
     "target_collection_id": "t"},
    {"requirement_key": "A.8.2", "code": "A.8.2", "label": "Access", "status": "missing",
     "reason": "No access review.", "summary": "No access review.",
     "next_step": {"action": "upload_document", "explain": "Add an access review"},
     "findings": [{"verdict": "missing", "text": "Access review", "quote": None, "location": None,
                   "suggested_place": None, "searched": "Searched 1 document"}],
     "searched": {"folders": [], "document_count": 1},
     "evidence": None, "source_citation": None, "recommendation": "Add an access review", "target_collection_id": None},
    {"requirement_key": "A.9", "code": "A.9", "label": "Backup", "status": "incomplete",
     "reason": "Weekly only.", "summary": "Weekly only.",
     "next_step": {"action": "revise_document", "explain": "Daily backups"},
     "findings": [{"verdict": "partial", "text": "Weekly", "quote": "Weekly only", "location": POINTER,
                   "suggested_place": None, "searched": None}], "searched": None,
     "evidence": "Weekly only", "source_citation": None, "recommendation": "Daily backups", "target_collection_id": "t"},
    {"requirement_key": "A.7.1", "code": "A.7.1", "label": "Screening", "status": "not_checked",
     "reason": "Needs HR records.", "summary": "Needs HR records.", "next_step": None, "findings": [],
     "searched": None, "evidence": None, "source_citation": None, "recommendation": None, "target_collection_id": None},
]
REQUIREMENTS = [{"key": i["requirement_key"], "code": i["code"], "label": i["label"], "expected_document": None}
                for i in ITEMS]
SUMMARY = {"done": 1, "incomplete": 1, "missing": 1, "needs_decision": 0, "not_applicable": 0}
ZERO = {"done": 0, "incomplete": 0, "missing": 0, "needs_decision": 0, "not_applicable": 0}


def engine_result(items=ITEMS, tokens=1234):
    """What processor.run_compliance_gap_check returns; fresh copies, since
    the service adds each item's trend in place."""
    return ([dict(i) for i in items], "AI generated", tokens, REQUIREMENTS)


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
            "run": patch.object(processor, "run_compliance_gap_check", side_effect=lambda **_: engine_result()),
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
        self.assertEqual(body["summary"], SUMMARY)  # not_checked is not counted
        self.assertEqual((body["not_checked"], body["total_requirements"]), (1, 4))
        self.assertEqual((body["disclaimer"], len(body["items"])), ("AI generated", 4))
        self.assertEqual(body["items"][0]["findings"][0]["location"]["label"], "policy.pdf · p. 2")
        self.assertEqual([i["trend"] for i in body["items"]], [None] * 4)  # first run: nothing to compare
        run = body["run"]
        self.assertEqual((run["skill_id"], run["framework_name"], run["reference_collection_ids"], run["target_collection_ids"]),
                         ("compliance_gap_check", "ISO 27001", [self.reference], [self.target]))
        self.assertEqual((run["goal"], run["rule_set_version"], run["searched"]),
                         ("ISO 27001", 1, {"folders": [], "document_count": 1}))
        self.assertTrue(run["analysis_id"])
        self.m["run"].assert_called_once_with(
            reference_collection_ids=[self.reference], target_collection_ids=[self.target],
            framework_name="ISO 27001", requirements=None, directions="",
            searched={"folders": [], "document_count": 1})
        self.m["log"].assert_called_once_with(self.owner.user_id, 1234)
        reserve = config.gap_check_token_reserve
        self.m["plan"].assert_called_once_with(self.owner, 0, reserve)
        self.assertEqual(inflight.in_flight_tokens(f"w-{self.owner.user_id}", self.owner.user_id), (0, 0))

        stored = self.client(self.owner).get(f"/api/v1/analysis/gap-analysis/{run['run_id']}").json()
        self.assertEqual(([i["label"] for i in stored["items"]], stored["summary"], stored["disclaimer"]),
                         ([i["label"] for i in ITEMS], body["summary"], None))
        self.assertEqual(stored["items"], body["items"])
        self.assertEqual(stored["run"]["analysis_id"], run["analysis_id"])

    def test_a_rerun_of_an_analysis_reuses_its_requirements_and_records_trends(self):
        first = self.run_check(goal="Internal audit", directions="Ignore drafts").json()
        analysis_id = first["run"]["analysis_id"]
        self.assertEqual(first["run"]["goal"], "Internal audit")
        rerun_items = [dict(ITEMS[0], status="incomplete"), dict(ITEMS[1], status="done"), ITEMS[2], ITEMS[3]]
        self.m["run"].side_effect = lambda **_: engine_result(rerun_items, tokens=10)

        # The rule set wins over whatever reference the request names.
        second = self.run_check(analysis_id=analysis_id, reference_collection_ids=[self.target]).json()
        _, kwargs = self.m["run"].call_args
        self.assertEqual((kwargs["requirements"], kwargs["directions"], kwargs["reference_collection_ids"]),
                         (REQUIREMENTS, "Ignore drafts", [self.reference]))
        self.assertEqual([i["trend"] for i in second["items"]], ["worse", "improved", "unchanged", None])
        self.assertEqual((second["run"]["analysis_id"], second["run"]["previous_run_id"]),
                         (analysis_id, first["run"]["run_id"]))

        self.assertEqual(self.run_check(analysis_id="nope").status_code, 404)
        self.assertEqual(self.run_check(user(), analysis_id=analysis_id).status_code, 403)

    def test_a_requirement_the_ai_gave_no_answer_for_gets_no_trend_and_is_no_baseline(self):
        from services.rag.gap_locations import unanswered_item
        analysis_id = self.run_check().json()["run"]["analysis_id"]  # A.5.1 done
        unanswered = unanswered_item(REQUIREMENTS[0])
        self.m["run"].side_effect = lambda **_: engine_result([unanswered, *ITEMS[1:]])
        second = self.run_check(analysis_id=analysis_id).json()
        self.assertEqual((second["items"][0]["trend"], second["items"][0]["answered"]), (None, False))

        # The next run compares A.5.1 with the last run that answered it (done), not with "needs decision".
        self.m["run"].side_effect = lambda **_: engine_result([dict(ITEMS[0], status="incomplete"), *ITEMS[1:]])
        third = self.run_check(analysis_id=analysis_id).json()
        self.assertEqual(third["items"][0]["trend"], "worse")

    def test_checking_again_after_a_document_was_deleted_says_so(self):
        analysis_id = self.run_check().json()["run"]["analysis_id"]
        self.assertTrue(storage.delete_collection_from_db(self.target))
        resp = self.run_check(analysis_id=analysis_id)
        self.assertEqual(resp.status_code, 404)
        self.assertIn("has been deleted", resp.json()["detail"])
        # A new run still only says it can't reach the document.
        self.assertEqual(self.run_check().status_code, 403)

    def test_changed_directions_are_saved_as_a_new_rule_set_version(self):
        analysis_id = self.run_check().json()["run"]["analysis_id"]
        changed = self.run_check(analysis_id=analysis_id, directions="Ignore drafts").json()["run"]
        self.assertEqual(changed["rule_set_version"], 2)
        self.assertEqual(storage.get_analysis(analysis_id)["directions"], "Ignore drafts")
        unchanged = self.run_check(analysis_id=analysis_id).json()["run"]
        self.assertEqual(unchanged["rule_set_version"], 2)

    def test_a_requirement_list_first_read_on_a_later_run_is_kept(self):
        self.m["run"].side_effect = lambda **_: ([], "AI generated", 5, [])  # the reference could not be read
        analysis_id = self.run_check().json()["run"]["analysis_id"]
        self.m["run"].side_effect = lambda **_: engine_result()
        rerun = self.run_check(analysis_id=analysis_id).json()["run"]
        _, kwargs = self.m["run"].call_args
        self.assertIsNone(kwargs["requirements"])  # read from the reference again
        self.assertEqual(rerun["rule_set_version"], 2)
        self.assertEqual(storage.get_analysis(analysis_id)["requirements"], REQUIREMENTS)

    def test_where_the_ai_searched_names_only_the_users_own_folders(self):
        mine, theirs = str(uuid.uuid4()), str(uuid.uuid4())
        self.assertTrue(storage.create_folder(mine, "02 IT Infra", self.owner.user_id))
        self.assertTrue(storage.create_folder(theirs, "Secret", user().user_id))
        body = self.run_check(target_folder_ids=[mine, theirs]).json()
        self.assertEqual(body["run"]["searched"], {"folders": ["02 IT Infra"], "document_count": 1})
        self.assertEqual(body["run"]["target_folder_ids"], [mine, theirs])

    def test_where_the_ai_searched_counts_files_not_collections(self):
        bundle = str(uuid.uuid4())
        self.assertTrue(storage.register_collection(bundle, ["a.pdf", "b.docx", "c.xlsx"], 3, title="Bundle",
                                                    owner_id=self.owner.user_id))
        body = self.run_check(target_collection_ids=[self.target, bundle]).json()
        self.assertEqual(body["run"]["searched"]["document_count"], 4)

    def test_runs_stored_before_ms_1112_read_with_the_new_statuses(self):
        run_id = str(uuid.uuid4())
        storage.create_gap_analysis_run(run_id, "compliance_gap_check", [self.reference], "ISO 27001",
                                        [self.target], owner_id=self.owner.user_id)
        conn = db.get_conn("test")
        try:
            with conn.cursor() as cur:
                for label, status in (("A", "met"), ("B", "partial"), ("C", "not_met"), ("D", "unknown")):
                    cur.execute("INSERT INTO gap_analysis_items (run_id, label, status, evidence, source_citation,"
                                " recommendation) VALUES (%s, %s, %s, %s, %s, %s)",
                                (run_id, label, status, f"{label} evidence.", f"{label}.pdf p.1",
                                 None if status == "met" else f"Fix {label}."))
        finally:
            conn.close()
        body = self.client(self.owner).get(f"/api/v1/analysis/gap-analysis/{run_id}").json()
        self.assertEqual([i["status"] for i in body["items"]], ["done", "incomplete", "missing", "needs_decision"])
        # Their old evidence, citation and recommendation still show on the new results page.
        partial = body["items"][1]
        self.assertEqual(partial["findings"][0]["text"], "B evidence. (B.pdf p.1)")
        self.assertEqual((partial["reason"], partial["next_step"]["explain"]), ("B evidence.", "Fix B."))
        self.assertEqual(body["summary"], {**ZERO, "done": 1, "incomplete": 1, "missing": 1, "needs_decision": 1})
        self.assertIsNone(body["run"]["analysis_id"])

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


class RunsListTest(ComplianceTestCase):
    """MS-916: the history page reads each run's result counts and, for an
    admin, who ran it, straight from the list."""

    def register(self, who, created_by=None):
        conn = db.get_conn("test")
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (user_id, email, password_hash, role, created_by) VALUES (%s, %s, 'x', %s, %s)",
                    (who.user_id, who.email, who.role, created_by),
                )
        finally:
            conn.close()

    def runs(self, who):
        return {r["run_id"]: r for r in self.client(who).get("/api/v1/analysis/gap-analysis/runs").json()}

    def test_runs_carry_their_summary_and_admins_see_who_ran_them(self):
        admin = user("admin")
        self.register(admin)
        self.register(self.owner, created_by=admin.user_id)
        run_id = self.run_check().json()["run"]["run_id"]

        [own] = self.runs(self.owner).values()
        self.assertEqual((own["summary"], own["not_checked"]), (SUMMARY, 1))
        self.assertIsNone(own["run_by"])

        listed = self.runs(admin)
        self.assertEqual(listed[run_id]["run_by"], self.owner.email)
        self.assertEqual(listed[run_id]["summary"]["done"], 1)

    def test_an_admin_of_another_workspace_does_not_see_who_ran_it(self):
        self.register(self.owner, created_by=user("admin").user_id)
        run_id = self.run_check().json()["run"]["run_id"]
        listed = self.runs(user("admin"))
        self.assertIsNone(listed[run_id]["run_by"])
        self.assertEqual(listed[run_id]["summary"]["done"], 1)

    def test_runs_name_their_own_collections(self):
        """An admin's own collection list can't name another user's documents."""
        run_id = self.run_check().json()["run"]["run_id"]
        labels = {self.reference: "ISO 27001", self.target: "Our policies"}
        admin = user("admin")
        self.assertEqual(self.runs(admin)[run_id]["collection_labels"], labels)
        run = self.client(admin).get(f"/api/v1/analysis/gap-analysis/{run_id}").json()["run"]
        self.assertEqual(run["collection_labels"], labels)

    def test_a_run_without_items_counts_zero(self):
        self.m["run"].side_effect = lambda **_: engine_result([], tokens=10)
        self.run_check()
        [run] = self.client(self.owner).get("/api/v1/analysis/gap-analysis/runs").json()
        self.assertEqual((run["summary"], run["not_checked"]), (ZERO, 0))


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
        self.assertIn("| A.5.1 Policies | policy.pdf | comply | Policy / approved yearly |  |", lines)
        self.assertIn("| A.8.2 Access |  | not comply |  | Add an access review |", lines)
        self.assertIn("| A.9 Backup | t | partially comply | Weekly only | Daily backups |", lines)
        self.assertIn("| A.7.1 Screening |  | not checked |  |  |", lines)

        pdf = self.client(self.owner).get(url, params={"format": "pdf"})
        self.assertEqual((pdf.headers["content-type"], pdf.content[:5]), ("application/pdf", b"%PDF-"))
        self.assertEqual(self.client(self.owner).get(url, params={"format": "docx"}).status_code, 422)
        self.assertEqual(self.client(user()).get(url).status_code, 403)
        self.assertEqual(self.client(self.owner).get("/api/v1/analysis/gap-analysis/nope/export").status_code, 404)


if __name__ == "__main__":
    import unittest
    unittest.main()
