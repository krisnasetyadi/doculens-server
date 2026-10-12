"""MS-657: behaviour of skills (router/skills.py) against real SQL, recorded
before it is moved: personal skills for their uploader, team skills for an
admin's members, and owner-only edits and deletes. Runs in a throwaway schema
(testing_db); skipped when no database is reachable."""

import os
import uuid

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from models import UserRecord
from router import skills
from services.auth import get_current_user
from testing_db import IsolatedSchemaTestCase


class SkillsTestCase(IsolatedSchemaTestCase):
    def account(self, role="user", created_by=None) -> UserRecord:
        """A users row (team skills are found through users.created_by)."""
        uid = f"{role}-{uuid.uuid4().hex[:8]}"
        self.sql("INSERT INTO users (user_id, email, password_hash, role, created_by) VALUES (%s, %s, 'x', %s, %s)",
                 (uid, f"{uid}@example.com", role, created_by))
        return UserRecord(user_id=uid, email=f"{uid}@example.com", role=role, is_active=True)

    def client(self, who):
        app = FastAPI()
        app.include_router(skills.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: who
        return TestClient(app)

    def create(self, who, **body):
        payload = {"name": "Summarise", "slash_command": "summarise", "instruction": "Answer in 3 bullets", **body}
        return self.client(who).post("/api/v1/skills", json=payload)


class SkillsTest(SkillsTestCase):
    def test_personal_and_team_visibility(self):
        admin = self.account("admin")
        member = self.account(created_by=admin.user_id)
        outsider = self.account()
        team = self.create(admin, name="Team skill", scope="team", description="for everyone")
        self.assertEqual(team.status_code, 201, team.text)
        self.assertEqual((team.json()["scope"], team.json()["owner_id"], team.json()["description"]),
                         ("team", admin.user_id, "for everyone"))
        mine = self.create(member, name="Mine").json()["skill_id"]

        names = lambda who: [s["name"] for s in self.client(who).get("/api/v1/skills").json()]
        self.assertEqual(sorted(names(member)), ["Mine", "Team skill"])
        self.assertEqual(names(admin), ["Team skill"])  # a member's personal skill stays theirs
        self.assertEqual(names(outsider), [])
        self.assertEqual(self.client(member).get(f"/api/v1/skills/{team.json()['skill_id']}").status_code, 200)
        self.assertEqual(self.client(outsider).get(f"/api/v1/skills/{mine}").status_code, 404)

    def test_only_admins_share_with_the_team(self):
        member = self.account()
        self.assertEqual(self.create(member, scope="team").status_code, 403)
        mine = self.create(member).json()["skill_id"]
        self.assertEqual(self.client(member).put(f"/api/v1/skills/{mine}", json={"scope": "team"}).status_code, 403)
        self.assertEqual(self.create(member, scope="everyone").status_code, 422)

    def test_owner_edits_and_deletes(self):
        admin = self.account("admin")
        member = self.account(created_by=admin.user_id)
        team_id = self.create(admin, scope="team").json()["skill_id"]
        self.assertEqual(self.client(member).put(f"/api/v1/skills/{team_id}", json={"name": "x"}).status_code, 403)
        self.assertEqual(self.client(member).delete(f"/api/v1/skills/{team_id}").status_code, 403)

        unchanged = self.client(admin).put(f"/api/v1/skills/{team_id}", json={})
        self.assertEqual(unchanged.json()["name"], "Summarise")
        updated = self.client(admin).put(f"/api/v1/skills/{team_id}", json={"name": "Brief", "instruction": "One line"}).json()
        self.assertEqual((updated["name"], updated["instruction"], updated["slash_command"]), ("Brief", "One line", "/summarise"))  # stored with its slash
        self.assertEqual(self.client(admin).put("/api/v1/skills/nope", json={"name": "x"}).status_code, 404)

        self.assertEqual(self.client(admin).delete(f"/api/v1/skills/{team_id}").json(), {"skill_id": team_id, "deleted": True})
        self.assertEqual(self.client(admin).get(f"/api/v1/skills/{team_id}").status_code, 404)
        self.assertEqual(self.client(admin).delete(f"/api/v1/skills/{team_id}").status_code, 404)


if __name__ == "__main__":
    import unittest
    unittest.main()
