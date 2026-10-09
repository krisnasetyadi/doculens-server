"""MS-657: behaviour of chat session storage (router/sessions.py) against real
SQL, recorded before it is moved. Runs in a throwaway schema (testing_db);
skipped when no database is reachable."""

import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from router import sessions
from router.auth import UserRecord, get_current_user
from testing_db import IsolatedSchemaTestCase

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def user(role="user") -> UserRecord:
    uid = f"{role}-{uuid.uuid4().hex[:8]}"
    return UserRecord(user_id=uid, email=f"{uid}@example.com", role=role, is_active=True)


def message(role, content, minute):
    return {"id": f"m-{uuid.uuid4().hex[:10]}", "role": role, "content": content,
            "created_at": (T0 + timedelta(minutes=minute)).isoformat()}


def chats(n, answers_per_question=1, start=0):
    """n question/answer exchanges, one minute apart."""
    out, minute = [], start
    for i in range(1, n + 1):
        out.append(message("user", f"question {i}", minute))
        minute += 1
        for a in range(answers_per_question):
            out.append(message("assistant", f"answer {i}.{a}", minute))
            minute += 1
    return out


class SessionsTestCase(IsolatedSchemaTestCase):
    def client(self, who: UserRecord) -> TestClient:
        app = FastAPI()
        app.include_router(sessions.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: who
        return TestClient(app)

    def create(self, who, messages, **fields):
        resp = self.client(who).post("/api/v1/sessions", json={"messages": messages, **fields})
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()


class UpsertTest(SessionsTestCase):
    def test_new_session_gets_a_default_title_and_counts_questions(self):
        owner = user()
        body = self.create(owner, chats(2), pdf_collections=["c1"])
        self.assertEqual(body["title"], "Untitled conversation")
        self.assertEqual(body["pdf_collections"], ["c1"])
        self.assertEqual(len(body["messages"]), 4)
        self.assertEqual(body["total_user_turns"], 2)

    def test_resending_updates_messages_by_id_and_keeps_the_title(self):
        owner = user()
        first = self.create(owner, chats(1), title="Budget")
        edited = dict(first["messages"][1], content="answer edited")
        again = self.create(owner, [first["messages"][0], edited], session_id=first["session_id"])
        self.assertEqual(again["title"], "Budget")
        self.assertEqual([m["content"] for m in again["messages"]], ["question 1", "answer edited"])

    def test_another_user_cannot_modify_a_session(self):
        owner, stranger = user(), user()
        created = self.create(owner, chats(1), title="Mine")
        resp = self.client(stranger).post("/api/v1/sessions", json={
            "session_id": created["session_id"], "title": "Taken", "messages": []})
        self.assertNotEqual(resp.status_code, 200)
        reloaded = self.client(owner).get(f"/api/v1/sessions/{created['session_id']}").json()
        self.assertEqual(reloaded["title"], "Mine")


class ListAndSearchTest(SessionsTestCase):
    def test_users_see_their_own_sessions_and_admins_see_all(self):
        alice, bob, admin = user(), user(), user("admin")
        a = self.create(alice, chats(2))["session_id"]
        b = self.create(bob, chats(1))["session_id"]
        alice_view = {s["session_id"]: s for s in self.client(alice).get("/api/v1/sessions").json()}
        self.assertIn(a, alice_view)
        self.assertNotIn(b, alice_view)
        self.assertEqual(alice_view[a]["message_count"], 4)
        admin_view = {s["session_id"] for s in self.client(admin).get("/api/v1/sessions").json()}
        self.assertTrue({a, b} <= admin_view)

    def test_title_match_has_no_snippet_and_message_match_has_one(self):
        owner = user()
        by_title = self.create(owner, chats(1), title="Quarterly zebra review")["session_id"]
        long_answer = "Intro text. " * 20 + "The **zebra** clause applies here. " + "More words. " * 20
        msgs = [message("user", "what applies?", 0), message("assistant", long_answer, 1)]
        by_message = self.create(owner, msgs, title="Other")["session_id"]

        found = {s["session_id"]: s for s in self.client(owner).get("/api/v1/sessions", params={"q": "zebra"}).json()}
        self.assertEqual(set(found), {by_title, by_message})
        self.assertIsNone(found[by_title]["matched_snippet"])
        snippet = found[by_message]["matched_snippet"]
        self.assertIn("zebra clause", snippet)        # markdown stripped
        self.assertTrue(snippet.startswith("…") and snippet.endswith("…"))
        self.assertEqual(found[by_message]["matched_message_id"], msgs[1]["id"])

    def test_search_treats_wildcards_literally(self):
        owner = user()
        self.create(owner, [message("user", "growth was 100% this year", 0)], title="Percent")
        self.create(owner, [message("user", "nothing special", 0)], title="Plain")
        titles = {s["title"] for s in self.client(owner).get("/api/v1/sessions", params={"q": "100%"}).json()}
        self.assertEqual(titles, {"Percent"})
        self.assertEqual(self.client(owner).get("/api/v1/sessions", params={"q": "%"}).json()[0]["title"], "Percent")


class PaginationTest(SessionsTestCase):
    def test_pages_by_chat_not_by_message(self):
        owner = user()
        sid = self.create(owner, chats(7, answers_per_question=2))["session_id"]
        client = self.client(owner)

        page1 = client.get(f"/api/v1/sessions/{sid}", params={"limit": 5}).json()
        self.assertEqual([m["content"] for m in page1["messages"] if m["role"] == "user"],
                         [f"question {i}" for i in range(3, 8)])
        self.assertEqual(len(page1["messages"]), 15)  # each question travels with both answers
        self.assertTrue(page1["has_more"])
        self.assertEqual(page1["total_user_turns"], 7)

        page2 = client.get(f"/api/v1/sessions/{sid}", params={"limit": 5, "before": page1["next_cursor"]}).json()
        self.assertEqual([m["content"] for m in page2["messages"] if m["role"] == "user"],
                         ["question 1", "question 2"])
        self.assertFalse(page2["has_more"])
        self.assertIsNone(page2["next_cursor"])

    def test_unknown_and_foreign_sessions(self):
        owner, stranger, admin = user(), user(), user("admin")
        sid = self.create(owner, chats(1))["session_id"]
        self.assertEqual(self.client(owner).get("/api/v1/sessions/nope").status_code, 404)
        self.assertEqual(self.client(stranger).get(f"/api/v1/sessions/{sid}").status_code, 403)
        self.assertEqual(self.client(admin).get(f"/api/v1/sessions/{sid}").status_code, 200)

    def test_question_index(self):
        owner = user()
        msgs = chats(3)
        msgs[0]["content"] = "x" * 300
        sid = self.create(owner, msgs)["session_id"]
        body = self.client(owner).get(f"/api/v1/sessions/{sid}/questions").json()
        self.assertEqual(body["total"], 3)
        self.assertEqual([q["turn"] for q in body["questions"]], [1, 2, 3])
        self.assertEqual(len(body["questions"][0]["preview"]), 120)
        self.assertEqual(self.client(user()).get(f"/api/v1/sessions/{sid}/questions").status_code, 403)


class RenameDeleteTest(SessionsTestCase):
    def test_rename(self):
        owner = user()
        sid = self.create(owner, chats(2))["session_id"]
        client = self.client(owner)
        self.assertEqual(client.put(f"/api/v1/sessions/{sid}", json={"title": "  "}).status_code, 400)
        renamed = client.put(f"/api/v1/sessions/{sid}", json={"title": "  New name "}).json()
        self.assertEqual((renamed["title"], renamed["message_count"]), ("New name", 4))
        self.assertEqual(client.put("/api/v1/sessions/nope", json={"title": "x"}).status_code, 404)
        self.assertEqual(self.client(user()).put(f"/api/v1/sessions/{sid}", json={"title": "x"}).status_code, 403)

    def test_delete_removes_messages_and_refuses_strangers(self):
        owner, stranger = user(), user()
        sid = self.create(owner, chats(2))["session_id"]
        self.assertNotEqual(self.client(stranger).delete(f"/api/v1/sessions/{sid}").status_code, 200)
        self.assertEqual(self.client(owner).get(f"/api/v1/sessions/{sid}").status_code, 200)
        self.assertEqual(self.client(owner).delete(f"/api/v1/sessions/{sid}").json(),
                         {"status": "deleted", "session_id": sid})
        self.assertEqual(self.sql("SELECT * FROM chat_messages WHERE session_id = %s", (sid,)), [])


class SnippetTest(unittest.TestCase):
    """Pure text helpers; no database needed."""

    def test_markdown_is_stripped(self):
        self.assertEqual(sessions._plain_text("## Title\n| a | b |\n| --- | --- |\n**bold** and `code` [link](http://x)"),
                         "Title a b bold and code link")

    def test_snake_case_survives(self):
        self.assertEqual(sessions._plain_text("use my_var_name here"), "use my_var_name here")

    def test_short_text_is_returned_whole(self):
        self.assertEqual(sessions._snippet_around("short answer", "answer"), "short answer")

    def test_unmatched_query_falls_back_to_the_start(self):
        text = "word " * 60
        snippet = sessions._snippet_around(text, "##")
        self.assertTrue(snippet.startswith("word") and snippet.endswith("…"))


if __name__ == "__main__":
    unittest.main()
