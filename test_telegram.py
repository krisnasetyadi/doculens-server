"""MS-657: behaviour of Telegram connections (router/ + services/telegram.py) against real
SQL, recorded before it is moved: the connect flow (code, 2FA), listing and
managing connections, listing dialogs and syncing chats into chat collections.
Telethon is replaced by a fake client, embeddings are faked and Supabase
Storage is off, so nothing leaves the machine. Runs in a throwaway schema
(testing_db); skipped when no database is reachable."""

import os
import tempfile
import unittest
import uuid
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from telethon.errors import PhoneCodeInvalidError, SessionPasswordNeededError

import storage
import telegram_crypto
from config import config
from models import UserRecord
from processor import processor
from router import telegram
from services import telegram as telegram_service
from services.auth import get_current_user
from testing_db import IsolatedSchemaTestCase

PHONE = "+6281234567890"


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.0]


# Entity classes are told apart by class name, as Telethon's are.
class User(SimpleNamespace):
    pass


class Chat(SimpleNamespace):
    pass


class Channel(SimpleNamespace):
    pass


class MessageMediaPhoto:
    pass


def message(msg_id, text, sender_id, minute, out=False, media=None):
    return SimpleNamespace(id=msg_id, text=text, message=text, sender_id=sender_id, out=out, media=media,
                           date=datetime(2026, 1, 1, 9, minute))


class FakeTelegram:
    """What the fake Telegram account holds, and every client made for it."""

    def __init__(self):
        self.clients = []
        self.send_code_error = None
        self.sign_in_error = None
        self.authorized = True
        self.me = User(id=1, first_name="Admin", last_name=None, username="admin")
        self.dialogs = []
        self.entities = {}
        self.participants = {}
        self.messages = {}

    def client_class(self):
        account = self

        class FakeClient:
            def __init__(self, session, api_id, api_hash):
                self.saved_session, self.api_id, self.api_hash = session, api_id, api_hash
                self.session = SimpleNamespace(save=lambda: "SESSION-STRING")
                self.disconnected = False
                account.clients.append(self)

            async def connect(self):
                return None

            def disconnect(self):
                self.disconnected = True

                async def done():
                    return None
                return done()

            async def send_code_request(self, phone):
                if account.send_code_error:
                    raise account.send_code_error
                return SimpleNamespace(phone_code_hash="HASH")

            async def sign_in(self, phone=None, code=None, phone_code_hash=None, password=None):
                if account.sign_in_error and not password:
                    raise account.sign_in_error
                self.signed_in = {"phone": phone, "code": code, "hash": phone_code_hash, "password": password}

            async def is_user_authorized(self):
                return account.authorized

            async def get_dialogs(self, limit):
                return account.dialogs

            async def get_entity(self, dialog_id):
                if dialog_id not in account.entities:
                    raise ValueError(f"Cannot find any entity corresponding to {dialog_id}")
                return account.entities[dialog_id]

            async def get_me(self):
                return account.me

            async def get_participants(self, entity, limit):
                if entity.id not in account.participants:
                    raise RuntimeError("cannot list participants of a channel")
                return account.participants[entity.id]

            async def iter_messages(self, entity, limit=None):
                for msg in account.messages.get(entity.id, []):
                    yield msg

        return FakeClient


class TelegramTestCase(IsolatedSchemaTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.account = FakeTelegram()
        self.stack.enter_context(patch("telethon.TelegramClient", self.account.client_class()))
        self.stack.enter_context(patch("telethon.sessions.StringSession", lambda session=None: session))
        self.stack.enter_context(patch.object(config, "telegram_session_encryption_key", Fernet.generate_key().decode()))
        telegram_crypto._fernet.cache_clear()
        self.addCleanup(telegram_crypto._fernet.cache_clear)
        self.stack.enter_context(patch.object(processor, "embeddings", FakeEmbeddings()))
        self.stack.enter_context(patch.object(storage, "is_enabled", lambda: False))
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for name in ("chat_upload_folder", "chat_index_folder"):
            (directory / name).mkdir()
            self.stack.enter_context(patch.object(config, name, str(directory / name)))
        self.chat_dir = directory / "chat_upload_folder"
        self.stack.enter_context(patch.object(telegram_service, "_pending_logins", {}))
        self.sql("DELETE FROM telegram_connections")
        self.role = "admin"

    @property
    def http(self):
        app = FastAPI()
        app.include_router(telegram.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: UserRecord(
            user_id="admin-1", email="admin@example.com", role=self.role, is_active=True)
        return TestClient(app)

    def start(self, **body):
        return self.http.post("/api/v1/telegram-connections/connect/start",
                              json={"api_id": 12345, "api_hash": "  app-hash  ", "phone": f" {PHONE} ", **body})

    def connect(self, label=None):
        """A finished connect flow; returns the connection as the API answered it."""
        flow_id = self.start(**({"label": label} if label else {})).json()["flow_id"]
        resp = self.http.post("/api/v1/telegram-connections/connect/verify", json={"flow_id": flow_id, "code": "11111"})
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["connection"]


class ConnectFlowTest(TelegramTestCase):
    def test_code_login_saves_an_encrypted_connection(self):
        started = self.start(label="  Work  ")
        self.assertEqual(started.json()["phone"], PHONE)
        flow_id = started.json()["flow_id"]
        resp = self.http.post("/api/v1/telegram-connections/connect/verify", json={"flow_id": flow_id, "code": "11111"})
        body = resp.json()
        self.assertEqual(body["status"], "connected")
        connection = body["connection"]
        self.assertEqual((connection["label"], connection["phone_masked"], connection["status"], connection["selected_chats"]),
                         ("Work", "+628*******890", "active", []))
        self.assertEqual(self.account.clients[0].signed_in, {"phone": PHONE, "code": "11111", "hash": "HASH", "password": None})
        self.assertTrue(self.account.clients[0].disconnected)

        row = self.sql("SELECT * FROM telegram_connections")[0]
        self.assertEqual((row["api_id"], row["phone"]), (12345, PHONE))
        self.assertEqual(telegram_crypto.decrypt_secret(row["api_hash_enc"]), "app-hash")
        self.assertEqual(telegram_crypto.decrypt_secret(row["session_enc"]), "SESSION-STRING")
        # the flow is used up
        again = self.http.post("/api/v1/telegram-connections/connect/verify", json={"flow_id": flow_id, "code": "11111"})
        self.assertEqual(again.status_code, 400)

    def test_two_step_verification_asks_for_the_password(self):
        self.account.sign_in_error = SessionPasswordNeededError(request=None)
        flow_id = self.start().json()["flow_id"]
        first = self.http.post("/api/v1/telegram-connections/connect/verify", json={"flow_id": flow_id, "code": "11111"})
        self.assertEqual(first.json(), {"status": "password_required", "connection": None})
        second = self.http.post("/api/v1/telegram-connections/connect/verify", json={"flow_id": flow_id, "code": "11111", "password": "pw"})  # chat-ui resends the code
        self.assertEqual(second.json()["status"], "connected")
        self.assertEqual(second.json()["connection"]["label"], "Telegram (+628*******890)")

    def test_bad_input_and_failures(self):
        self.assertEqual(self.start(phone="  ").status_code, 400)
        self.assertEqual(self.start(api_hash=" ").status_code, 400)
        self.account.send_code_error = RuntimeError("flood wait")
        resp = self.start()
        self.assertEqual((resp.status_code, resp.json()["detail"]), (400, "Could not send login code: flood wait"))
        self.assertTrue(self.account.clients[-1].disconnected)
        self.account.send_code_error = None

        self.account.sign_in_error = PhoneCodeInvalidError(request=None)
        flow_id = self.start().json()["flow_id"]
        resp = self.http.post("/api/v1/telegram-connections/connect/verify", json={"flow_id": flow_id, "code": "00000"})
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(resp.json()["detail"].startswith("Invalid or expired code"))
        self.assertEqual(self.http.post("/api/v1/telegram-connections/connect/verify",
                                        json={"flow_id": "nope", "code": "1"}).status_code, 400)
        self.assertEqual(self.sql("SELECT * FROM telegram_connections"), [])

    def test_disabled_without_an_encryption_key_and_admin_only(self):
        with patch.object(config, "telegram_session_encryption_key", None):
            self.assertEqual(self.start().status_code, 503)
        self.role = "user"
        self.assertEqual(self.start().status_code, 403)
        self.assertEqual(self.http.get("/api/v1/telegram-connections").status_code, 403)


class ManageTest(TelegramTestCase):
    def test_list_activate_and_delete(self):
        first = self.connect(label="One")["connection_id"]
        second = self.connect(label="Two")["connection_id"]
        listed = self.http.get("/api/v1/telegram-connections").json()
        self.assertEqual((listed["count"], [c["label"] for c in listed["connections"]]), (2, ["Two", "One"]))

        resp = self.http.post("/api/v1/telegram-connections/activate", json={"connection_id": first, "active": False})
        self.assertEqual(resp.json(), {"status": "success", "connection_id": first, "active": False})
        self.assertEqual(self.sql("SELECT status FROM telegram_connections WHERE connection_id = %s", (first,))[0]["status"], "inactive")
        self.assertEqual(self.http.post("/api/v1/telegram-connections/activate",
                                        json={"connection_id": "nope", "active": True}).status_code, 404)

        self.assertEqual(self.http.delete(f"/api/v1/telegram-connections/{second}").json(),
                         {"status": "success", "message": "Telegram connection deleted", "connection_id": second})
        self.assertEqual(self.http.delete(f"/api/v1/telegram-connections/{second}").status_code, 404)
        self.assertEqual([c["connection_id"] for c in self.http.get("/api/v1/telegram-connections").json()["connections"]], [first])


class DialogsTest(TelegramTestCase):
    def test_dialog_types(self):
        connection_id = self.connect()["connection_id"]
        self.account.dialogs = [
            SimpleNamespace(id=10, name="Alice", entity=User(id=10)),
            SimpleNamespace(id=-20, name="Team", entity=Chat(id=20, participants_count=4)),
            SimpleNamespace(id=-30, name="", entity=Channel(id=30, megagroup=True)),
            SimpleNamespace(id=-40, name="News", entity=Channel(id=40, broadcast=True)),
        ]
        body = self.http.get(f"/api/v1/telegram-connections/{connection_id}/dialogs").json()
        self.assertEqual(body["count"], 4)
        self.assertEqual([(d["dialog_id"], d["title"], d["type"], d["participants_count"]) for d in body["dialogs"]],
                         [("10", "Alice", "user", None), ("-20", "Team", "group", 4),
                          ("-30", "-30", "group", None), ("-40", "News", "channel", None)])
        self.assertTrue(self.account.clients[-1].disconnected)
        self.assertEqual(self.account.clients[-1].saved_session, "SESSION-STRING")

    def test_unknown_expired_and_undecryptable_connections(self):
        self.assertEqual(self.http.get("/api/v1/telegram-connections/nope/dialogs").status_code, 404)
        connection_id = self.connect()["connection_id"]
        self.account.authorized = False
        self.assertEqual(self.http.get(f"/api/v1/telegram-connections/{connection_id}/dialogs").status_code, 401)
        self.sql("UPDATE telegram_connections SET session_enc = 'not-a-token'")
        self.assertEqual(self.http.get(f"/api/v1/telegram-connections/{connection_id}/dialogs").status_code, 500)


class SyncTest(TelegramTestCase):
    def setUp(self):
        super().setUp()
        self.connection_id = self.connect()["connection_id"]
        group = Chat(id=-20, title="Team: Ops/Room")
        self.account.entities[-20] = group
        self.account.participants[-20] = [User(id=2, first_name="Budi", last_name="S", username=None),
                                         User(id=3, first_name=None, last_name=None, username="citra")]
        self.account.messages[-20] = [  # newest first, as Telegram returns them
            message(5, None, 3, 4, media=MessageMediaPhoto()),
            message(4, None, 2, 3),  # service event: no text, no media
            message(3, "anyone?", 1, 2, out=True),
            message(2, "line one\nline two", 9, 1),  # sender not in the participant list
            message(1, "Hello team", 2, 0),
        ]

    def sync(self, *dialog_ids):
        return self.http.post(f"/api/v1/telegram-connections/{self.connection_id}/sync", json={"dialog_ids": list(dialog_ids)})

    def test_a_chat_becomes_a_collection_with_a_transcript(self):
        result = self.sync("-20").json()["results"][0]
        self.assertEqual((result["status"], result["title"], result["message_count"]), ("success", "Team: Ops/Room", 4))
        collection_id = result["chat_collection_id"]
        transcript = (self.chat_dir / collection_id / "telegram-Team_ Ops_Room.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(transcript, [
            "2026-01-01 09:00:00 | Budi S: Hello team",
            "2026-01-01 09:01:00 | 9: line one\\nline two",
            "2026-01-01 09:02:00 | Admin: anyone?",
            "2026-01-01 09:04:00 | citra: [Media: Photo]",
        ])
        self.assertEqual([p.name for p in self.chat_dir.iterdir()], [collection_id])  # no staged file left
        chat = self.sql("SELECT * FROM telegram_selected_chats WHERE connection_id = %s", (self.connection_id,))[0]
        self.assertEqual((chat["dialog_id"], chat["dialog_type"], chat["chat_collection_id"], chat["message_count"]),
                         ("-20", "group", collection_id, 4))
        self.assertIsNotNone(self.sql("SELECT last_synced_at FROM telegram_connections")[0]["last_synced_at"])
        listed = self.http.get("/api/v1/telegram-connections").json()["connections"][0]["selected_chats"]
        self.assertEqual([(c["dialog_id"], c["chat_collection_id"], c["status"]) for c in listed], [("-20", collection_id, "active")])

    def test_resync_reuses_the_collection_and_replaces_the_transcript(self):
        collection_id = self.sync("-20").json()["results"][0]["chat_collection_id"]
        processor.vector_store_cache[f"chat_{collection_id}"] = "stale"
        self.account.entities[-20].title = "Renamed"
        result = self.sync("-20").json()["results"][0]
        self.assertEqual(result["chat_collection_id"], collection_id)
        self.assertEqual([p.name for p in (self.chat_dir / collection_id).iterdir()], ["telegram-Renamed.txt"])
        self.assertNotIn(f"chat_{collection_id}", processor.vector_store_cache)
        self.assertEqual(len(self.sql("SELECT * FROM telegram_selected_chats")), 1)

    def test_failed_and_empty_chats_are_reported_per_chat(self):
        self.account.entities[10] = User(id=10, first_name="Dewi", last_name=None, username=None)
        body = self.sync("-20", "10", "999", "abc").json()["results"]
        self.assertEqual([(r["dialog_id"], r["status"]) for r in body],
                         [("-20", "success"), ("10", "error"), ("999", "error"), ("abc", "error")])
        self.assertEqual((body[1]["title"], body[1]["error"]), ("Dewi", "No messages with text or media found in this chat"))
        self.assertIn("Cannot find any entity", body[2]["error"])
        self.assertEqual(body[3]["title"], "abc")

    def test_bad_requests(self):
        self.assertEqual(self.sync().status_code, 400)
        self.assertEqual(self.http.post("/api/v1/telegram-connections/nope/sync", json={"dialog_ids": ["1"]}).status_code, 404)


class HelperTest(unittest.TestCase):
    """Pure helpers; no database needed."""

    def test_phone_masking(self):
        self.assertEqual(telegram_service._mask_phone("12345"), "1****")
        self.assertEqual(telegram_service._mask_phone("1234567"), "1234*567")
        self.assertEqual(telegram_service._mask_phone(PHONE), "+628*******890")

    def test_transcript_names(self):
        self.assertEqual(telegram_service._telegram_transcript_name("  ...  ", 42), "telegram-42.txt")
        self.assertEqual(len(telegram_service._telegram_transcript_name("x" * 200, 1)), len("telegram-.txt") + 80)


if __name__ == "__main__":
    unittest.main()
