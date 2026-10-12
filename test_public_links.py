"""MS-657: behaviour of public link sources (router/ + services/public_links.py) against
real SQL, recorded before it is moved. DNS and Google Drive are faked at the
socket/httpx level, so nothing leaves the machine. Runs in a throwaway schema
(testing_db); skipped when no database is reachable."""

import asyncio
import os
import socket
import unittest
import uuid
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms657-local-test-secret-not-for-production")

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from models import UserRecord
from router import public_links
from services import public_links as link_service
from services.auth import get_current_user
from testing_db import IsolatedSchemaTestCase

FOLDER_URL = "https://drive.google.com/drive/folders/ROOTFOLDER"
FOLDER_HTML = """
<a href="https://drive.google.com/file/d/FILE1/view">report.pdf</a>
<a href="https://drive.google.com/file/d/FILE1/view">report.pdf (duplicate)</a>
<a href="/drive/folders/SUBFOLDER">Archive</a>
<a href="https://example.com/elsewhere">not drive</a>
"""
SUBFOLDER_HTML = '<a href="https://drive.google.com/file/d/FILE2/view">notes.txt</a>'

_real_getaddrinfo = socket.getaddrinfo


def fake_getaddrinfo(host, *args, **kwargs):
    """Hostnames resolve to a public address without DNS; literal IPs resolve as themselves."""
    if host.replace(".", "").isdigit():
        return _real_getaddrinfo(host, *args, **kwargs)
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("142.250.4.100", 0))]


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


def drive_pages(pages):
    async def fake_get(self, url, **kwargs):
        for folder_id, html in pages.items():
            if f"id={folder_id}" in url:
                return FakeResponse(html)
        raise httpx.ConnectError("unexpected fetch: " + url)
    return fake_get


def user(role="user") -> UserRecord:
    uid = f"{role}-{uuid.uuid4().hex[:8]}"
    return UserRecord(user_id=uid, email=f"{uid}@example.com", role=role, is_active=True)


class PublicLinksTestCase(IsolatedSchemaTestCase):
    def setUp(self):
        for p in (patch("socket.getaddrinfo", fake_getaddrinfo),
                  patch("httpx.AsyncClient.get", drive_pages({"ROOTFOLDER": FOLDER_HTML, "SUBFOLDER": SUBFOLDER_HTML}))):
            p.start()
            self.addCleanup(p.stop)

    def client(self, who):
        app = FastAPI()
        app.include_router(public_links.router, prefix="/api/v1")
        app.dependency_overrides[get_current_user] = lambda: who
        return TestClient(app)

    def create(self, who, **body):
        return self.client(who).post("/api/v1/public-links", json=body)


class CreateTest(PublicLinksTestCase):
    def test_explicit_items_are_stored_deduplicated(self):
        resp = self.create(user(), url="https://example.com/docs/", title="  Docs ",
                           item_urls=["https://example.com/a.pdf", "https://example.com/a.pdf", "https://example.com/b.pdf"])
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        self.assertEqual((body["title"], body["status"], body["item_count"]), ("Docs", "active", 2))
        self.assertEqual([i["name"] for i in body["items"]], ["a.pdf", "b.pdf"])

    def test_drive_folder_lists_its_pdfs_and_subfolders(self):
        body = self.create(user(), url=FOLDER_URL).json()
        self.assertEqual(body["title"], "drive.google.com/ROOTFOLDER")
        self.assertEqual([(i["name"], i["item_type"]) for i in body["items"]],
                         [("report.pdf", "file"), ("Archive", "folder")])

    def test_single_drive_file_and_plain_page(self):
        file_body = self.create(user(), url="https://drive.google.com/file/d/ABC/view").json()
        self.assertEqual([i["item_type"] for i in file_body["items"]], ["file"])
        plain = self.create(user(), url="https://example.com/").json()
        self.assertEqual((plain["title"], plain["item_count"]), ("example.com", 0))

    def test_invalid_and_private_urls_are_refused(self):
        who = user()
        self.assertEqual(self.create(who, url="ftp://example.com/x").status_code, 400)
        self.assertEqual(self.create(who, url="http://127.0.0.1/admin").status_code, 400)
        self.assertEqual(self.sql("SELECT * FROM public_links WHERE user_id = %s", (who.user_id,)), [])

    def test_private_item_url_is_refused_and_nothing_is_stored(self):
        who = user()
        resp = self.create(who, url="https://example.com/",
                           item_urls=["https://example.com/a.pdf", "http://127.0.0.1/secret"])
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(self.sql("SELECT * FROM public_links WHERE user_id = %s", (who.user_id,)), [])
        self.assertEqual(self.sql("SELECT * FROM public_link_items WHERE url = %s", ("http://127.0.0.1/secret",)), [])


class ManageTest(PublicLinksTestCase):
    def test_users_see_their_own_links_and_admins_see_all(self):
        alice, bob, admin = user(), user(), user("admin")
        a = self.create(alice, url="https://example.com/a").json()["link_id"]
        b = self.create(bob, url="https://example.com/b").json()["link_id"]
        alice_view = self.client(alice).get("/api/v1/public-links").json()
        self.assertEqual(([l["link_id"] for l in alice_view["links"]], alice_view["count"]), ([a], 1))
        admin_ids = {l["link_id"] for l in self.client(admin).get("/api/v1/public-links").json()["links"]}
        self.assertTrue({a, b} <= admin_ids)

    def test_activate_own_link_only(self):
        owner, stranger, admin = user(), user(), user("admin")
        link_id = self.create(owner, url="https://example.com/x").json()["link_id"]
        self.assertEqual(self.client(owner).post("/api/v1/public-links/activate", json={"link_id": link_id, "active": False}).json(),
                         {"status": "success", "link_id": link_id, "active": False})
        self.assertEqual(self.client(stranger).post("/api/v1/public-links/activate",
                                                    json={"link_id": link_id, "active": True}).status_code, 404)
        self.assertEqual(self.client(admin).post("/api/v1/public-links/activate",
                                                 json={"link_id": link_id, "active": True}).status_code, 200)
        self.assertEqual(self.sql("SELECT status FROM public_links WHERE link_id = %s", (link_id,))[0]["status"], "active")

    def test_delete_own_link_with_its_items(self):
        owner, stranger = user(), user()
        link_id = self.create(owner, url="https://example.com/", item_urls=["https://example.com/a.pdf"]).json()["link_id"]
        self.assertEqual(self.client(stranger).delete(f"/api/v1/public-links/{link_id}").status_code, 404)
        self.assertEqual(self.client(owner).delete(f"/api/v1/public-links/{link_id}").json(),
                         {"status": "success", "message": "Public link deleted", "link_id": link_id})
        self.assertEqual(self.sql("SELECT * FROM public_link_items WHERE link_id = %s", (link_id,)), [])


class RuntimeResolutionTest(PublicLinksTestCase):
    def resolve(self, **kwargs):
        return asyncio.run(link_service.resolve_active_public_link_sources(**kwargs))

    def test_only_active_selected_links_and_folders_expand_recursively(self):
        who = user()
        folder = self.create(who, url=FOLDER_URL).json()["link_id"]
        other = self.create(who, url="https://example.com/other.pdf").json()["link_id"]
        inactive = self.create(who, url="https://example.com/off.pdf").json()["link_id"]
        self.client(who).post("/api/v1/public-links/activate", json={"link_id": inactive, "active": False})

        everything = {l["link_id"]: l for l in self.resolve(user_id=who.user_id)}
        self.assertEqual(set(everything), {folder, other})
        # report.pdf from the folder, notes.txt from its subfolder; the folder entry itself is expanded away
        self.assertEqual(sorted(i["name"] for i in everything[folder]["items"]), ["notes.txt", "report.pdf"])
        self.assertEqual(everything[other]["item_count"], 1)

        only = self.resolve(link_ids=[other], user_id=who.user_id)
        self.assertEqual([l["link_id"] for l in only], [other])

    def test_unreachable_folder_falls_back_to_stored_items(self):
        who = user()
        link_id = self.create(who, url=FOLDER_URL).json()["link_id"]
        with patch("httpx.AsyncClient.get", drive_pages({})):
            resolved = self.resolve(link_ids=[link_id], user_id=who.user_id)
        self.assertEqual(sorted(i["name"] for i in resolved[0]["items"]), ["Archive", "report.pdf"])


class DriveParsingTest(unittest.TestCase):
    """Pure URL/HTML helpers; no database needed."""

    def test_drive_ids(self):
        self.assertEqual(link_service._extract_google_drive_file_id("https://www.drive.google.com/file/d/X1_y-2/view"), "X1_y-2")
        self.assertEqual(link_service._extract_google_drive_file_id("https://drive.google.com/open?id=Q9"), "Q9")
        self.assertIsNone(link_service._extract_google_drive_file_id("https://example.com/file/d/X/view"))
        self.assertEqual(link_service._extract_google_drive_folder_id(FOLDER_URL), "ROOTFOLDER")

    def test_folder_page_parsing(self):
        files, folders = link_service._parse_drive_folder_items(FOLDER_HTML)
        self.assertEqual([f["name"] for f in files], ["report.pdf"])
        self.assertEqual([f["url"] for f in folders], ["https://drive.google.com/drive/folders/SUBFOLDER"])


if __name__ == "__main__":
    unittest.main()
