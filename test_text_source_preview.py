import hashlib
import tempfile
import unittest
from pathlib import Path

from text_source_preview import read_text_source_page


class TextSourcePreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "source.txt"

    def preview(self, content, platform, offset=0, limit=50):
        self.path.write_text(content, encoding="utf-8")
        before = hashlib.sha256(self.path.read_bytes()).digest()
        result = read_text_source_page(str(self.path), platform, offset, limit)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).digest(), before)
        return result

    def test_recognized_whatsapp_has_sender_and_timestamp(self):
        page = self.preview("[2025-12-18 08:30:15] Sarah: Hello\n", "whatsapp")
        self.assertEqual(page["subtype"], "whatsapp")
        self.assertEqual(page["messages"][0]["sender"], "Sarah")
        self.assertEqual(page["messages"][0]["timestamp"], "2025-12-18T08:30:15")

    def test_missing_metadata_detects_whatsapp(self):
        page = self.preview("[2025-12-18 08:30:15] Sarah: Hello\n", None)
        self.assertEqual(page["subtype"], "whatsapp")

    def test_other_subtypes_never_use_whatsapp_formatting(self):
        content = "[2025-12-18 08:30:15] Sarah: Hello\n"
        for platform in ("slack", "teams", "generic"):
            with self.subTest(platform=platform):
                page = self.preview(content, platform)
                self.assertEqual(page["subtype"], "plain_text")
                self.assertEqual(page["messages"], [])
                self.assertEqual(page["lines"][0]["content"], content.rstrip("\n"))

    def test_unrecognized_text_preserves_indentation_and_blank_lines(self):
        for platform in (None, "whatsapp"):
            with self.subTest(platform=platform):
                page = self.preview("\n  Meeting notes\n\nNext steps\n", platform)
                self.assertEqual(page["subtype"], "plain_text")
                self.assertEqual([line["content"] for line in page["lines"]],
                                 ["", "  Meeting notes", "", "Next steps"])

    def test_plain_text_pagination_reaches_last_line(self):
        content = "\n".join(f"Line {i}" for i in range(90))
        first = self.preview(content, "generic")
        last = read_text_source_page(str(self.path), "generic", 50, 50)
        self.assertTrue(first["has_more"])
        self.assertFalse(last["has_more"])
        self.assertEqual(first["total"], 90)
        self.assertEqual([line["content"] for line in first["lines"] + last["lines"]],
                         content.splitlines())
        self.assertEqual(last["lines"][-1]["line_number"], 90)

    def test_whatsapp_pagination_reaches_last_message(self):
        content = "\n".join(f"[2025-12-18 08:30:15] Sarah: Message {i}" for i in range(90))
        first = self.preview(content, "whatsapp")
        last = read_text_source_page(str(self.path), "whatsapp", 50, 50)
        self.assertTrue(first["has_more"])
        self.assertFalse(last["has_more"])
        self.assertEqual(len(first["messages"] + last["messages"]), 90)
        self.assertEqual(last["messages"][-1]["content"], "Message 89")

    def test_empty_content_returns_empty_plain_text(self):
        page = self.preview(" \n\n", None)
        self.assertEqual(page["subtype"], "plain_text")
        self.assertEqual(page["total"], 0)
        self.assertEqual(page["lines"], [])
        self.assertFalse(page["has_more"])


if __name__ == "__main__":
    unittest.main()
