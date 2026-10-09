"""Tests for MS-657: scripts/sync_hf.py decides which files the HF Space repo must mirror."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

import sync_hf


def write(root, rel, data: bytes):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path) or root, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.source = tempfile.mkdtemp()
        self.target = tempfile.mkdtemp()

    def plan(self, source_files, target_files):
        return sync_hf.plan(self.source, self.target, set(source_files), set(target_files))

    def test_identical_repos_need_nothing(self):
        for root in (self.source, self.target):
            write(root, "main.py", b"x = 1\n")
        self.assertEqual(self.plan(["main.py"], ["main.py"]), ([], []))

    def test_line_endings_alone_are_not_a_difference(self):
        write(self.source, "main.py", b"x = 1\r\ny = 2\r\n")
        write(self.target, "main.py", b"x = 1\ny = 2\n")
        self.assertEqual(self.plan(["main.py"], ["main.py"]), ([], []))

    def test_changed_and_new_modules_are_copied(self):
        write(self.source, "main.py", b"x = 2\n")
        write(self.source, "schema/new.py", b"")
        write(self.target, "main.py", b"x = 1\n")
        to_copy, _ = self.plan(["main.py", "schema/new.py"], ["main.py"])
        self.assertEqual(to_copy, ["main.py", "schema/new.py"])

    def test_source_only_files_are_not_mirrored(self):
        for rel in ("notes.ipynb", "AGENTS.md", "data/chat.txt", "perf/timings/before.json", ".gitignore"):
            write(self.source, rel, b"local")
        to_copy, _ = self.plan(["notes.ipynb", "AGENTS.md", "data/chat.txt", "perf/timings/before.json", ".gitignore"], [])
        self.assertEqual(to_copy, [])

    def test_readme_is_mirrored_because_the_space_reads_it(self):
        write(self.source, "README.md", b"---\nsdk: docker\n---\n")
        self.assertEqual(self.plan(["README.md"], [])[0], ["README.md"])

    def test_stray_target_files_are_deleted_but_target_only_files_kept(self):
        for rel in ("debug_paths.py", ".gitattributes", ".gitignore", ".vscode/settings.json"):
            write(self.target, rel, b"")
        _, to_delete = self.plan([], ["debug_paths.py", ".gitattributes", ".gitignore", ".vscode/settings.json"])
        self.assertEqual(to_delete, ["debug_paths.py"])


if __name__ == "__main__":
    unittest.main()
