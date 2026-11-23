"""MS-1181: matching a name the user said to folders in their Lens Guide scope."""

import unittest

from services.lens_guide.match import (
    collection_ids_in, find_mentioned_folders, folder_label, match_folders, resolve_folder_label, subfolders,
)
from services.lens_guide.scope import Scope, ScopeCollection, ScopeFolder


def folder(folder_id, *path):
    return ScopeFolder(folder_id=folder_id, name=path[-1], parent_folder_id=None, path=list(path))


SCOPE = Scope(folders=[
    folder("d", "Delivery"),
    folder("y", "Delivery", "Proyek Klien Y"),
    folder("yb", "Delivery", "Proyek Klien Y", "Backend"),
    folder("x", "Delivery", "Proyek Klien X"),
    folder("a", "Arsip", "Proyek Klien Y"),
    folder("k", "Kebijakan Perusahaan"),
])


def ids(folders):
    return [f.folder_id for f in folders]


class MatchFoldersTests(unittest.TestCase):
    def test_exact_name_at_any_level_ignoring_filler_and_case(self):
        self.assertEqual(ids(match_folders("project kebijakan PERUSAHAAN", SCOPE)), ["k"])
        self.assertEqual(ids(match_folders("Backend", SCOPE)), ["yb"])

    def test_same_name_in_two_places_is_ambiguous(self):
        self.assertEqual(ids(match_folders("Klien Y", SCOPE)), ["a", "y"])

    def test_path_words_pick_one_and_skip_its_subfolders(self):
        self.assertEqual(ids(match_folders("Delivery Klien Y", SCOPE)), ["y"])

    def test_part_of_a_name_matches_every_folder_holding_it(self):
        self.assertEqual(ids(match_folders("klien", SCOPE)), ["a", "x", "y"])

    def test_no_match_and_empty_name(self):
        self.assertEqual(match_folders("Klien Z", SCOPE), [])
        self.assertEqual(match_folders("proyek", SCOPE), [])
        self.assertEqual(match_folders("", SCOPE), [])

    def test_only_folders_in_the_scope_can_match(self):
        self.assertEqual(match_folders("Klien Y", Scope()), [])


class FolderHelpersTests(unittest.TestCase):
    def test_subfolders_and_their_collections(self):
        tree = Scope(
            folders=[
                ScopeFolder("d", "Delivery", None, ["Delivery"]),
                ScopeFolder("y", "Proyek Klien Y", "d", ["Delivery", "Proyek Klien Y"]),
                ScopeFolder("b", "Backend", "y", ["Delivery", "Proyek Klien Y", "Backend"]),
                ScopeFolder("k", "Kebijakan", None, ["Kebijakan"]),
            ],
            collections=[
                ScopeCollection("c1", "Arsitektur", 1, "b", []),
                ScopeCollection("c2", "Cuti", 1, "k", []),
                ScopeCollection("c3", "Root", 1, None, []),
            ],
        )
        self.assertEqual(ids(subfolders(tree.folders[0], tree)), ["y", "b"])
        self.assertEqual(collection_ids_in([tree.folders[0]], tree), ["c1"])
        self.assertEqual(collection_ids_in([tree.folders[3]], tree), ["c2"])

    def test_resolve_folder_label_exact_then_by_name(self):
        self.assertEqual(ids(resolve_folder_label("delivery › proyek klien y", SCOPE)), ["y"])
        self.assertEqual(ids(resolve_folder_label("Delivery > Klien Y", SCOPE)), ["y"])  # free text past the cap
        self.assertEqual(folder_label(SCOPE.folders[2]), "Delivery › Proyek Klien Y › Backend")


class FindMentionedFoldersTests(unittest.TestCase):
    def test_whole_name_as_words(self):
        self.assertEqual(
            ids(find_mentioned_folders("aku mau baca dokumen Kebijakan Perusahaan dulu", SCOPE)), ["k"]
        )
        self.assertEqual(ids(find_mentioned_folders("aku baru masuk proyek klien Y.", SCOPE)), ["y", "a"])

    def test_a_one_word_name_counts_only_after_proyek_modul_or_folder(self):
        self.assertEqual(ids(find_mentioned_folders("pelajari modul Backend", SCOPE)), ["yb"])
        self.assertEqual(find_mentioned_folders("tolong cek backend nya", SCOPE), [])
        generic = Scope(folders=[folder("l", "Laporan")])
        self.assertEqual(find_mentioned_folders("ringkas dokumen laporan bulan ini", generic), [])
        self.assertEqual(ids(find_mentioned_folders("isi folder laporan apa saja?", generic)), ["l"])

    def test_part_of_a_word_or_a_short_name_is_not_a_mention(self):
        self.assertEqual(find_mentioned_folders("ringkas kebijakan cuti", SCOPE), [])
        self.assertEqual(find_mentioned_folders("klien yang baru", SCOPE), [])
        short = Scope(folders=[folder("q", "QA")])
        self.assertEqual(find_mentioned_folders("modul qa", short), [])


if __name__ == "__main__":
    unittest.main()
