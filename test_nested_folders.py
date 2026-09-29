import unittest
from unittest.mock import MagicMock, patch

import storage
from models import FolderRename


FOLDERS = [
    {"folder_id": "contracts", "parent_folder_id": None},
    {"folder_id": "year", "parent_folder_id": "contracts"},
    {"folder_id": "client", "parent_folder_id": "year"},
    {"folder_id": "finance", "parent_folder_id": None},
    {"folder_id": "finance-year", "parent_folder_id": "finance"},
]


class FolderHierarchyTests(unittest.TestCase):
    def test_three_levels_are_allowed_but_four_are_not(self):
        storage._validate_folder_parent(FOLDERS, "new", "year")
        with self.assertRaisesRegex(storage.FolderHierarchyError, "3 levels"):
            storage._validate_folder_parent(FOLDERS, "new", "client")

    def test_cannot_move_folder_into_itself_or_descendant(self):
        for parent_id in ("contracts", "year", "client"):
            with self.subTest(parent_id=parent_id):
                with self.assertRaises(storage.FolderHierarchyError):
                    storage._validate_folder_parent(FOLDERS, "contracts", parent_id)

    def test_moving_a_subtree_checks_its_deepest_child(self):
        with self.assertRaisesRegex(storage.FolderHierarchyError, "3 levels"):
            storage._validate_folder_parent(FOLDERS, "year", "finance-year")
        storage._validate_folder_parent(FOLDERS, "year", None)

    def test_update_distinguishes_omitted_parent_from_root(self):
        self.assertNotIn("parent_folder_id", FolderRename(name="New name").model_fields_set)
        self.assertIn("parent_folder_id", FolderRename(name="New name", parent_folder_id=None).model_fields_set)

    def test_delete_reparents_children_and_files_before_removing_folder(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = {"parent_folder_id": "contracts"}
        cursor.rowcount = 1
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor

        with patch.object(storage, "ensure_schema"), patch.object(storage, "_db_conn", return_value=connection):
            self.assertTrue(storage.delete_folder("year"))

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertIn("UPDATE folders SET parent_folder_id", statements[1])
        self.assertIn("UPDATE collections SET folder_id", statements[2])
        self.assertIn("DELETE FROM folders", statements[3])
        self.assertEqual(cursor.execute.call_args_list[1].args[1], ("contracts", "year"))
        self.assertEqual(cursor.execute.call_args_list[2].args[1], ("contracts", "year"))
        connection.commit.assert_called_once()

    def test_delete_rolls_back_if_moving_contents_fails(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = {"parent_folder_id": "contracts"}
        cursor.execute.side_effect = [None, None, RuntimeError("move failed")]
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor

        with patch.object(storage, "ensure_schema"), patch.object(storage, "_db_conn", return_value=connection):
            self.assertFalse(storage.delete_folder("year"))

        connection.rollback.assert_called_once()
        connection.commit.assert_not_called()

    def test_deleting_root_folder_moves_its_contents_to_root(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = {"parent_folder_id": None}
        cursor.rowcount = 1
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor

        with patch.object(storage, "ensure_schema"), patch.object(storage, "_db_conn", return_value=connection):
            self.assertTrue(storage.delete_folder("contracts"))

        self.assertEqual(cursor.execute.call_args_list[1].args[1], (None, "contracts"))
        self.assertEqual(cursor.execute.call_args_list[2].args[1], (None, "contracts"))


if __name__ == "__main__":
    unittest.main()
