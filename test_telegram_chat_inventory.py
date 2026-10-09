import unittest
from unittest.mock import MagicMock, patch

import storage


class TelegramChatInventoryTests(unittest.TestCase):
    def test_only_chats_linked_to_active_telegram_connections_remain_active(self):
        rows = [
            {"collection_id": "linked", "metadata": {"platform": "telegram"}, "status": "active"},
            {"collection_id": "orphan", "metadata": {"platform": "telegram"}, "status": "active"},
            {"collection_id": "whatsapp", "metadata": {"platform": "whatsapp"}, "status": "active"},
        ]
        cursor = MagicMock()
        cursor.fetchall.side_effect = [rows, [{"chat_collection_id": "linked"}]]
        cursor.fetchone.return_value = {"connections": "telegram_connections", "selected_chats": "telegram_selected_chats"}
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor

        with patch.object(storage, "ensure_schema"), patch.object(storage.db, "get_conn", return_value=connection):
            collections = storage.list_chat_collections()

        self.assertEqual(
            {collection["collection_id"]: collection["status"] for collection in collections},
            {"linked": "active", "orphan": "inactive", "whatsapp": "active"},
        )
        self.assertIn("connection.status = 'active'", cursor.execute.call_args_list[-1].args[0])

    def test_orphan_is_inactive_when_telegram_tables_are_absent(self):
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            {"collection_id": "orphan", "metadata": {"platform": "telegram"}, "status": "active"},
        ]
        cursor.fetchone.return_value = {"connections": None, "selected_chats": None}
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor

        with patch.object(storage, "ensure_schema"), patch.object(storage.db, "get_conn", return_value=connection):
            collections = storage.list_chat_collections()

        self.assertEqual(collections[0]["status"], "inactive")


if __name__ == "__main__":
    unittest.main()
