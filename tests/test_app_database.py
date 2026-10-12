"""MS-657 part 5: the health check of the app database manager
(services/rag/app_database.py), behind GET /health. A live connection must
report healthy without reconnecting; a dead one is replaced."""

import unittest
from unittest.mock import MagicMock, patch

from services.rag.app_database import DatabaseManager


def live_connection(table_count=7):
    conn = MagicMock(closed=False)
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = {"total": table_count}
    return conn


class HealthCheckTest(unittest.TestCase):
    def manager(self, conn):
        m = DatabaseManager()
        m.connection = conn
        return m

    def test_a_live_connection_is_healthy_without_reconnecting(self):
        m = self.manager(live_connection())
        with patch.object(m, "connect") as connect:
            self.assertTrue(m.check_connection_health())
            status = m.is_healthy()
        connect.assert_not_called()
        self.assertEqual((status["status"], status["table_count"]), ("connected", 7))
        self.assertGreater(m._last_health_check, 0)

    def test_a_dead_connection_is_replaced(self):
        dead = MagicMock(closed=False)
        dead.cursor.side_effect = RuntimeError("server closed the connection")
        m = self.manager(dead)
        fresh = live_connection()
        with patch.object(m, "connect", side_effect=lambda: setattr(m, "connection", fresh)) as connect:
            self.assertTrue(m.check_connection_health())
        connect.assert_called_once()
        self.assertIs(m.connection, fresh)

    def test_no_connection_is_not_healthy(self):
        self.assertFalse(self.manager(None).check_connection_health())
        self.assertEqual(self.manager(None).is_healthy()["status"], "disconnected")


if __name__ == "__main__":
    unittest.main()
