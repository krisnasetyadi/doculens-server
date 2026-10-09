# app_db.py
"""
Shared helper for connecting to THIS APP's own database — the metadata
store for routers that keep a small schema of their own (database
connections, Telegram connections, ...), never a user-connected external
source. Previously duplicated near-verbatim across router/database_connections.py
and router/telegram.py; kept in one place so a fix (e.g. the sslmode
heuristic, connect timeout) can't silently apply to only one of them.
"""

from datetime import datetime, timezone
from typing import Any

import db


def get_app_conn(source: str = "app_db"):
    """A pooled connection to the app's own database, or None if unconfigured/unreachable."""
    return db.get_conn(source)


def ts(value: Any) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
