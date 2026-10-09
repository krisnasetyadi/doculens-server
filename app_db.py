# app_db.py
"""
Formatting helper for rows read from THIS APP's own database. Connections
come from db.get_conn().
"""

from datetime import datetime, timezone
from typing import Any


def ts(value: Any) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
