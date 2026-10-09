# schema.py
"""
Creates and upgrades THIS APP's own tables once, at startup, instead of
from inside request handlers.

Each domain still owns its DDL (router/auth.py, storage.py, ...); this module
only decides when it runs. Before MS-657 every handler called its domain's
_ensure_* first: auth ran four ALTER TABLE users statements, each taking an
ACCESS EXCLUSIVE lock on users, on every login, register and admin call.

If the database is unreachable at startup the app still starts (sources
that don't need the database keep working) and the error is logged. The
tables are created on the next start.
"""

import logging

import db

logger = logging.getLogger(__name__)


def ensure_all() -> bool:
    """Run every domain's table creation. Returns False if the database was
    unreachable; each step logs and swallows its own SQL errors, as before."""
    # Imported here: the routers import half the app, and main.py imports
    # this module before the routers are mounted.
    import storage
    from router import auth, database_connections, payment, public_links, sessions, telegram

    # storage first: it creates collections, folders and chat_sessions,
    # which later tables reference. It opens its own connection.
    storage.ensure_schema()

    conn = db.get_conn("schema")
    if conn is None:
        logger.error("schema: app database unreachable at startup, tables not ensured")
        return False
    try:
        auth._ensure_users_table(conn)
        sessions._ensure_tables(conn)
        public_links._ensure_tables(conn)
        database_connections._ensure_tables(conn)
        payment._ensure_tables(conn)
        payment._ensure_usage_tables(conn)
        telegram._ensure_tables(conn)
    finally:
        conn.close()
    logger.info("schema: app tables ensured")
    return True
