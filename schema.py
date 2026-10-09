# schema.py
"""
Creates and upgrades THIS APP's own tables once, at startup, instead of
from inside request handlers.

Each domain still owns its DDL (router/auth.py, storage.py, ...); this module
only decides when it runs. Before MS-657 every handler called its domain's
_ensure_* first: auth ran four ALTER TABLE users statements, each taking an
ACCESS EXCLUSIVE lock on users, on every login, register and admin call.

Startup never fails because of the schema. If the database is unreachable,
or one domain's DDL fails (several _ensure_* raise HTTPException(500) for
that), the error is logged, the other domains still run, and the app starts
so the parts that don't need the failed tables keep working. The failed
tables are retried on the next start.
"""

import logging

import db

logger = logging.getLogger(__name__)


def ensure_all() -> bool:
    """Run every domain's table creation. Returns False if the database was
    unreachable or any domain failed; never raises."""
    # Imported here: the routers import half the app, and main.py imports
    # this module before the routers are mounted.
    import storage
    from router import auth, database_connections, payment, public_links, sessions, telegram

    # storage first: it creates collections, folders and chat_sessions,
    # which later tables reference. It opens its own connection and logs
    # its own failures.
    storage.ensure_schema()

    conn = db.get_conn("schema")
    if conn is None:
        logger.error("schema: app database unreachable at startup, tables not ensured")
        return False
    steps = [
        ("auth", auth._ensure_users_table),
        ("sessions", sessions._ensure_tables),
        ("public_links", public_links._ensure_tables),
        ("database_connections", database_connections._ensure_tables),
        ("payment", payment._ensure_tables),
        ("payment usage", payment._ensure_usage_tables),
        ("telegram", telegram._ensure_tables),
    ]
    failed = []
    try:
        for name, ensure in steps:
            try:
                ensure(conn)
            except Exception as exc:
                # Caught so one domain's failure can't stop startup or the
                # other domains; its own routes fail until the next start.
                logger.error("schema: %s tables not ensured: %s", name, getattr(exc, "detail", exc))
                failed.append(name)
    finally:
        conn.close()
    if failed:
        logger.error("schema: failed for %s, retried on next start", ", ".join(failed))
        return False
    logger.info("schema: app tables ensured")
    return True
