# schema/__init__.py
"""
Creates and upgrades THIS APP's own tables once, at startup, instead of
from inside request handlers.

Each domain's DDL lives in its own module here (schema/auth.py,
schema/payment.py, ...); this one decides when it runs. Before MS-657 every
handler called its domain's _ensure_* first: auth ran four ALTER TABLE users
statements, each taking an ACCESS EXCLUSIVE lock on users, on every login,
register and admin call.

Startup never fails because of the schema. If the database is unreachable,
or one domain's DDL fails (several ensure() raise HTTPException(500) for
that), the error is logged, the other domains still run, and the app starts
so the parts that don't need the failed tables keep working. The failed
tables are retried on the next start.
"""

import logging

import db
from schema import auth, database_connections, payment, public_links, sessions, storage, telegram

logger = logging.getLogger(__name__)


def ensure_all() -> bool:
    """Run every domain's table creation. Returns False if the database was
    unreachable or any domain failed; never raises."""
    # storage first: it creates collections, folders and chat_sessions,
    # which later tables reference. It opens its own connection and logs
    # its own failures.
    storage.ensure()

    conn = db.get_conn("schema")
    if conn is None:
        logger.error("schema: app database unreachable at startup, tables not ensured")
        return False
    steps = [
        ("auth", auth.ensure),
        ("sessions", sessions.ensure),
        ("public_links", public_links.ensure),
        ("database_connections", database_connections.ensure),
        ("payment", payment.ensure),
        ("payment usage", payment.ensure_usage),
        ("telegram", telegram.ensure),
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
