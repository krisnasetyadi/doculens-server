# db.py
"""
The one way application code reaches THIS APP's own PostgreSQL database
(users, sessions, collections, payments, ... — never a user-connected
external source, which router/database_connections.py opens per request on
purpose).

Connections come from a process-wide pool instead of being opened per call:
opening one costs a TCP + TLS + auth round trip (~50-200 ms locally, more to
a cloud database), while the queries most requests run take well under a
millisecond.

get_conn() returns an object that behaves like a psycopg2 connection
(RealDictCursor rows, autocommit on), so existing call sites keep their
`conn = ...; try: ... finally: conn.close()` shape — close() hands the
connection back to the pool rather than closing it.

psycopg2.pool is not used: it keeps only `minconn` idle connections and
opens all of them up front, so it can't be both lazy and actually reuse
connections.
"""

import logging
import os
import threading
import time
from typing import Optional

from config import config

logger = logging.getLogger(__name__)

# Upper bound on connections held open by this process. Requests beyond it
# get a one-off connection (closed after use) instead of waiting, so a burst
# is never slower than before pooling — just not faster.
POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))
# A connection idle longer than this is checked with SELECT 1 before reuse:
# cloud poolers and NAT gateways drop idle connections silently, and a dead
# one would otherwise fail the request that picked it up.
PING_AFTER_SECONDS = float(os.getenv("DB_POOL_PING_AFTER_SECONDS", "30"))
CONNECT_TIMEOUT_SECONDS = 10

_lock = threading.Lock()
_idle: list = []        # [(raw connection, monotonic time it was returned)]
_open = 0               # pooled connections that exist, idle or checked out
_generation = 0         # bumped by close_pool(); older connections are closed on return


def database_url() -> Optional[str]:
    url = os.getenv("DATABASE_URL") or getattr(config, "database_url", None)
    if not url:
        return None
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url


def _connect(url: str):
    import psycopg2
    from psycopg2.extras import RealDictCursor
    conn = psycopg2.connect(url, cursor_factory=RealDictCursor, connect_timeout=CONNECT_TIMEOUT_SECONDS)
    conn.autocommit = True
    return conn


class PooledConnection:
    """A psycopg2 connection whose close() returns it to the pool."""

    __slots__ = ("_conn", "_generation")

    def __init__(self, conn, generation: Optional[int]):
        object.__setattr__(self, "_conn", conn)
        # None marks a one-off connection that is not counted in the pool.
        object.__setattr__(self, "_generation", generation)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        setattr(self._conn, name, value)

    def close(self) -> None:
        conn = self._conn
        if conn is None:
            return
        object.__setattr__(self, "_conn", None)
        _release(conn, self._generation)


def _forget_one() -> None:
    global _open
    with _lock:
        _open -= 1


def _release(conn, generation: Optional[int]) -> None:
    if generation is None:
        conn.close()
        return
    keep = not conn.closed
    if keep and not conn.autocommit:
        # A caller that turned autocommit off (storage.py's multi-row
        # writes) and returned early must not hand an open transaction to
        # the next request.
        try:
            conn.rollback()
            conn.autocommit = True
        except Exception as exc:
            logger.warning("db: discarding connection that failed to reset: %s", exc)
            keep = False
    global _open
    with _lock:
        current = generation == _generation
        if keep and current:
            _idle.append((conn, time.monotonic()))
            return
        if current:
            _open -= 1
    conn.close()


def _is_alive(conn) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return True
    except Exception:
        # Any failure here means the connection can't serve a request;
        # the caller drops it and takes another.
        return False


def _checkout(url: str):
    """(connection, generation) from the pool, or None when it is full."""
    global _open
    while True:
        with _lock:
            generation = _generation
            if _idle:
                conn, returned_at = _idle.pop()   # most recently used: least likely to be stale
            elif _open < POOL_MAX:
                _open += 1
                conn = None
            else:
                return None
        if conn is None:
            try:
                return _connect(url), generation
            except Exception:
                _forget_one()
                raise
        if not conn.closed and (time.monotonic() - returned_at < PING_AFTER_SECONDS or _is_alive(conn)):
            conn.autocommit = True
            return conn, generation
        _forget_one()
        conn.close()


def get_conn(source: str = "app_db") -> Optional[PooledConnection]:
    """A connection to the app's own database, or None if unconfigured or
    unreachable. `source` only labels the warning logged on failure."""
    url = database_url()
    if not url:
        return None
    try:
        checked_out = _checkout(url)
        if checked_out is None:
            logger.warning("db: pool of %d is full, opening a one-off connection for %s", POOL_MAX, source)
            return PooledConnection(_connect(url), None)
        return PooledConnection(*checked_out)
    except Exception as exc:
        logger.warning("%s: app DB connection failed: %s", source, exc)
        return None


def close_pool() -> None:
    """Close every idle pooled connection; ones still checked out are closed
    when returned (application shutdown, tests)."""
    global _open, _generation
    with _lock:
        idle = [conn for conn, _ in _idle]
        _idle.clear()
        _open = 0
        _generation += 1
    for conn in idle:
        try:
            conn.close()
        except Exception as exc:
            logger.debug("db: closing idle connection failed: %s", exc)
