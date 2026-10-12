"""MS-657 part 5: every function in services/storage/supabase.py that takes a
pooled connection gives it back, also when the query fails. A connection
closed only on the happy path never returns to the pool (db.get_conn), so
each failure used to cost the pool one connection for good."""

import inspect
import unittest
from unittest.mock import patch

from services.storage import supabase as storage

# Functions that take a connection (db.get_conn) — found by reading the module.
TAKES_A_CONNECTION = [
    name for name, fn in inspect.getmembers(storage, inspect.isfunction)
    if fn.__module__ == storage.__name__ and "get_conn" in inspect.getsource(fn)
]


class FailingConn:
    def __init__(self):
        self.closed = False

    def cursor(self, *args, **kwargs):
        raise RuntimeError("query failed")

    def __getattr__(self, name):  # autocommit, commit, rollback ...
        return lambda *a, **k: None

    def close(self):
        self.closed = True


def dummy_args(fn):
    values = {str: "x", int: 1, bool: False, list: [], dict: {}}
    args = []
    for p in inspect.signature(fn).parameters.values():
        if p.default is not inspect.Parameter.empty:
            continue
        annotation = p.annotation if isinstance(p.annotation, type) else str
        origin = getattr(p.annotation, "__origin__", None)
        args.append(False if p.name.startswith("is_") else values.get(origin or annotation, "x"))
    return args


class ConnectionReturnedTest(unittest.TestCase):
    def test_the_list_is_found(self):
        self.assertGreaterEqual(len(TAKES_A_CONNECTION), 25)

    def test_every_function_closes_its_connection_when_the_query_fails(self):
        leaking = []
        for name in TAKES_A_CONNECTION:
            fn = getattr(storage, name)
            conn = FailingConn()
            taken = []
            with patch.object(storage.db, "get_conn", lambda source="": taken.append(1) or conn), \
                    patch.object(storage, "_s3_client", lambda: None):
                try:
                    fn(*dummy_args(fn))
                except Exception:
                    pass  # raising is fine here; keeping the connection is not
            if taken and not conn.closed:
                leaking.append(name)
        self.assertEqual(leaking, [])


if __name__ == "__main__":
    unittest.main()
