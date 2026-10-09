"""MS-657: time the main request paths so a refactor step can be shown not
to make any of them slower.

Paths timed (each N times, median and p95 reported):
  agnostic_query       POST /api/v1/agnostic/query. The first call is
                       reported separately as "cold"; it is only truly
                       cold when the server was started just before this
                       script runs.
  list_pdf_sources     GET /api/v1/pdf-collections
  list_chat_sources    GET /api/v1/chat-collections
  list_sessions        GET /api/v1/sessions
  load_session         GET /api/v1/sessions/{first session id}

It also times, in-process, how long the app takes to get a database
connection (app_db.get_app_conn() + close()), which is the per-request
cost this story removes.

Usage:
    python scripts/perf_baseline.py --label before \
        --base-url http://127.0.0.1:8765 --mint-for-user-id <local user id>
    python scripts/perf_baseline.py --compare before after

Results are written to perf/timings/<label>.json.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
TIMINGS_DIR = os.path.join(ROOT, "perf", "timings")

QUESTION = "Apa isi utama dokumen yang tersedia?"


def _stats(samples_ms):
    ordered = sorted(samples_ms)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {"runs": len(ordered), "median_ms": round(statistics.median(ordered), 1),
            "p95_ms": round(p95, 1), "min_ms": round(ordered[0], 1)}


def _time(fn, runs):
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def time_db_connect(runs):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    import app_db

    def once():
        conn = app_db.get_app_conn("perf_baseline")
        if conn is None:
            raise SystemExit("app database not reachable")
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        conn.close()

    once()  # module import / DNS warm-up is not what is being measured
    return _stats(_time(once, runs))


def time_http(args):
    import httpx
    from contract_snapshot import resolve_token

    client = httpx.Client(base_url=args.base_url, timeout=300,
                          headers={"Authorization": f"Bearer {resolve_token(args)}"})

    def get(path):
        def call():
            resp = client.get(path)
            resp.raise_for_status()
            return resp
        return call

    results = {}
    sessions = get("/api/v1/sessions")().json()

    def query():
        resp = client.post("/api/v1/agnostic/query", json={"question": QUESTION})
        resp.raise_for_status()

    cold = _time(query, 1)
    results["agnostic_query_cold"] = _stats(cold)
    results["agnostic_query_warm"] = _stats(_time(query, args.query_runs))

    for name, path in (("list_pdf_sources", "/api/v1/pdf-collections"),
                       ("list_chat_sources", "/api/v1/chat-collections"),
                       ("list_sessions", "/api/v1/sessions")):
        results[name] = _stats(_time(get(path), args.runs))
    if sessions:
        results["load_session"] = _stats(_time(get(f"/api/v1/sessions/{sessions[0]['session_id']}"), args.runs))
    return results


def compare(old_label, new_label):
    old = json.load(open(os.path.join(TIMINGS_DIR, f"{old_label}.json"), encoding="utf-8"))["paths"]
    new = json.load(open(os.path.join(TIMINGS_DIR, f"{new_label}.json"), encoding="utf-8"))["paths"]
    slower = 0
    print(f"{'path':24} {old_label:>12} {new_label:>12}   change")
    for name in old:
        if name not in new:
            print(f"{name:24} missing in {new_label}")
            continue
        a, b = old[name]["median_ms"], new[name]["median_ms"]
        change = (b - a) / a * 100 if a else 0
        # 10% slack: below that, run-to-run noise on one machine is larger
        # than any real regression.
        flag = "  SLOWER" if change > 10 else ""
        slower += bool(flag)
        print(f"{name:24} {a:>10.1f}ms {b:>10.1f}ms {change:+7.1f}%{flag}")
    sys.exit(1 if slower else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token")
    parser.add_argument("--mint-for-user-id")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--query-runs", type=int, default=5)
    parser.add_argument("--db-only", action="store_true", help="only time getting a DB connection")
    parser.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if not args.label:
        parser.error("--label is required")

    paths = {"db_connection": time_db_connect(args.runs)}
    if not args.db_only:
        paths.update(time_http(args))
    os.makedirs(TIMINGS_DIR, exist_ok=True)
    out = os.path.join(TIMINGS_DIR, f"{args.label}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "machine": platform.platform(), "base_url": args.base_url,
                   "paths": paths}, f, indent=2)
        f.write("\n")
    for name, s in paths.items():
        print(f"{name:24} median {s['median_ms']:>8.1f}ms  p95 {s['p95_ms']:>8.1f}ms  ({s['runs']} runs)")
    print(f"written to {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
