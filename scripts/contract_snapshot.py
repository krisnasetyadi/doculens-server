"""MS-657: record the API contract so every refactor step can be checked
against it.

Two parts, stored under perf/contract/<label>/:
  openapi.json   the app's /openapi.json (paths, methods, request/response
                 schemas), generated in-process — no server needed.
  shapes.json    for every route that can be called safely, the status
                 code and the *shape* of a real response (keys and value
                 types, not values), so a field that disappears or changes
                 type shows up even where no response_model is declared yet.

Raw responses go to perf/contract/<label>/raw/ (gitignored: they hold real
chat content and emails). Routes that change data are never called; they
are listed in shapes.json under "not_captured" with the reason.

Usage:
    python scripts/contract_snapshot.py openapi --label before
    python scripts/contract_snapshot.py capture --label before \
        --base-url http://127.0.0.1:8765 --mint-for-user-id <local user id>
    python scripts/contract_snapshot.py compare before after

--mint-for-user-id signs a token with JWT_SECRET from .env; only use it
against a local database. Pass --token instead for any other server.
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CONTRACT_DIR = os.path.join(ROOT, "perf", "contract")

# Path parameters filled from ids found in earlier list responses:
# {id name: (list route, key holding the id in each item)}.
ID_SOURCES = {
    "session_id": ("/api/v1/sessions", "session_id"),
    "pdf_collection_id": ("/api/v1/pdf-collections", "collection_id"),
    "chat_collection_id": ("/api/v1/chat-collections", "collection_id"),
    "run_id": ("/api/v1/analysis/gap-analysis/runs", "run_id"),
    "skill_id": ("/api/v1/skills", "skill_id"),
    "connection_id": ("/api/v1/database-connections", "connection_id"),
    "folder_id": ("/api/v1/source-folders", "folder_id"),
}


def _id_name(path: str, param: str) -> str:
    """PDF and chat routes both call their path parameter collection_id."""
    if param == "collection_id":
        return "chat_collection_id" if "/chat-collections/" in path else "pdf_collection_id"
    return param

# Read-only routes skipped on purpose.
SKIP_GET = {
    "/api/v1/payments/session/{session_id}": "needs a Stripe checkout session id",
    "/api/v1/pdf-collections/uploads/{upload_id}": "needs an in-flight upload id",
    "/api/v1/chat-collections/uploads/{upload_id}": "needs an in-flight upload id",
    "/api/v1/files/{collection_id}/{file_name}": "binary file download",
    "/api/v1/analysis/gap-analysis/{run_id}/export": "binary export",
    "/api/v1/telegram-connections/{connection_id}/dialogs": "calls Telegram",
    "/api/v1/database-connections/{connection_id}/tables": "connects to the user's external database",
}


def _out_dir(label: str) -> str:
    path = os.path.join(CONTRACT_DIR, label)
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _load_spec() -> dict:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    import main
    return main.app.openapi()


def cmd_openapi(args) -> None:
    spec = _load_spec()
    _write_json(os.path.join(_out_dir(args.label), "openapi.json"), spec)
    count = sum(len(ops) for ops in spec["paths"].values())
    print(f"openapi.json: {count} routes")


# ---------------------------------------------------------------------------
# Response shapes
# ---------------------------------------------------------------------------

def shape(value):
    """Keys and types of a JSON value, with list items merged into one shape."""
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    if isinstance(value, list):
        merged = None
        for item in value:
            merged = merge(merged, shape(item))
        return [merged] if merged is not None else []
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return "str"


def merge(a, b):
    if a is None:
        return b
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: merge(a.get(k), b.get(k)) if k in a and k in b else (a.get(k) if k in a else b.get(k))
                for k in sorted(set(a) | set(b))}
    if isinstance(a, list) and isinstance(b, list):
        return [merge(a[0] if a else None, b[0] if b else None)] if (a or b) else []
    if a == b:
        return a
    if isinstance(a, str) and isinstance(b, str):
        return "|".join(sorted(set(a.split("|")) | set(b.split("|"))))
    # A scalar in one item and an object/list in another: keep both visibly.
    return "|".join(sorted({json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True)}))


def _mint_token(user_id: str) -> str:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    import psycopg2
    from jose import jwt
    conn = psycopg2.connect(os.environ["DATABASE_URL"], connect_timeout=10)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT email, role, name FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise SystemExit(f"no user {user_id} in the local database")
    payload = {"sub": user_id, "email": row[0], "role": row[1],
               "exp": datetime.now(timezone.utc) + timedelta(hours=2)}
    if row[2]:
        payload["name"] = row[2]
    return jwt.encode(payload, os.environ["JWT_SECRET"], algorithm="HS256")


def resolve_token(args) -> str:
    if args.token:
        return args.token
    if args.mint_for_user_id:
        return _mint_token(args.mint_for_user_id)
    raise SystemExit("pass --token or --mint-for-user-id")


def _first_id(body, key):
    items = body if isinstance(body, list) else next((v for v in (body or {}).values() if isinstance(v, list)), [])
    for item in items:
        if isinstance(item, dict) and item.get(key):
            return str(item[key])
    return None


def cmd_capture(args) -> None:
    import httpx

    spec = json.load(open(os.path.join(_out_dir(args.label), "openapi.json"), encoding="utf-8"))
    out = _out_dir(args.label)
    raw_dir = os.path.join(out, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    client = httpx.Client(base_url=args.base_url, timeout=120,
                          headers={"Authorization": f"Bearer {resolve_token(args)}"})

    gets = sorted(p for p, ops in spec["paths"].items() if "get" in ops)
    # List routes first so their ids are known before the detail routes run.
    gets.sort(key=lambda p: "{" in p)
    ids, shapes, not_captured = {}, {}, {}

    for path, ops in sorted(spec["paths"].items()):
        for method in ops:
            if method != "get":
                not_captured[f"{method.upper()} {path}"] = "changes data"

    for path in gets:
        if path in SKIP_GET:
            not_captured[f"GET {path}"] = SKIP_GET[path]
            continue
        url = path
        missing = None
        for param in re.findall(r"{(\w+)}", path):
            value = ids.get(_id_name(path, param))
            if value is None:
                missing = param
                break
            url = url.replace("{" + param + "}", value)
        if missing:
            not_captured[f"GET {path}"] = f"no {missing} available"
            continue
        if path == "/api/v1/pdf-collections/{collection_id}/text-content":
            if "txt_source" not in ids:
                not_captured[f"GET {path}"] = "no .txt document source available"
                continue
            txt_collection, txt_file = ids["txt_source"]
            url = path.replace("{collection_id}", txt_collection) + f"?file_name={quote(txt_file)}"
        resp = client.get(url)
        try:
            body = resp.json()
        except ValueError:
            body = {"_non_json_bytes": len(resp.content)}
        shapes[f"GET {path}"] = {"status": resp.status_code, "shape": shape(body)}
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_") or "root"
        _write_json(os.path.join(raw_dir, f"{safe_name}.json"), {"status": resp.status_code, "body": body})
        for id_name, (source_path, key) in ID_SOURCES.items():
            if path == source_path and id_name not in ids and resp.status_code == 200:
                found = _first_id(body, key)
                if found:
                    ids[id_name] = found
        if path == "/api/v1/pdf-collections" and resp.status_code == 200:
            for item in body:
                txt = next((f for f in item.get("file_names") or [] if f.lower().endswith(".txt")), None)
                if txt:
                    ids["txt_source"] = (item["collection_id"], txt)
                    break
        print(f"{resp.status_code}  GET {path}")

    _write_json(os.path.join(out, "shapes.json"), {"captured": shapes, "not_captured": not_captured})
    print(f"captured {len(shapes)}, not captured {len(not_captured)}")


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def _same_scalar_shape(a, b) -> bool:
    """"str" vs "null|str" is the same contract seen with different data."""
    ta, tb = set(a.split("|")) - {"null"}, set(b.split("|")) - {"null"}
    return not ta or not tb or ta == tb


def _diff(a, b, where, out, shapes=False):
    if shapes:
        if isinstance(a, str) and isinstance(b, str) and _same_scalar_shape(a, b):
            return
        if a == [] or b == [] or a == "null" or b == "null":
            return
    if type(a) is not type(b) or (not isinstance(a, (dict, list)) and a != b):
        out.append(f"{where}: {json.dumps(a, sort_keys=True)[:120]} -> {json.dumps(b, sort_keys=True)[:120]}")
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in b:
                out.append(f"{where}.{k}: removed")
            elif k not in a:
                out.append(f"{where}.{k}: added")
            else:
                _diff(a[k], b[k], f"{where}.{k}", out, shapes)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{where}: list length {len(a)} -> {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{where}[{i}]", out, shapes)


def cmd_compare(args) -> None:
    problems = []
    for name in ("openapi.json", "shapes.json"):
        old_path = os.path.join(CONTRACT_DIR, args.old, name)
        new_path = os.path.join(CONTRACT_DIR, args.new, name)
        if not (os.path.exists(old_path) and os.path.exists(new_path)):
            print(f"skip {name}: missing in one snapshot")
            continue
        old = json.load(open(old_path, encoding="utf-8"))
        new = json.load(open(new_path, encoding="utf-8"))
        if name == "openapi.json":
            # Route set and schemas only; info.title/version are not contract.
            old, new = {k: old.get(k) for k in ("paths", "components")}, {k: new.get(k) for k in ("paths", "components")}
        else:
            old, new = old["captured"], new["captured"]
        _diff(old, new, name, problems, shapes=(name == "shapes.json"))
    for p in problems:
        print(p)
    print(f"{len(problems)} difference(s)")
    sys.exit(1 if problems else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("openapi")
    p.add_argument("--label", required=True)
    p = sub.add_parser("capture")
    p.add_argument("--label", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--token")
    p.add_argument("--mint-for-user-id")
    p = sub.add_parser("compare")
    p.add_argument("old")
    p.add_argument("new")
    args = parser.parse_args()
    {"openapi": cmd_openapi, "capture": cmd_capture, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
