# MS-657 contract snapshots and timings

Every refactor step in MS-657 is checked against what is recorded here: the
API contract must not change, and no main path may get slower.

## Contract (`contract/<label>/`)

- `openapi.json` — the app's OpenAPI spec (all 89 routes).
- `shapes.json` — status code and shape (keys and value types) of a real
  response for each of the 30 routes that can be called without changing
  data. The other 59 routes are listed under `not_captured` with the reason
  (mostly "changes data").
- `raw/` — the full responses. Gitignored: they contain real chat content
  and emails.

```
python scripts/contract_snapshot.py openapi --label <label>
python scripts/contract_snapshot.py capture --label <label> --base-url http://127.0.0.1:8765 --mint-for-user-id <local user id>
python scripts/contract_snapshot.py compare before <label>     # exits 1 on any difference
```

## Timings (`timings/<label>.json`)

```
python scripts/perf_baseline.py --label <label> --base-url http://127.0.0.1:8765 --mint-for-user-id <local user id>
python scripts/perf_baseline.py --compare before <label>       # exits 1 if a path is >10% slower
```

Measured on 9 October 2026 on the local dev machine (Windows 11, local
PostgreSQL, Gemini as the LLM) as the admin user of the local database
(26 sessions, 15 collections). The server ran with
`FREE_PLAN_TOKEN_LIMIT=100000000` so the workspace's free-plan quota didn't
block the repeated queries. Use the same setting when you compare.

| Path (median) | before | after pool | change |
|---|---|---|---|
| Get a DB connection | 51.9 ms | 0.1 ms | -99.8% |
| List PDF sources | 186.2 ms | 4.4 ms | -97.6% |
| List chat sources | 259.8 ms | 8.0 ms | -96.9% |
| List sessions | 164.6 ms | 4.9 ms | -97.0% |
| Load one session | 144.8 ms | 5.0 ms | -96.5% |
| `/agnostic/query` warm | 5455 ms | 5293 ms | -3.0% |
| `/agnostic/query` cold | 9.9 s | 7.6 s | see below |

Cold query: `perf_baseline.py` takes one cold sample per run, and its
first "after" sample (10.3 s) came out slower than "before" (8.3 s). To check
whether that was noise, the server was restarted three times on each version
in alternating order, timing the first query each time:

| round | before | after pool |
|---|---|---|
| 1 | 9.91 s | 7.64 s |
| 2 | 11.88 s | 6.86 s |
| 3 | 7.69 s | 8.23 s |

Median: 9.9 s before, 7.6 s after. The time is mostly Gemini's response time,
which ranges from 7 to 12 s between runs, so the one-sample difference was
noise.
