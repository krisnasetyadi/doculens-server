# Answer streaming (MS-906)

Technical plan for showing an answer word by word while the model is still
writing it. Status: server and UI built and tested with a fake model (24 tests for streaming
on the server, a live uvicorn check, and 46 UI tests). The model layer was run
against real Gemini, on two models (section 8); the Space (`curl -N`) has not been checked.

Live check with uvicorn and a fake model that writes a piece every 0.4 s:
`sources` arrived at +0.04 s, each `delta` on its own 0.4 s later (not in one
lump), then `done`. A client that left after the first `delta` had the tokens
received so far logged once and its reservation freed.

Scope: `POST /api/v1/agnostic/query` in this repo (`pdf-reader`), mirrored
unchanged to `hf-doculens-api` by CI (`scripts/sync_hf.py`), and the chat UI
(`chat-ui`, repo `doculens-ui`). This file is a `.md`, so `sync_hf.py` does not
copy it to the Space.

## 1. Problem

Today the endpoint answers once, when everything is done. MS-657 measured
where the time goes: about 98% is the model writing the whole answer, and
retrieval is about 60 ms. The user sees a spinner for the full duration
(a median of about 5.5 s warm and 8 to 10 s cold on the dev machine, longer
for summaries) and cannot read or stop a bad answer early.

Where the wait is in the code (`services/agnostic.py`, `services/rag/answer.py`):

```
answer_query -> _answer_query
  checks: collections, canned replies, rate limit / quota, begin_in_flight
  -> _run_metered_query
       hybrid_search (retrieval, about 60 ms)
       generate_hybrid_answer
         llm.invoke(prompt)                 <- blocks for the whole answer
         _validate_and_clean_answer         <- may replace the answer
       log_token_usage
       build AgnosticQueryResponse
  finally: end_in_flight
```

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| 1 | Same path and method, `?stream=true` switches the mode | `services/endpoint.ts` in chat-ui needs no new entry. Without the flag nothing changes for existing clients. |
| 2 | Everything before the model call runs before the HTTP response starts | 401, 403, 422 and 429 stay normal status codes. The stream starts only when the model is about to be called. |
| 3 | Both modes share one pipeline | `done.result` must equal the non-streaming response. Two copies of the answer logic would drift. |
| 4 | The final validated answer replaces the streamed text | `_validate_and_clean_answer` can rewrite the answer after it is complete (for example a "not found" reply). The client must expect this. |
| 5 | Answers that never reach the model send only `done` | `/help`, unknown commands, guard replies, "select a source first", quota fallbacks and router replies have nothing to stream. The client shows them at once, as today. |
| 6 | The model is stopped when the client leaves | Same rule as MS-667, extended to a model call that is already running. Tokens used so far are still counted. |

## 3. Wire format

`Content-Type: application/x-ndjson`, one JSON object per line, each ending in
`\n`. Headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no`. The
repo already streams NDJSON for uploads (`services/ingest/progress.py`,
`stream_progress=true`) and chat-ui already reads it
(`services/upload-progress.ts`), so the shape follows that precedent.

Order: `sources`, then zero or more `delta`, then `done`. If the model fails
after the stream has started: `error` instead of `done`. Exactly one terminal
event (`done` or `error`) ends every stream.

```
{"type":"sources","pdf_sources":[...],"pdf_sources_detailed":[...],"chat_results":[...],
 "source_type":"Indexed Collections","retrieved_count":5}
{"type":"delta","text":"The policy "}
{"type":"delta","text":"requires ..."}
{"type":"done","result":{ ...the full AgnosticQueryResponse, as today... }}
```

```
{"type":"error","detail":"Failed to answer the question"}
```

Rules for the client:

- `delta.text` is appended to what is on screen. It is never a replacement.
- `done.result.answer` always replaces the streamed text, even when it is
  identical.
- A stream that ends without `done` or `error` (connection lost) is treated
  as an error, not as a finished answer.
- A reply with only `done` (decision 5) is shown at once.

Not in the stream: `search_terms`, `target_tables`, `db_results` and
`efficiency` only exist at the end, so they are in `done.result` only.

## 4. Server design

### 4.1 Split the pipeline

`answer_query` becomes two phases, so the endpoint can tell "failed before
the stream" from "failed after it":

1. **Prepare** (awaited by the router before it returns a response): all the
   checks in `_answer_query` up to and including `begin_in_flight`, the router
   LLM call when the message needs it, and retrieval (`hybrid_search`, the
   skill lookup). Any `HTTPException` here is a normal HTTP error. A canned
   reply is returned from here as a finished `AgnosticQueryResponse`.
2. **Generate** (the streamed part): build the prompt, call the model, send
   the deltas, validate, log tokens, send `done`.

`generate_hybrid_answer` (`services/rag/answer.py`) is split at the model call:

- `build_answer_prompt(...)` returns the prompt and the pieces the metadata
  needs (`merged_results`, `source_breakdown`, `conflicts`, `efficiency_meta`,
  `processing_steps`).
- `finish_answer(raw_answer, total_tokens, ...)` runs
  `_validate_and_clean_answer` and builds the metadata dict.
- `generate_hybrid_answer` stays as the non-streaming entry point: build,
  `llm.invoke`, finish. Its behavior and return value do not change.

The streaming path uses the same two functions around `llm.stream`. That is
what makes `done.result` identical to the non-streaming response.

`sources` is sent as the first line of the stream, built from the retrieval
result that Prepare already holds. The existing source mapping in
`_run_metered_query` moves into one helper used by both modes.

### 4.2 Calling the model

`llm.stream(prompt)` is a blocking generator, so it runs in a worker thread
and hands chunks to the event loop through an `asyncio.Queue` (the same
pattern as `progress_response`). The event loop side:

- yields one `delta` line per chunk,
- does not poll `StopCheck` between chunks: when the client leaves, Starlette
  cancels the stream itself (uvicorn reports ASGI spec 2.3, so
  `StreamingResponse` listens for `http.disconnect` next to the generator),
- on that cancel (or any other end), `settle()` calls `ModelStream.stop()`,
  which sets a `threading.Event` the worker checks between chunks and closes
  the model stream.

Per provider:

- **Gemini** (`ChatGoogleGenerativeAI`): streams `AIMessageChunk`s. Token
  usage is read from the summed chunks (`chunk_a + chunk_b` merges
  `usage_metadata`). Verified on 20 Oct 2026 (`gemini-flash-latest`, thinking
  budget 0): every chunk carries its own token delta, and the sum is the
  total (a 169-token answer in 8 chunks summed to `input 23 + output 169 =
  total 192`). **The last chunk reports 0**, so the total must be summed
  over all chunks, never read from the last one.
- **HuggingFace** (`HuggingFacePipeline`): has no token streaming, so
  `.stream()` returns the whole answer as one chunk. It still works: one
  `delta`, then `done`. Token count stays 0, as today.

If a model call fails before the first chunk, the existing fallback applies
(`_generate_fallback_answer`) and the result goes out as a normal `done`. If
it fails after text was already sent, send `error` (the text on screen is not
trusted any more).

### 4.3 Tokens and the reservation

- **Counted once.** One `log_token_usage` call per answer, made in the
  generator's `finally`, from whatever usage the model reported. This covers
  normal completion, stop, disconnect and error with the same code path.
- **On stop or disconnect** the model call is cancelled. If the provider
  reported no usage for a cancelled call, fall back to an estimate from the
  characters received so far (about 4 characters per token) so the tokens
  already used are not lost. Verified on 20 Oct 2026: a stream cut after 2
  chunks still reported usage for what had been received (`input 23 +
  output 40 = total 63`), because each chunk carries its own delta. So the
  sum of the chunks received is the count to log on stop or disconnect. It
  can be slightly below what Gemini bills, because tokens it had already
  generated but not yet sent are not seen. The character estimate is only
  the fallback for a provider or chunk that reports no usage at all
  (HuggingFace). Two real requests were used for this check.
- **A model call that cannot stop yet.** The worker thread only notices a stop
  when the model sends its next chunk, and the first chunk can take 8 to 14 s
  (section 8). `ModelStream.stop()` waits at most 5 s and reports whether the
  worker ended. If it did not, `settle()` logs what was counted so far, keeps
  the reservation, and a background task (`_settle_late`, at most 120 s) waits
  for the call to end, logs the tokens that arrived after that point, and then
  releases the reservation. So a call that is still running stays counted, and
  its late tokens are not lost.
- **Router tokens** (`routing_tokens`) are added to the same row, as today.
- **The reservation** (`begin_in_flight`) is released with `end_in_flight`
  after the usage is logged, exactly once. `end_in_flight` subtracts, so it is
  not safe to call twice: wrap it in a once-only guard.
- **The leak to avoid:** Prepare reserves tokens before the response exists.
  If the client disconnects before the generator ever starts, a `finally`
  inside the generator never runs. A `BackgroundTask` does not cover it
  either: with ASGI spec 2.4 servers Starlette raises `ClientDisconnect`
  and skips the background task. So the response class itself
  (`_StreamedAnswer`) settles in a `finally` around the whole response, and
  the generator settles too; `settle()` does its work once however many
  times it is called, and runs inside a shielded cancel scope because the
  server cancels the stream when the client leaves. A response that is
  returned but never run at all (the server fails between the two) is covered
  by a `weakref.finalize` on the answer, which releases the reservation. Tests disconnect before
  the first byte, mid-stream and after `done`, and check the in-flight
  counters are back to 0 (`tests/test_agnostic_stream.py`).
- **Counted before `done`.** Usage is logged before the `done` event is
  sent, so a client that asks for its usage the moment the answer ends sees
  this answer counted.

### 4.4 Stop (MS-667)

- Before the first chunk: unchanged. A stop before the model is called means
  the model is never called.
- During streaming: the browser aborts the fetch, `request.is_disconnected()`
  turns true, the worker is told to stop, tokens so far are logged, nothing
  is saved. The 499 status of MS-667 no longer exists once a 200 has been
  sent, so for a stream the stop is only visible in the log
  (`agnostic_query: stopped by the user`).

### 4.5 Router file

`router/agnostic.py` stays routes-only (per `AGENTS.md`):

```python
@router.post("/agnostic/query", response_model=AgnosticQueryResponse,
             responses={200: {"content": {"application/x-ndjson": {}}}})
async def agnostic_query(req, request, stream: bool = Query(False), user=Depends(...)):
    ...
    return await agnostic_service.answer_query(req, base_url, user, is_disconnected=..., stream=stream)
```

`response_model` is kept so the OpenAPI contract for the plain JSON case does
not change; the NDJSON content type is declared the same way the upload
routes do it.

## 5. UI design (`doculens-ui`)

Placement follows `RULES.md` section 5 (anything that calls the backend lives
under `services/`).

- **Reader.** A new method on `RequestHandler` for a streamed POST
  (`fetch` with `response.body.getReader()`, one JSON object per line). It
  takes the same `signal` as `store`, rejects through `rejectWithError` for
  non-2xx so 401 and 429 behave as today, and calls `onEvent` per parsed line.
  `HybridQueryApi` uses it with `?stream=true`.
- **Event types** (`sources`, `delta`, `done`, `error`) are typed next to
  `HybridResponse`.
- **Typing effect.** Gemini sends the text in pieces of a sentence or more, at
  moments of its own, so showing each piece as it arrives would look like text
  appearing in lumps. The text that has arrived is instead typed out a few
  characters per animation frame (`lib/typewriter.ts`): between 60 and 400
  characters per second (faster the more is waiting, never faster than 400, so
  even a whole answer that arrives at once is seen being typed for a moment,
  while the screen stays close behind the model), and one render
  per frame. When `done` arrives, what is left is typed out at 400 to 1200
  characters per second before the finished answer replaces it; a hidden tab,
  which runs no frames, waits at most 4 s.
- **Answers without a model call appear at once.** An answer that was not
  streamed (a canned reply, the passage shown when the model fails, a backend
  that does not stream) has nothing to type: the finished answer replaces the
  spinner whole, as the ticket says ("appear at once, as today"). Messages the
  page writes itself (the `/help` command, "pick a source first") never reach
  the backend and also appear at once.
- **One waiting state.** From the moment a question is sent until the finished
  message takes over, the answer is one bubble: a spinner, then the sources,
  then the text. There is no separate spinner panel, and the waiting flag goes
  down in the same tick as the finished message, so no frame shows a spinner
  next to the answer.
- **Message state.** A message can be `streaming`. The chat shows the
  spinner until the first `delta`, then the text as it grows with a typing
  cursor at its end (`lib/rehype-streaming-cursor.ts`, styled by
  `.streaming-cursor` in `app/globals.css`), then `done` replaces the text and
  the cursor goes. The source cards appear on `sources`, before any text.
- **Half-finished markdown.** While `streaming`, the text passed to
  `ReactMarkdown` is repaired first: an unclosed code fence is closed, and a
  trailing table row or list marker without content is trimmed so it is not
  rendered as a broken block. This is a pure function with unit tests
  (`node:test`). The repair is not applied to the final text.
- **Layout jumps.** The streaming message keeps a stable container (no
  remounting between `sources`, deltas and `done`), so the bubble grows only downward.
- **Stop and Escape.** `controller.abort()` from `stopGenerating` ends the
  read. The partial text is replaced by the "Stopped" note and nothing is
  saved, as MS-667 does today.
- **Which chat gets the text.** `use-chat-thread.ts` already captures
  `sentThreadKey` and `sentMessages` when the question is sent and keeps
  `inFlightAnswers` per thread. Deltas are written to the thread with that
  key (the live view or the parked snapshot), never to "the chat on screen".
- **Same behavior everywhere.** `runQuery` (first question, follow-up, slash
  command, skill) and the regenerate call both go through the same streaming
  call.
- **Saving.** The conversation is still saved once, when `done` arrives,
  using `done.result` (so the saved answer is the validated one).
- **Instant answers** (decision 5) arrive as a lone `done` and are appended
  right away.

## 6. Test plan

Server (`python -m unittest discover -s tests -t .`, with a fake model that
yields scripted chunks):

- Event order for a normal question, and the exact lines.
- `done.result` equals the non-streaming response for the same inputs.
- 401, 403, 422 and 429 are plain HTTP errors with `stream=true`.
- Canned replies produce a lone `done`.
- A model that fails before the first chunk gives `done` with the fallback; a
  model that fails mid-stream gives `error`.
- Tokens logged once for: completion, stop mid-stream, disconnect before the
  first chunk, error mid-stream. The in-flight counters return to 0 in all of
  them.
- Without `stream=true` the response is byte-identical to today's (compare
  with the MS-657 contract snapshot: `scripts/contract_snapshot.py compare`).

UI (`node:test`): NDJSON line reader (split lines across chunks, a bad line, a
missing terminal event), markdown repair for each broken shape, frame batching.

Manual: the acceptance criteria in MS-906, run against the local server and
against the Space.

## 7. Rollout and risks

- Push the server first; the UI only sends `stream=true` once the endpoint
  exists on both backends. An old backend ignores the unknown query
  parameter and returns plain JSON, so the UI reader must also accept a
  plain JSON response (check `Content-Type`).
- **Proxy buffering on the HF Space.** The two headers are necessary but not
  proof. Check from outside the Space with `curl -N` that `delta` lines
  arrive one by one, and record it below.
- **Single worker.** The workspace lock and in-flight counters are in-process
  (see `services/billing/inflight.py`), which stays true with streaming.
  Streams hold a reservation longer than before, so two long answers in one
  workspace count against the quota for their whole duration.
- **Efficient mode and summaries** use the same prompt path, so both stream.
  Summaries are the longest answers and the best test of the batching.
- **Mirror.** Code changes only in this repo; run
  `python scripts/sync_hf.py --write` then `--check`.

## 8. Measurements

"Time to first word" is the time from sending the request until the first
`delta` reaches the client (after the change) or until the answer arrives
(before the change, because the answer arrives in one piece).

Reference from MS-657 (`perf/README.md`, 9 Oct 2026, Windows 11, Gemini, local
database), the whole answer arriving at once:

| Path | median |
|---|---|
| `/agnostic/query` warm | 5.3 s (5455 ms before the pool change, 5293 ms after) |
| `/agnostic/query` cold | 7.6 s |

Measured on 21 Oct 2026, macOS, real Gemini, at the model layer (`llm.invoke`
for before, the real `ModelStream` for after; the script feeds both the same
prompt, one after the other). Retrieval and the router call are not included
(about 60 ms in MS-657), so "time to first `sources`" was not measured here.
Two models were used: first `gemini-3.1-flash-lite` (the app's model had run
out of its daily free quota), then the app's own model `gemini-3.8-flash`
(`gemini-flash-latest`) after the key was replaced. The first runs set a 60 s
timeout and one retry on each call; the short question was then run again
with a 150 s timeout and no retry, so no hidden retry is inside its numbers.

`gemini-3.8-flash` (the app's model), one run per row:

| Question | Before: time to the full answer | After: time to first `delta` | After: time to `done` | After: chunks |
|---|---|---|---|---|
| Short factual question (162 chars) | 26.9 s | 64.0 s | 64.1 s | 2 |
| Short factual question, again with no retry (169 chars) | 70.8 s | 55.1 s | 55.2 s | 2 (119 tokens) |
| Long answer (about 2600 to 2900 chars) | 55.2 s | 30.2 s | 34.2 s | 25 (median gap 152 ms, 959 tokens) |

`gemini-3.1-flash-lite`, two runs per cell:

| Question | Before: time to the full answer | After: time to first `delta` | After: time to `done` |
|---|---|---|---|
| Short factual question (132 chars, 3 chunks) | 12.9 s, 12.7 s | 8.8 s, 11.5 s | 8.9 s, 11.6 s |
| Long answer (about 2100 chars, 20 to 21 chunks) | 13.0 s, 6.3 s | 11.8 s, 13.9 s | 12.9 s, 15.1 s |

What this shows, and what it does not:

- The mechanism works against the real models: the long answers arrived in
  20 to 25 pieces, with a median gap of 43 to 152 ms between them, and the
  tokens were counted (830 and 959 for the long ones).
- The wait is the model's, not ours. `invoke` was as slow as the streamed call
  on every question (for example 26.9 s by `invoke` for a 162 character
  answer, 70.8 s on the rerun), and the delay is before the first piece,
  while the server is still waiting for Gemini. What `ModelStream` itself
  adds was measured with a fake model that sends 25 pieces 150 ms apart (4
  runs, 100 pieces): from the model handing a piece over to the event loop
  receiving it took 0.29 ms (median), 0.51 ms at most.
- What streaming gives is the head start between the first piece and the end
  of the answer: about 1 s on `flash-lite`, and 4.0 s on `gemini-3.8-flash`
  for the long answer (30.2 s to 34.2 s). That is the part of the wait the
  reader can already spend reading. It depends on how the model sends its
  text, so it will differ between models and between answers.
- The variance is very large: the same long question took 6.3 s and 13.0 s by
  `invoke` on `flash-lite`, and on `gemini-3.8-flash` the short question took
  26.9 s and then 70.8 s by `invoke`. The 64 s of the first run was first put
  down to the script's 60 s timeout and retry, but the rerun with no retry was
  as slow (55.1 s to the first piece), so it is the API's latency on the free
  tier that day. "Before" and "after" are separate requests, so
  a before/after difference of tens of seconds (55.2 s against 34.2 s) is the
  API's latency, not the change. Compare only within one call: first piece to
  `done`.
- All of it is slower than the 5.3 s reference from MS-657, and it is one run
  per question on the app's model. Treat it as an indication, not a median.

Space check with `curl -N` (command and a few lines of output, with their
arrival times):

```
(to fill in)
```

## 9. UI check (Browser pane, stub API)

Run against a stub API that streams scripted NDJSON (no Gemini). Checked in
the real browser: spinner, then the sources with a spinner row, then the text
growing with the typing cursor at its end, then `done` (cursor gone, actions shown); the thread follows the text while the reader is at the bottom;
a table never shows as raw pipes while it is written; a stream cut after text
shows a toast and drops the partial text; `done` replaces the streamed text;
Escape and the Stop button close the connection (the stub saw the client
leave) and leave the "Stopped" note; Regenerate writes in place of the old
answer; starting a new chat while an answer streams leaves the new chat
clean; one save per answer, made after `done`, none while streaming; a
backend that answers with plain JSON still works.

Not checked: smoothness at a real frame rate (the Browser pane throttles
animation frames to about 3 per second, so batching is covered by unit
tests only), and coming back to a chat while its answer is still being
written (the answer is saved into its own conversation, but the view that was
reopened meanwhile did not refresh; the code that does this is unchanged by
MS-906, but it was not compared with a run of the old code).

