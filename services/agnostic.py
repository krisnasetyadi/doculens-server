"""
services/agnostic.py
--------------------
The question-answering path behind POST /api/v1/agnostic/query: picks the
sources this user may search, routes the message (canned reply, conversation
router or RAG), meters tokens and maps retrieved documents to the response.
router/agnostic.py calls answer_query. With ?stream=true the same answer goes
out as NDJSON events while the model writes it (MS-906, docs/answer-streaming.md).
"""

import asyncio
import json
import logging
import re
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterator, List, Optional, Tuple, Union

import anyio
from fastapi import HTTPException, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from config import config
from services.rag.conversation.guard import is_internal_question
from services.rag.conversation.intent import (
    is_app_question,
    is_summary_request,
    is_whole_document_summary,
    mentions_document_part,
    rule_based_intent,
    scope_titles_for,
    small_talk_kind,
)
from services.rag.conversation.language import ReplyLanguage, detect_reply_language
from services.rag.answer import AnswerPlan
from services.rag.conversation.messages import internal_refusal, no_source_selected
from services.rag.model_stream import ModelStream
from services.rag.processor import processor
from services.storage import supabase as supabase_storage
from models import AgnosticQueryRequest, AgnosticQueryResponse, MemoryTurn, PdfSourceDetail, UserRecord
from services.public_links import resolve_active_public_link_sources
from services.database_connections import resolve_active_database_connections
from services.billing.allocation import get_usage_snapshot
from services.billing.enforcement import enforce_member_allocation, enforce_plan_limit
from services.billing.inflight import (
    InFlightReservation, begin_in_flight, end_in_flight, get_workspace_lock, in_flight_tokens, resolve_workspace_id,
)
from services.billing.ledger import enforce_rate_limit, log_token_usage

logger = logging.getLogger(__name__)

# MS-237 poin 1: hard cap on the client-sent conversation-history window,
# applied here regardless of what the client actually sends — a client that
# sends 200 messages must not be able to balloon the prompt/token cost. The
# window is counted in *chats* (one question plus the answer that followed
# it), so 5 chats is up to ~10 messages; MAX_MEMORY_MESSAGES is the backstop
# for a caller that pads one chat with a hundred assistant turns.
MAX_MEMORY_CHATS = 5
MAX_MEMORY_MESSAGES = MAX_MEMORY_CHATS * 4
MAX_MEMORY_CHARS = 600


def _describe_source_type(has_public_links: bool, has_external_db: bool) -> str:
    extras = []
    if has_public_links:
        extras.append("Realtime Public Links")
    if has_external_db:
        extras.append("Realtime Database Connections")
    if not extras:
        return "Indexed Collections"
    return "Indexed Collections + " + " + ".join(extras)


def _clamp_memory(memory: Optional[List[MemoryTurn]]) -> Optional[List[Dict[str, str]]]:
    """MS-237: clamp the client's conversation-history window before it
    ever reaches a prompt — see MAX_MEMORY_* above. Cut at the start of the
    5th-from-last question rather than at a raw message count, so every
    remembered question keeps the answer that belongs to it."""
    if not memory:
        return None
    question_at = [i for i, m in enumerate(memory) if m.role == "user"]
    start = question_at[-MAX_MEMORY_CHATS] if len(question_at) > MAX_MEMORY_CHATS else 0
    return [
        {"role": m.role, "content": _clip_turn(m.content)}
        for m in memory[start:][-MAX_MEMORY_MESSAGES:]
    ]


def _clip_turn(content: str) -> str:
    """Keep the START and the END of a long turn: an assistant reply's
    closing question ("Mau saya ringkas X?") is exactly what the router
    needs to resolve a following "ok", and a head-only cut dropped it."""
    if len(content) <= MAX_MEMORY_CHARS:
        return content
    half = (MAX_MEMORY_CHARS - 3) // 2
    return f"{content[:half]} … {content[-half:]}"


def _system_response(
    answer: str,
    model_used: str,
    question: str,
    start_time: datetime,
    source_type: str = "System",
) -> AgnosticQueryResponse:
    """A reply that involved no retrieval (system answers, refusals,
    conversation mode) — no sources, nothing retrieved."""
    return AgnosticQueryResponse(
        answer=answer,
        model_used=model_used,
        pdf_sources=[],
        pdf_sources_detailed=[],
        db_results={},
        chat_results=[],
        processing_time=(datetime.now() - start_time).total_seconds(),
        search_terms=[question],
        target_tables=[],
        source_type=source_type,
        retrieved_count=0,
    )


async def _log_tokens(user_id: str, tokens: int) -> None:
    """Best-effort token metering (MS-248) for the conversation router's
    LLM call: same never-break-the-reply rule as _log_answer_tokens."""
    if not tokens:
        return
    try:
        await asyncio.to_thread(log_token_usage, user_id, tokens)
    except Exception:
        logger.warning("agnostic_query: failed to log token usage", exc_info=True)


def _strip_titles(text: str, titles: List[str]) -> str:
    """`text` without the collection titles it names — see
    conversation.intent.is_whole_document_summary for why."""
    for title in sorted((t for t in titles if t), key=len, reverse=True):
        text = re.sub(re.escape(title), " ", text, flags=re.IGNORECASE)
    return text


def _collection_ids_for(titles: List[str], collections: List[Tuple[str, str]]) -> List[str]:
    """Collection ids whose title matches one of `titles` (case-insensitive).
    Titles aren't unique, so one title can map to several collections."""
    wanted = {t.strip().lower() for t in titles if t}
    return [cid for cid, title in collections if title.strip().lower() in wanted]


# MS-667: the user can stop a question while it is being answered; the
# browser then aborts the request. "Client closed request": nobody is
# listening any more, so the response body is empty.
STOPPED_STATUS = 499


class QueryStopped(Exception):
    """The user stopped the question. Not an error: nothing is logged as one."""


class StopCheck:
    """Asks the request whether the user has stopped the question. Asked right
    before and right after each model call rather than polled on a timer:
    retrieval can take less than any polling interval, and a stop in that
    window must still keep the model from being called."""

    def __init__(self, is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None):
        self._is_disconnected = is_disconnected
        self._stopped = False

    async def stopped(self) -> bool:
        if not self._stopped and self._is_disconnected is not None:
            self._stopped = await self._is_disconnected()
        return self._stopped


class _Reservation:
    """The in-flight token reservation of one question. `end_in_flight`
    subtracts, so ending one twice would free tokens that another question
    reserved: this ends it once, whoever asks and however often."""

    def __init__(self, reservation: InFlightReservation):
        self._reservation = reservation
        self._released = False
        self._handed_over = False

    def hand_over(self) -> None:
        """The answer now owns the reservation and releases it itself."""
        self._handed_over = True

    def release(self) -> None:
        if not self._released:
            self._released = True
            end_in_flight(self._reservation)

    def release_unless_handed_over(self) -> None:
        if not self._handed_over:
            self.release()


@dataclass
class PendingAnswer:
    """A question that passed every check and has its sources: only the model
    call is left. Owns the question's in-flight reservation, so whoever
    answers it must release it, after the answer's cost is on the ledger.

    `question` is what gets answered: req.question, or the router's search
    query. `routing_tokens` is the router call that sent the message here."""
    req: AgnosticQueryRequest
    user: UserRecord
    base_url: str
    start_time: datetime
    hybrid_results: Dict[str, Any]
    question: str
    language: Optional[ReplyLanguage]
    memory: Optional[List[Dict[str, str]]]
    skill_instruction: Optional[str]
    task: str
    routing_tokens: int
    source_type: str
    stop: StopCheck
    reservation: _Reservation

    def __post_init__(self) -> None:
        self.reservation.hand_over()

    def generate_args(self) -> tuple:
        """The arguments of processor.generate_hybrid_answer / plan_hybrid_answer."""
        return (
            self.hybrid_results, self.question, self.req.llm_provider, self.req.llm_model, self.memory,
            self.skill_instruction, bool(self.req.efficient_mode), self.language, self.task,
        )


@contextmanager
def _failures_as_500() -> Iterator[None]:
    """Whatever goes wrong unexpectedly is logged and answered as a plain 500
    (the detail never carries the cause)."""
    try:
        yield
    except (HTTPException, QueryStopped):
        raise
    except Exception as e:
        logger.exception("agnostic_query failed")
        raise HTTPException(status_code=500, detail="Failed to answer the question") from e


async def answer_query(
    req: AgnosticQueryRequest,
    base_url: str,
    user: UserRecord,
    is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    stream: bool = False,
) -> Union[AgnosticQueryResponse, StreamingResponse, Response]:
    """Everything behind POST /agnostic/query; `base_url` builds the file links in the sources.

    `is_disconnected` (the request's) is polled while the question runs. Once
    the user has stopped it, the model is not called any more; a model call
    already running is left to finish so its tokens are counted, then no
    answer is sent (MS-667).

    With `stream` the answer is NDJSON events written as the model writes it
    (MS-906); see docs/answer-streaming.md. Every check, and the search for
    sources, still runs before the first byte, so a refusal (401, 403, 422,
    429) is an ordinary HTTP error and not an event."""
    try:
        outcome = await _answer_query(req, base_url, user, StopCheck(is_disconnected))
        if stream:
            return _stream_response(outcome)
        return await _answer_in_full(outcome)
    except QueryStopped:
        logger.info("agnostic_query: stopped by the user (user=%s)", user.user_id)
        return Response(status_code=STOPPED_STATUS)


async def _answer_query(
    req: AgnosticQueryRequest, base_url: str, user: UserRecord, stop: StopCheck,
) -> Union[AgnosticQueryResponse, PendingAnswer]:
    """Run every check and the search for sources. The result is the finished
    response when no model call is needed (canned replies, router replies), or
    a PendingAnswer when one is."""
    start_time = datetime.now()
    logger.info("agnostic_query: question=%r user=%s", req.question, user.user_id)
    is_admin = user.role == "admin"
    with _failures_as_500():
        # Resolve collections — scoped to what this user is allowed to see.
        # Admins can reach every PDF collection; everyone else only their own.
        # Deactivated collections are left out entirely: not searched, not
        # summarized, not named to the router (MS-665). One query returns both
        # the ids and their display titles (which the router needs to
        # name/scope collections), filtered in SQL.
        collections = await asyncio.to_thread(
            supabase_storage.list_active_collection_titles_for_user, user.user_id, is_admin
        )
        allowed_pdf_ids = {cid for cid, _ in collections}
        titles = [title for _, title in collections]
        if req.pdf_collection_ids:
            pdf_collection_ids = [cid for cid in req.pdf_collection_ids if cid in allowed_pdf_ids]
        elif req.include_pdf_results:
            pdf_collection_ids = list(allowed_pdf_ids)
        else:
            pdf_collection_ids = []
        logger.info("agnostic_query: %d PDF collection(s) accessible to user", len(pdf_collection_ids))

        memory_payload = _clamp_memory(req.memory)
        # Reply language (en/id) for every branch below — from the question,
        # then earlier user turns, then English. Never hardcoded Indonesian.
        language = detect_reply_language(req.question, memory_payload)

        # "/help" — deterministic capability summary built from real state
        # (this user's own collections), never LLM-generated, free.
        if req.question.strip().lower() == "/help":
            return _system_response(
                processor.build_meta_help_answer(len(allowed_pdf_ids), titles, language),
                "system/meta-help", req.question, start_time,
            )

        # Unrecognized "/" command — safety net for requests that bypass the
        # chat-ui command menu (which already intercepts unmatched slash
        # input client-side). Short-circuits before hybrid_search/
        # generate_hybrid_answer: a mistyped command has no document to
        # ground an LLM answer in.
        if processor.is_unknown_slash_command(req.question):
            return _system_response(
                processor.build_unknown_command_answer(req.question, language),
                "system/unknown-command", req.question, start_time,
            )

        # Questions about how DocuLens itself is built (tech stack, model,
        # infra, prompts) or prompt-injection attempts — canned refusal,
        # before any LLM sees the message. See services/rag/conversation/guard.py.
        if is_internal_question(req.question):
            return _system_response(
                internal_refusal(language), "system/guard", req.question, start_time,
            )

        # Two paths (services/rag/conversation/intent.py):
        # - "retrieval" by rule (the message points at the user's own data,
        #   or a Skill is armed) → straight to the existing RAG pipeline, no
        #   routing call. Scoped to any collection the message names.
        # - everything else → the conversation router (processor.route_message):
        #   ONE LLM call that either replies, or calls search_documents with
        #   structured args (query / collection / task) that then run through
        #   the same RAG pipeline. Small talk goes there too, so "ok" after an
        #   offer becomes a real, scoped search instead of a pleasantry.
        # The PDF toggle only counts with at least one active collection
        # behind it — with every file deactivated it has nothing to search.
        has_source = any([
            req.include_pdf_results and bool(pdf_collection_ids), req.include_db_results,
            req.include_chat_results, req.include_public_links,
            req.include_external_db,
        ])
        intent = "retrieval" if req.skill_id else rule_based_intent(req.question, titles)

        # No source selected for a data question — safety net for requests
        # that bypass the chat-ui toggle guard (regenerate, direct API
        # calls). Without this, hybrid_search silently returns empty results
        # and the LLM produces a misleading "not found in documents" answer.
        if intent == "retrieval" and not has_source:
            return _system_response(
                no_source_selected(language), "system/no-source-selected", req.question, start_time,
            )

        # Past this point every branch may call the LLM, so this is where
        # rate limiting (MS-248) has to sit.
        #
        # The workspace lock (keyed by workspace, not just this user —
        # enforce_plan_limit/enforce_member_allocation check state shared
        # across the whole team) is held only for the CHECK. Passing it
        # registers this request's in-flight reservation, which every later
        # check counts as already spent, so the LLM call itself can run
        # outside the lock in parallel with other members' — see
        # services.billing.inflight.begin_in_flight for why this replaced holding the lock
        # through the whole call.
        workspace_id = await asyncio.to_thread(resolve_workspace_id, user)
        async with get_workspace_lock(workspace_id):
            pending_workspace, pending_user = in_flight_tokens(workspace_id, user.user_id)
            try:
                await asyncio.to_thread(enforce_rate_limit, user.user_id, pending_user)
                workspace_cap = await asyncio.to_thread(enforce_plan_limit, user, pending_workspace)
                user_cap = await asyncio.to_thread(enforce_member_allocation, user, pending_user)
            except HTTPException:
                # A capped user still gets a (free, canned) reply to "hi" or
                # "what can DocuLens do?" — help must stay reachable even
                # when the quota is spent. Data questions stay blocked,
                # except that with no source on they'd only ever have been
                # told to pick one, which costs nothing either.
                if intent != "retrieval" and not has_source and not (
                    small_talk_kind(req.question) or is_app_question(req.question)
                ):
                    return _system_response(
                        no_source_selected(language), "system/no-source-selected", req.question, start_time,
                    )
                if intent == "retrieval" or not (small_talk_kind(req.question) or is_app_question(req.question)):
                    raise
                return _system_response(
                    processor.build_conversation_fallback(req.question, language, len(allowed_pdf_ids), titles),
                    "system/conversation", req.question, start_time, source_type="Conversation",
                )
            reservation = _Reservation(begin_in_flight(
                workspace_id, user.user_id, config.query_in_flight_estimate, workspace_cap, user_cap
            ))

        try:
            if intent == "retrieval":
                scoped_titles = scope_titles_for(req.question, titles)
                return await _retrieve(
                    req, user, is_admin, allowed_pdf_ids, pdf_collection_ids, base_url, start_time,
                    memory_payload, language,
                    question=req.question,
                    scope_collection_ids=_collection_ids_for(scoped_titles, collections),
                    task="summarize" if scoped_titles and is_whole_document_summary(req.question, scoped_titles) else "answer",
                    stop=stop,
                    reservation=reservation,
                )

            usage = await asyncio.to_thread(get_usage_snapshot, user)
            if await stop.stopped():
                raise QueryStopped
            decision = await asyncio.to_thread(
                processor.route_message,
                req.question, language, memory_payload, is_admin, titles, len(allowed_pdf_ids),
                usage, req.llm_provider, req.llm_model, has_source,
            )
            if await stop.stopped():
                # The router call already ran: count it, answer nothing.
                await _log_tokens(user.user_id, decision.total_tokens)
                raise QueryStopped

            if decision.action == "reply":
                await _log_tokens(user.user_id, decision.total_tokens)
                return _system_response(
                    decision.answer, decision.model_id, req.question, start_time, source_type="Conversation",
                )

            if not has_source:
                await _log_tokens(user.user_id, decision.total_tokens)
                return _system_response(
                    no_source_selected(language), "system/no-source-selected", req.question, start_time,
                )

            scoped_titles = [decision.search_collection] if decision.search_collection else []
            # The model picks the task, but a summary of one PART ("ringkas
            # pasal 5", "summarize the access control section") must still be
            # a search for that part — never the whole-document sample.
            task = decision.search_task
            if task == "summarize" and not (
                is_whole_document_summary(req.question, titles) or not is_summary_request(req.question)
            ) or task == "summarize" and mentions_document_part(
                _strip_titles(decision.search_query, titles), include_topics=False
            ):
                task = "answer"
            return await _retrieve(
                req, user, is_admin, allowed_pdf_ids, pdf_collection_ids, base_url, start_time,
                memory_payload, language,
                routing_tokens=decision.total_tokens,
                question=decision.search_query,
                scope_collection_ids=_collection_ids_for(scoped_titles, collections),
                task=task,
                stop=stop,
                reservation=reservation,
            )
        finally:
            # A reply that needed no model call has logged its usage before
            # returning, so the reservation only drops once the real cost is on
            # the ledger. A PendingAnswer keeps it: its answer releases it
            # after logging what the model call cost.
            reservation.release_unless_handed_over()


async def _retrieve(
    req: AgnosticQueryRequest,
    user: UserRecord,
    is_admin: bool,
    allowed_pdf_ids: set,
    pdf_collection_ids: List[str],
    base_url: str,
    start_time: datetime,
    memory_payload: Optional[List[Dict[str, str]]],
    language: Optional[ReplyLanguage],
    stop: StopCheck,
    reservation: _Reservation,
    routing_tokens: int = 0,
    question: Optional[str] = None,
    scope_collection_ids: Optional[List[str]] = None,
    task: str = "answer",
) -> PendingAnswer:
    """The token-costing half of agnostic_query up to the model call:
    source resolution and the search, run while the question holds its
    in-flight reservation. Split out from agnostic_query so that
    reservation's scope is a plain, visible function call rather than a large
    indented block sharing the outer try/except.

    `question` is what gets searched/answered — req.question, or the
    router's search_documents query. `scope_collection_ids` narrows the PDF
    search to collections the user named (ignored when it doesn't overlap
    what they may see); `task="summarize"` on a scoped search reads chunks
    spread across those collections instead of the top similarity hits.
    `routing_tokens` is the router call that sent the message here.
    `stop` (MS-667): a question stopped before the model call goes no
    further. The PendingAnswer that comes back takes over the reservation."""
    question = question or req.question
    scoped_ids = [cid for cid in (scope_collection_ids or []) if cid in allowed_pdf_ids]
    if scoped_ids and req.include_pdf_results:
        pdf_collection_ids = scoped_ids
    # Chat is admin-only (business rule — non-admins never search it,
    # regardless of what flags/ids the client sends).
    chat_collection_ids = req.chat_collection_ids if is_admin else []
    if is_admin and not chat_collection_ids and req.include_chat_results:
        chat_collection_ids = processor.get_all_chat_collections()

    should_search_pdfs = bool(req.include_pdf_results) and bool(pdf_collection_ids)
    should_search_db   = bool(req.include_db_results)
    should_search_chat = is_admin and bool(req.include_chat_results) and bool(chat_collection_ids)
    public_link_sources: List[Dict[str, Any]] = []
    if bool(req.include_public_links):
        public_link_sources = await resolve_active_public_link_sources(
            req.public_link_ids, user_id=user.user_id, is_admin=is_admin
        )
    should_search_public_links = bool(req.include_public_links) and bool(public_link_sources)

    # Database connections are admin-only (business rule), same as chat.
    external_db_connections: List[Dict[str, Any]] = []
    if is_admin and bool(req.include_external_db):
        external_db_connections = await resolve_active_database_connections(req.external_db_connection_ids)
    should_search_external_db = is_admin and bool(req.include_external_db) and bool(external_db_connections)

    if task == "summarize" and scoped_ids and should_search_pdfs:
        hybrid_results = await asyncio.to_thread(processor.build_summary_results, pdf_collection_ids)
    else:
        # Run hybrid search against pre-built FAISS indexes
        hybrid_results = await asyncio.to_thread(
            processor.hybrid_search,
            question,
            pdf_collection_ids or [],
            should_search_chat,
            should_search_pdfs,
            should_search_db,
            chat_collection_ids or [],
            should_search_public_links,
            public_link_sources,
            should_search_external_db,
            external_db_connections,
        )

    # MS-252: resolve the one-shot skill (if any) to its instruction text.
    # get_skill_for_user already enforces personal/team visibility — an id
    # that doesn't exist or isn't visible to this user just yields None, so
    # a deleted/foreign skill_id silently falls back to normal chat instead
    # of erroring the whole request.
    skill_instruction: Optional[str] = None
    if req.skill_id:
        skill_row = await asyncio.to_thread(
            supabase_storage.get_skill_for_user, req.skill_id, user.user_id, is_admin
        )
        if skill_row:
            skill_instruction = skill_row["instruction"]
        else:
            logger.warning(
                "agnostic_query: skill_id=%s not visible to user=%s — ignoring",
                req.skill_id, user.user_id,
            )

    if await stop.stopped():
        # Stopped while the sources were being retrieved: the model is never
        # called, so only a router call that brought the message here costs.
        await _log_tokens(user.user_id, routing_tokens)
        raise QueryStopped

    return PendingAnswer(
        req=req, user=user, base_url=base_url, start_time=start_time,
        hybrid_results=hybrid_results, question=question, language=language,
        memory=memory_payload, skill_instruction=skill_instruction, task=task,
        routing_tokens=routing_tokens,
        source_type=_describe_source_type(should_search_public_links, should_search_external_db),
        stop=stop, reservation=reservation,
    )


async def _answer_in_full(outcome: Union[AgnosticQueryResponse, PendingAnswer]) -> AgnosticQueryResponse:
    """The whole answer as one JSON response (no `stream`)."""
    if isinstance(outcome, AgnosticQueryResponse):
        return outcome
    with _failures_as_500():
        try:
            return await _generate_in_full(outcome)
        finally:
            outcome.reservation.release()


async def _generate_in_full(pending: PendingAnswer) -> AgnosticQueryResponse:
    """The model call in one piece, usage logging and the response.
    A question stopped during the model call has its tokens logged and no
    response built (MS-667)."""
    answer_result = await asyncio.to_thread(processor.generate_hybrid_answer, *pending.generate_args())

    if isinstance(answer_result, tuple) and len(answer_result) >= 2:
        answer, model_used = answer_result[0], answer_result[1]
        answer_metadata = (
            answer_result[2] if len(answer_result) >= 3 and isinstance(answer_result[2], dict) else {}
        )
    else:
        answer     = str(answer_result)
        model_used = pending.req.llm_model or config.default_llm_model
        answer_metadata = {}

    efficiency = answer_metadata.get("efficiency")
    # routing_tokens: the router call that sent this message here (0 when
    # the rules decided), metered on the same row.
    await _log_answer_tokens(
        pending.user.user_id, answer_metadata.get("total_tokens", 0) + pending.routing_tokens, efficiency,
    )

    # Stopped while the model was answering: its tokens are on the ledger
    # above; the answer itself goes nowhere.
    if await pending.stop.stopped():
        raise QueryStopped

    return _build_response(pending, answer, model_used, efficiency)


async def _log_answer_tokens(user_id: str, tokens: int, efficiency: Optional[Dict[str, Any]]) -> None:
    """Best-effort token metering (MS-248). Never let a logging hiccup
    break a chat response that already succeeded. Run in a thread: this
    does blocking DB I/O, and a blocking call here would stall every other
    coroutine on the event loop."""
    efficient_mode_on = bool(efficiency and efficiency.get("enabled"))
    try:
        # Local/free models (e.g. HuggingFace) always report 0 tokens —
        # still worth a row when Efficient Mode was on, so the dashboard
        # (GET /payments/efficient-mode/stats) isn't silently blind to
        # that whole usage class. log_token_usage's own tokens<=0 guard
        # is relaxed the same way for this same reason.
        if tokens or efficient_mode_on:
            await asyncio.to_thread(
                log_token_usage,
                user_id,
                tokens,
                efficient_mode_on,
                efficiency.get("raw_tokens_est") if efficiency else None,
                efficiency.get("final_tokens_est") if efficiency else None,
            )
    except Exception:
        logger.warning("agnostic_query: failed to log token usage", exc_info=True)


def _build_response(
    pending: PendingAnswer, answer: str, model_used: str, efficiency: Optional[Dict[str, Any]],
    mapped_sources: Optional[Tuple[List[str], List[PdfSourceDetail], List[Dict[str, Any]]]] = None,
) -> AgnosticQueryResponse:
    """`mapped_sources` is `_map_sources` of this answer when the caller already has it."""
    pdf_sources, pdf_sources_detailed, chat_results = mapped_sources or _map_sources(
        pending.hybrid_results, pending.base_url, pending.language,
    )
    return AgnosticQueryResponse(
        answer=answer,
        model_used=model_used,
        pdf_sources=pdf_sources,
        pdf_sources_detailed=pdf_sources_detailed,
        db_results=pending.hybrid_results.get("database_results", {}),
        chat_results=chat_results,
        processing_time=(datetime.now() - pending.start_time).total_seconds(),
        search_terms=pending.hybrid_results.get("search_terms", [pending.question]),
        target_tables=pending.hybrid_results.get("target_tables", []),
        source_type=pending.source_type,
        retrieved_count=len(pdf_sources) + len(chat_results),
        efficiency=efficiency,
    )


def _map_sources(
    hybrid_results: Dict[str, Any], base_url: str, language: Optional[ReplyLanguage],
) -> Tuple[List[str], List[PdfSourceDetail], List[Dict[str, Any]]]:
    """The retrieved documents as the response's source fields."""
    # Map PDF docs -> response fields
    pdf_sources: List[str] = []
    pdf_sources_detailed: List[PdfSourceDetail] = []
    for doc in hybrid_results.get("pdf_documents", []):
        meta  = getattr(doc, "metadata", {})
        fname = meta.get("source", "Unknown")
        page  = meta.get("page")
        page_label = "Page" if language == "en" else "Halaman"
        pdf_sources.append(f"{fname} ({page_label} {page})" if page else fname)

        try:
            page_num = int(page) if page is not None else None
        except (ValueError, TypeError):
            page_num = None

        collection_id = meta.get("collection_id", "")
        file_url = f"{base_url}/api/v1/files/{collection_id}/{fname}" if collection_id else None
        page_url = f"{file_url}#page={page_num}" if file_url and page_num else file_url
        content_text = doc.page_content.strip() if hasattr(doc, "page_content") else ""

        pdf_sources_detailed.append(PdfSourceDetail(
            file_name=fname,
            collection_id=collection_id,
            page=page_num,
            relevance_score=meta.get("similarity_score", 0.0),
            content_preview=(doc.page_content[:300]
                             if hasattr(doc, "page_content") else ""),
            file_url=file_url,
            page_url=page_url,
            search_text=' '.join(content_text.split()[:15]),
        ))

    for doc in hybrid_results.get("public_link_documents", []):
        meta = getattr(doc, "metadata", {})
        fname = meta.get("source", "Unknown")
        title = meta.get("public_link_title")
        display_name = f"{title} / {fname}" if title else fname
        pdf_sources.append(display_name)
        item_url = meta.get("item_url")
        content_text = doc.page_content.strip() if hasattr(doc, "page_content") else ""
        pdf_sources_detailed.append(PdfSourceDetail(
            file_name=display_name,
            collection_id=meta.get("collection_id", "public-link"),
            relevance_score=meta.get("similarity_score", 0.0),
            content_preview=(doc.page_content[:300] if hasattr(doc, "page_content") else ""),
            file_url=item_url,
            page_url=item_url,
            search_text=' '.join(content_text.split()[:15]),
        ))

    for doc in hybrid_results.get("external_db_documents", []):
        meta = getattr(doc, "metadata", {})
        table_name = meta.get("source", "Unknown")
        label = meta.get("external_db_label", "Database")
        display_name = f"{label} / {table_name}"
        pdf_sources.append(display_name)
        pdf_sources_detailed.append(PdfSourceDetail(
            file_name=display_name,
            collection_id=meta.get("collection_id", "external-db"),
            relevance_score=meta.get("similarity_score", 0.0),
            content_preview=(doc.page_content[:300] if hasattr(doc, "page_content") else ""),
        ))

    chat_results = []
    for doc in hybrid_results.get("chat_documents", []):
        meta = getattr(doc, "metadata", {})
        chat_results.append({
            "source":          meta.get("source", "Unknown"),
            "platform":        meta.get("platform", "chat"),
            "relevance_score": meta.get("similarity_score", 0),
            "content_preview": (doc.page_content[:200]
                                if hasattr(doc, "page_content") else ""),
        })

    return pdf_sources, pdf_sources_detailed, chat_results


# ---------------------------------------------------------------------------
# Streaming (MS-906)
# ---------------------------------------------------------------------------

# Without these a proxy (the HF Space's) may hold the response back until it
# ends, and the answer would arrive in one lump after all.
STREAM_MEDIA_TYPE = "application/x-ndjson"
STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

# A model call that did not stop in time keeps its question's reservation until
# it ends, so it stays counted while it runs. This is the most it waits.
LATE_SETTLE_SECONDS = 120.0
_late_settlements: set = set()  # strong references: a task nobody holds can be collected mid-run


def _event(kind: str, **fields: Any) -> str:
    """One NDJSON line: {"type": kind, ...fields}."""
    return json.dumps(jsonable_encoder({"type": kind, **fields}), ensure_ascii=False) + "\n"


def _stream_response(outcome: Union[AgnosticQueryResponse, PendingAnswer]) -> StreamingResponse:
    """The NDJSON response for an outcome of _answer_query. An answer that
    needs no model call (canned or router reply) is a single `done` event."""
    if isinstance(outcome, AgnosticQueryResponse):
        return StreamingResponse(_finished_events(outcome), media_type=STREAM_MEDIA_TYPE, headers=STREAM_HEADERS)
    return _StreamedAnswer(_AnswerStream(outcome))


async def _finished_events(response: AgnosticQueryResponse) -> AsyncIterator[str]:
    yield _event("done", result=response)


class _AnswerStream:
    """The events of one streamed answer: `sources` as soon as the search is
    done, a `delta` per piece of text the model writes, then `done` with the
    same body the non-streamed endpoint returns (the text is validated once it
    is complete, so `done` can replace what was streamed). If the model fails
    after text went out: `error`.

    However the stream ends (done, error, or the client leaving) `settle` runs
    exactly once: the tokens used so far are logged and the reservation freed."""

    def __init__(self, pending: PendingAnswer):
        self._pending = pending
        self._plan: Optional[AnswerPlan] = None
        self._model: Optional[ModelStream] = None
        self._settled = False
        # Backstop for an answer that is dropped without ever being run (the
        # response is returned but the server fails before calling it): the
        # reservation would otherwise stay counted until restart. Releasing is
        # idempotent, so this is harmless after a normal end.
        weakref.finalize(self, pending.reservation.release)

    async def events(self) -> AsyncIterator[str]:
        pending = self._pending
        try:
            mapped_sources = _map_sources(pending.hybrid_results, pending.base_url, pending.language)
            pdf_sources, pdf_sources_detailed, chat_results = mapped_sources
            yield _event(
                "sources",
                pdf_sources=pdf_sources,
                pdf_sources_detailed=pdf_sources_detailed,
                chat_results=chat_results,
                source_type=pending.source_type,
                retrieved_count=len(pdf_sources) + len(chat_results),
            )

            plan = self._plan = await asyncio.to_thread(processor.plan_hybrid_answer, *pending.generate_args())
            if plan.ready is not None:
                answer, model_used, metadata = plan.ready
            else:
                model = self._model = ModelStream(plan.llm, plan.prompt)
                written: List[str] = []
                try:
                    async for text in model.chunks():
                        written.append(text)
                        yield _event("delta", text=text)
                except Exception as error:
                    if written:
                        raise  # the user has already read part of it: no quiet replacement
                    # Same as the non-streamed endpoint: nothing sent yet, so
                    # the best retrieved passage stands in for the answer.
                    answer, model_used, metadata = await asyncio.to_thread(
                        processor.fallback_hybrid_answer, plan, error,
                    )
                else:
                    # The model finished. A failure in validating its text takes
                    # the same fallback as the non-streamed endpoint (done.result
                    # must match it), and `done` replaces the text already shown.
                    try:
                        answer, model_used, metadata = await asyncio.to_thread(
                            processor.finish_hybrid_answer, plan, "".join(written), model.total_tokens,
                        )
                    except Exception as error:
                        answer, model_used, metadata = await asyncio.to_thread(
                            processor.fallback_hybrid_answer, plan, error,
                        )

            # Before `done`, so a client that asks for its usage the moment the
            # answer ends already sees this one counted.
            await self.settle()
            yield _event(
                "done",
                result=_build_response(pending, answer, model_used, metadata.get("efficiency"), mapped_sources),
            )
        except Exception:
            logger.exception("agnostic_query: streamed answer failed")
            yield _event("error", detail="Failed to answer the question")
        finally:
            await self.settle()

    async def settle(self) -> None:
        """Stop the model call if it is still running, log what it cost and
        free the reservation. Does nothing after the first call. Shielded: when
        the client leaves, the server cancels the stream, and the tokens
        already used must still be counted.

        A model call that is still waiting for the network when it is stopped
        cannot end before its next chunk. It is then not forgotten: the
        reservation stays until it does end, and what it cost after this point
        is logged then (`_settle_late`)."""
        if self._settled:
            return
        self._settled = True
        with anyio.CancelScope(shield=True):
            logged = 0
            still_running = False
            try:
                tokens = self._pending.routing_tokens
                if self._model is not None:
                    still_running = not await self._model.stop()
                    logged = self._model.total_tokens
                    tokens += logged
                efficiency = self._plan.metadata.get("efficiency") if self._plan is not None else None
                await _log_answer_tokens(self._pending.user.user_id, tokens, efficiency)
            finally:
                if still_running:
                    self._hold_until_model_ends(logged)
                else:
                    self._pending.reservation.release()

    def _hold_until_model_ends(self, logged_tokens: int) -> None:
        try:
            task = asyncio.get_running_loop().create_task(self._settle_late(logged_tokens))
        except RuntimeError:  # no loop to run it on (shutting down): nothing can wait
            self._pending.reservation.release()
            return
        _late_settlements.add(task)
        task.add_done_callback(_late_settlements.discard)

    async def _settle_late(self, logged_tokens: int) -> None:
        """Wait for the model call that was stopped too late to end, log the
        tokens it used after `settle` logged its share, and free the reservation."""
        try:
            await self._model.wait_finished(LATE_SETTLE_SECONDS)
            extra = self._model.total_tokens - logged_tokens
            if extra > 0:
                await _log_answer_tokens(self._pending.user.user_id, extra, None)
        finally:
            self._pending.reservation.release()


class _StreamedAnswer(StreamingResponse):
    """The NDJSON response of an _AnswerStream. It settles the answer on every
    way out, because when the client leaves before the first byte the event
    generator never starts and so never reaches its own `finally`."""

    def __init__(self, answer: _AnswerStream):
        super().__init__(answer.events(), media_type=STREAM_MEDIA_TYPE, headers=STREAM_HEADERS)
        self._answer = answer

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._answer.settle()
