"""The compliance gap check skill: extracting a framework's items and checking target documents against them in parallel batches.

Mixin of services.rag.processor.PDFQAProcessor; the methods were moved
verbatim from that class by MS-657. MS-1112 gave every requirement a stable
key, five statuses plus not_checked, and findings with structured pointers
(rules in services/rag/gap_locations.py)."""
import logging
import json
import os
import re
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import zip_longest
from config import config
from services.ingest.documents import extract_docx_blocks, extract_xlsx_blocks
from services.rag.conversation.language import ReplyLanguage, detect_text_language
from services.rag.gap_locations import build_requirements, finalize_item, unanswered_item
from services.storage import supabase as supabase_storage
from typing import Any, Callable, Dict, List, Optional, Tuple
from services.rag.shared import _extract_total_tokens

logger = logging.getLogger(__name__)


class GapCheckMixin:
    # ===================== SKILL: Reference Framework Gap Analysis =====================
    # Generic "Skill" plumbing: skill_id picks a prompt builder + output schema.
    # "compliance_gap_check" (Skill 1) is validated first via ISO 27001/9001 —
    # nothing below is ISO-specific, ISO is just the first framework_name used.
    # "scenario_regulatory_impact" (Skill 2) is scaffolded elsewhere and not
    # wired into this orchestration yet — it is not validated for real use.

    GAP_CHECK_BATCH_SIZE = 6

    GAP_CHECK_MAX_WORKERS = 3  # cap concurrency to respect Gemini rate limits

    # MS-1112: each requirement is judged once against passages from all the
    # selected documents together. Each document is asked for its own best
    # passages, so one document that mentions a topic often cannot crowd out
    # the one that actually covers it; the best of each document come first,
    # up to a cap that keeps a batch's prompt bounded.
    GAP_CHECK_PASSAGES_PER_DOCUMENT = 2
    GAP_CHECK_MAX_SOURCES_PER_ITEM = 8
    GAP_CHECK_SOURCE_CHARS = 1200

    # Chunking is 500 chars/chunk (config.py), so the old 12,000-char cap kept
    # only ~24 chunks — often just a framework doc's title/scope/definitions,
    # cut off before the actual itemized clause/control list. Gemini's context
    # window comfortably fits far more, so extraction gets the real content.
    GAP_CHECK_REFERENCE_MAX_CHARS = 60000

    # Enough target-document text to tell English from Indonesian reliably,
    # without reading whole collections just to pick a reply language.
    GAP_CHECK_LANGUAGE_SAMPLE_CHARS = 8000

    def _get_reference_text(self, reference_collection_ids: List[str], max_chars: Optional[int] = None) -> str:
        """Concatenate the raw text of one or more reference collections."""
        if max_chars is None:
            max_chars = self.GAP_CHECK_REFERENCE_MAX_CHARS
        parts = []
        for cid in reference_collection_ids:
            vector_store = self.get_vector_store(cid)
            if not vector_store:
                logger.warning(f"_get_reference_text: no vector store found for reference collection {cid}")
                continue
            try:
                docs = list(vector_store.docstore._dict.values())
            except Exception as e:
                logger.warning(f"_get_reference_text: could not read docstore for {cid}: {e}")
                docs = []
            if not docs:
                logger.warning(f"_get_reference_text: reference collection {cid} has no indexed documents")
            for doc in docs:
                parts.append(doc.page_content)
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            logger.info(f"_get_reference_text: truncating reference text from {len(text)} to {max_chars} chars")
            text = text[:max_chars] + "..."
        return text

    def _parse_json_list(self, raw: str) -> List[str]:
        """Best-effort JSON array parsing — strips markdown fences the model
        might add despite instructions, falls back to substring extraction.
        Also unwraps a single-key object (e.g. {"items": [...]})  the model
        may return instead of a bare array despite being asked for one."""
        text = re.sub(r'^```(json)?|```$', '', raw.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict):
                list_values = [v for v in data.values() if isinstance(v, list)]
                if len(list_values) == 1:
                    return [str(x) for x in list_values[0]]
        except ValueError:  # not JSON; try the next way
            pass
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return [str(x) for x in data]
            except ValueError:  # not JSON; try the next way
                pass
        return []

    def _parse_json_objects(self, raw: str) -> List[Dict[str, Any]]:
        """Same as _parse_json_list but for an array of objects."""
        text = re.sub(r'^```(json)?|```$', '', raw.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except ValueError:  # not JSON; try the next way
            pass
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
            except ValueError:  # not JSON; try the next way
                pass
        return []

    def extract_framework_items(
        self, reference_collection_ids: List[str], framework_name: str, language: ReplyLanguage = "id"
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Ask the LLM to enumerate the requirements in the reference
        collection(s), each with the kind of document that would show it is
        met, and give each a stable key (gap_locations.build_requirements).
        Generic — works for any standard the user uploads, not just ISO.
        Returns (requirements, tokens_consumed); metering is the caller's job
        (MS-248)."""
        reference_text = self._get_reference_text(reference_collection_ids)
        if not reference_text.strip():
            return [], 0

        # The example nudges the label language: an Indonesian example made
        # Gemini translate an English standard's item names into Indonesian.
        example = (
            '[{"label": "A.5.1 Kebijakan keamanan informasi", "expected_document": "kebijakan keamanan informasi"}, '
            '{"label": "A.8.13 Cadangan data", "expected_document": "prosedur backup dan rekaman uji pemulihan"}]'
            if language == "id"
            else '[{"label": "A.5.1 Policies for information security", "expected_document": "information security policy"}, '
            '{"label": "A.8.13 Information backup", "expected_document": "backup procedure and restore test records"}]'
        )
        prompt = f"""You are reading the standard/framework document "{framework_name}".

DOCUMENT:
{reference_text}

TASK: List EVERY item/clause/control/requirement this document states as a separate unit.
For each, keep its code at the start of "label" when it has one, and write in "expected_document"
the kind of company document that would show it is met (a few words).

REQUIRED FORMAT: return ONLY a JSON array of objects, no markdown fence, no extra text.
Example: {example}

JSON:"""

        llm, _ = self.get_llm()
        raw = ""
        tokens_consumed = 0
        entries: List[Any] = []
        for attempt in range(2):
            try:
                result = llm.invoke(prompt if attempt == 0 else prompt + "\n\nATTENTION: reply with ONLY a valid JSON array, no other text, no markdown fence.")
                raw = result.content.strip() if hasattr(result, 'content') else str(result).strip()
                tokens_consumed += _extract_total_tokens(result)
                # Objects as asked; a bare list of strings is still usable.
                entries = self._parse_json_objects(raw) or self._parse_json_list(raw)
            except Exception as e:
                logger.error(f"extract_framework_items attempt {attempt + 1} failed: {e}")
                entries = []
            if entries:
                break
            logger.warning(f"extract_framework_items: attempt {attempt + 1} returned no items (raw length={len(raw)})")
        if not entries:
            logger.error(f"extract_framework_items: no items; raw response (first 500 chars): {raw[:500]!r}")
        return build_requirements(entries), tokens_consumed

    def _build_gap_check_batch_prompt(
        self,
        framework_name: str,
        batch: List[Dict[str, Any]],
        document_names: List[str],
        directions: str = "",
        language: ReplyLanguage = "id",
    ) -> str:
        """batch: list of {"requirement", "sources"}. Generic prompt — no
        ISO-specific text. `language` only changes what the free text is
        written in — status values and the JSON shape never change."""
        language_line = (
            "Write reason, summary, next_step.explain, findings[].text and suggested_place in Indonesian (Bahasa Indonesia)."
            if language == "id"
            else "Write reason, summary, next_step.explain, findings[].text and suggested_place in English."
        )
        directions_block = f"\nDIRECTIONS FROM THE USER (follow them):\n{directions.strip()}\n" if directions.strip() else ""

        def source_line(s: Dict[str, Any]) -> str:
            page = f" (page {s['page']})" if s.get("page") not in (None, "") else ""
            return f"[{s['id']}] {s['file_name']}{page}:\n{s['text']}"

        def item_block(i: int, it: Dict[str, Any]) -> str:
            req = it["requirement"]
            name = f"{req['code']} {req['label']}" if req.get("code") else req["label"]
            sources = "\n".join(source_line(s) for s in it["sources"]) or "(no passages found in the selected documents)"
            return (
                f"ITEM {i + 1} (key: {req['key']}): {name}\n"
                f"Expected kind of document: {req.get('expected_document') or '(not stated)'}\n"
                f"SOURCES:\n{sources}"
            )

        items_block = "\n\n".join(item_block(i, it) for i, it in enumerate(batch))
        documents = "\n".join(f"- {name}" for name in document_names) or "- (none)"
        return f"""You are a compliance audit assistant. Standard/framework used: "{framework_name}".
The source of this standard is likely a third-party summary/interpretation, NOT the official licensed text —
do not pretend to quote official clauses word for word.

THE USER SELECTED ONLY THESE DOCUMENTS TO CHECK:
{documents}
{directions_block}
For each ITEM below, judge from its SOURCES (passages from the selected documents) whether the requirement is met.

{items_block}

Valid "status" values:
- "done": the sources clearly show the requirement is met
- "incomplete": the sources cover it, but something is missing
- "missing": the selected documents are the kind that should cover it, but nothing does
- "needs_decision": the sources conflict, or you cannot decide; a person must decide
- "not_applicable": the sources explicitly state this requirement does not apply (e.g. excluded in a Statement of Applicability)
- "not_checked": the requirement needs a kind of document that is NOT among the selected documents above, so it could not be checked. Use this instead of "missing" when the sources are about something else.

For every item also give:
- "reason": ONE short sentence a non-expert understands
- "summary": 1-2 sentences: what is missing (or why it is done / why it needs a decision)
- "next_step": {{"action": "upload_document|revise_document|add_evidence|decide_status", "explain": one line}} (omit for done, not_applicable, not_checked)
- "findings": 1-3 things you found or did not find. Each: "verdict" ("met"|"partial"|"missing"), "text" (short),
  "source" (the [S#] it comes from, or null), "quote" (a few words copied EXACTLY from that source, or null),
  "suggested_place" (for a missing part of an existing document: where it should go, e.g. "after Section 4 of Backup Procedure", else null).
  Every finding with verdict "met" or "partial" MUST name its source.

{language_line}

REQUIRED FORMAT: ONLY a JSON array, no markdown fence, one object per item, in ITEM order:
[{{"key": "...", "status": "...", "reason": "...", "summary": "...", "next_step": {{"action": "...", "explain": "..."}}, "findings": [{{"verdict": "...", "text": "...", "source": "S1", "quote": "...", "suggested_place": null}}]}}]

JSON:"""

    def run_compliance_gap_check(
        self,
        reference_collection_ids: List[str],
        target_collection_ids: List[str],
        framework_name: str,
        requirements: Optional[List[Dict[str, Any]]] = None,
        directions: str = "",
        searched: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], str, int, List[Dict[str, Any]]]:
        """Skill 1 — Compliance Gap Check (MS-1112). Map-per-batch
        orchestration, not an autonomous agent:

        1. pick the reply language from the company's documents;
        2. take the analysis's requirement list, or read it from the reference
           on the analysis's first run;
        3. judge each requirement once, against passages from every selected
           document, in concurrent batches; each finding gets a pointer into
           the file it came from.

        `searched` ({"folders": [names], "document_count": n}) is what a
        missing item reports as where the AI looked.
        Returns (items, disclaimer, tokens_consumed, requirements); metering
        is the caller's job (MS-248)."""
        # The findings are about the COMPANY's documents, so they're written
        # in those documents' language (no clear signal → English).
        language = detect_text_language(
            self._get_reference_text(target_collection_ids, max_chars=self.GAP_CHECK_LANGUAGE_SAMPLE_CHARS)
        )
        logger.info("run_compliance_gap_check: reply language=%s (from target documents)", language)

        tokens_consumed = 0
        if requirements is None:
            requirements, tokens_consumed = self.extract_framework_items(reference_collection_ids, framework_name, language)
        disclaimer = self._gap_check_disclaimer(framework_name, language)
        if not requirements:
            return [], disclaimer, tokens_consumed, []

        check = _BatchCheck(
            framework_name=framework_name,
            target_collection_ids=target_collection_ids,
            document_names=self._target_document_names(target_collection_ids),
            directions=directions,
            language=language,
            searched=searched or {"folders": [], "document_count": len(target_collection_ids)},
            load_blocks=self._document_blocks_loader(),
        )
        batches = [
            requirements[i:i + self.GAP_CHECK_BATCH_SIZE]
            for i in range(0, len(requirements), self.GAP_CHECK_BATCH_SIZE)
        ]
        results: List[Optional[List[Dict[str, Any]]]] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=self.GAP_CHECK_MAX_WORKERS) as executor:
            futures = {executor.submit(self._check_batch, batch, check): i for i, batch in enumerate(batches)}
            for future, i in futures.items():
                try:
                    results[i], batch_tokens = future.result()
                    tokens_consumed += batch_tokens
                except Exception as e:
                    logger.error(f"gap-check batch future failed: {e}")

        # A batch that crashed outright still keeps its requirements in the
        # run, marked unanswered: not a verdict on the documents, and left out
        # of trends.
        items: List[Dict[str, Any]] = []
        for batch, batch_items in zip(batches, results):
            if batch_items is None:
                batch_items = [unanswered_item(r) for r in batch]
            items.extend(batch_items)
        return items, disclaimer, tokens_consumed, requirements

    def _check_batch(self, requirements: List[Dict[str, Any]], check: "_BatchCheck") -> Tuple[List[Dict[str, Any]], int]:
        """Judge one batch of requirements in one LLM call (retried once on a
        malformed answer: per batch, not per run, so one bad batch doesn't cost
        a full re-run), then turn each verdict into a stored item."""
        batch = [{"requirement": r, "sources": self._retrieve_sources(r, check.target_collection_ids)} for r in requirements]
        prompt = self._build_gap_check_batch_prompt(
            check.framework_name, batch, check.document_names, check.directions, check.language
        )
        llm, _ = self.get_llm()
        tokens = 0
        verdicts: List[Dict[str, Any]] = []
        for attempt in range(2):
            try:
                result = llm.invoke(prompt if attempt == 0 else prompt + "\n\nATTENTION: reply with ONLY a valid JSON array, no other text.")
                raw = result.content.strip() if hasattr(result, 'content') else str(result).strip()
                tokens += _extract_total_tokens(result)
                verdicts = [v for v in self._parse_json_objects(raw) if isinstance(v, dict)]
            except Exception as e:
                logger.error(f"gap-check batch attempt {attempt + 1} failed: {e}")
            if len(verdicts) == len(requirements):
                break

        # Matched by key; by position only when the answer has one per item.
        by_key = {str(v.get("key")): v for v in verdicts if v.get("key")}
        in_order = len(verdicts) == len(requirements)
        items = []
        for i, entry in enumerate(batch):
            verdict = by_key.get(entry["requirement"]["key"]) or (verdicts[i] if in_order else None)
            items.append(finalize_item(entry["requirement"], verdict, entry["sources"], check.searched, check.load_blocks))
        return items, tokens

    def _retrieve_sources(self, requirement: Dict[str, Any], target_collection_ids: List[str]) -> List[Dict[str, Any]]:
        """The passages the LLM judges a requirement on, numbered S1, S2, ...:
        each selected document's own best passages, interleaved so every
        document's best comes before any document's second best, capped at
        GAP_CHECK_MAX_SOURCES_PER_ITEM."""
        query = " ".join(filter(None, [requirement.get("code"), requirement["label"], requirement.get("expected_document")]))
        per_document = []
        for collection_id in target_collection_ids:
            try:
                per_document.append(self.search_across_collections(
                    query, collection_ids=[collection_id], top_k=self.GAP_CHECK_PASSAGES_PER_DOCUMENT
                )[:self.GAP_CHECK_PASSAGES_PER_DOCUMENT])
            except Exception as e:
                logger.warning(f"gap-check retrieval failed for '{requirement['key']}' in {collection_id}: {e}")
        interleaved = [doc for rank in zip_longest(*per_document) for doc in rank if doc is not None]
        return [
            {
                "id": f"S{n}",
                "file_name": doc.metadata.get("source") or "",
                "collection_id": doc.metadata.get("collection_id"),
                "page": doc.metadata.get("page"),
                "text": doc.page_content[:self.GAP_CHECK_SOURCE_CHARS],
            }
            for n, doc in enumerate(interleaved[:self.GAP_CHECK_MAX_SOURCES_PER_ITEM], 1)
        ]

    @staticmethod
    def _gap_check_disclaimer(framework_name: str, language: ReplyLanguage) -> str:
        if language == "id":
            return (
                f"Analisis ini berdasarkan dokumen \"{framework_name}\" yang diupload sebagai referensi "
                "— kemungkinan ringkasan/interpretasi pihak ketiga, bukan teks standar resmi berlisensi. "
                "Hasil ini bersifat bantuan awal, bukan audit/sertifikasi resmi."
            )
        return (
            f"This analysis is based on the \"{framework_name}\" document uploaded as the reference "
            "— likely a third-party summary/interpretation, not the official licensed standard text. "
            "These results are an initial aid, not an official audit/certification."
        )

    def _target_document_names(self, target_collection_ids: List[str]) -> List[str]:
        """File names of the selected documents, for the prompt's "only these
        were selected" list that decides not_checked versus missing."""
        return selected_document_names(target_collection_ids)

    def _document_blocks_loader(self):
        """load(collection_id, file_name) -> the structured blocks of one Word
        or Excel file, read once per run: from the local upload folder, or
        downloaded from the bucket when the disk copy is gone (HF Spaces).
        MS-1112 re-reads at check time so documents uploaded before exact
        pointers existed get them too, without re-indexing. Each file has its
        own lock, so batches only wait for a file another batch is reading."""
        cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        file_locks: Dict[Tuple[str, str], threading.Lock] = defaultdict(threading.Lock)
        locks_lock = threading.Lock()

        def load(collection_id: Optional[str], file_name: str) -> List[Dict[str, Any]]:
            if not collection_id or not file_name or ".." in file_name or file_name.startswith(("/", "\\")):
                return []
            extract = {".docx": extract_docx_blocks, ".xlsx": extract_xlsx_blocks}.get(os.path.splitext(file_name)[1].lower())
            if not extract:
                return []
            key = (collection_id, file_name)
            with locks_lock:
                file_lock = file_locks[key]
            with file_lock:
                if key not in cache:
                    cache[key] = _read_document_blocks(collection_id, file_name, extract)
                return cache[key]

        return load


def selected_document_names(collection_ids: List[str]) -> List[str]:
    """The file names in the selected collections: one collection can hold
    several files, so this, not the number of collections, is how many
    documents a run checks."""
    names: List[str] = []
    for cid in collection_ids:
        row = supabase_storage.get_collection(cid) or {}
        for name in row.get("file_names") or [row.get("title") or cid]:
            if name and name not in names:
                names.append(name)
    return names


@dataclass
class _BatchCheck:
    """What every batch of one run shares."""
    framework_name: str
    target_collection_ids: List[str]
    document_names: List[str]
    directions: str
    language: ReplyLanguage
    searched: Dict[str, Any]
    load_blocks: Callable[[Optional[str], str], List[Dict[str, Any]]]


def _read_document_blocks(collection_id: str, file_name: str, extract) -> List[Dict[str, Any]]:
    """A Word or Excel file's blocks, from the local upload folder or, when the
    disk copy is gone, a temporary download from the bucket."""
    path = os.path.join(os.path.abspath(config.upload_folder), collection_id, file_name)
    if os.path.exists(path):
        return extract(path)
    if not supabase_storage.is_enabled():
        return []
    fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file_name)[1])
    os.close(fd)
    try:
        return extract(temp_path) if supabase_storage.download_document(collection_id, file_name, temp_path) else []
    finally:
        os.remove(temp_path)
