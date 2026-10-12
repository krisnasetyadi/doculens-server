"""The compliance gap check skill: extracting a framework's items and checking target documents against them in parallel batches.

Mixin of services.rag.processor.PDFQAProcessor; the methods were moved
verbatim from that class by MS-657."""
import logging
import json
import re
from concurrent.futures import ThreadPoolExecutor
from services.rag.conversation.language import ReplyLanguage, detect_text_language
from typing import Any, Dict, List, Optional, Tuple
from services.rag.shared import _extract_total_tokens

logger = logging.getLogger(__name__)


class GapCheckMixin:
    # ===================== SKILL: Reference Framework Gap Analysis =====================
    # Generic "Skill" plumbing: skill_id picks a prompt builder + output schema.
    # "compliance_gap_check" (Skill 1) is validated first via ISO 27001/9001 —
    # nothing below is ISO-specific, ISO is just the first framework_name used.
    # "scenario_regulatory_impact" (Skill 2) is scaffolded elsewhere and not
    # wired into this orchestration yet — it is not validated for real use.

    GAP_CHECK_BATCH_SIZE = 8

    GAP_CHECK_MAX_WORKERS = 3  # cap concurrency to respect Gemini rate limits

    # Batches now multiply by len(target_collection_ids) (one guideline vs N
    # files), so a flat cap of 3 would serialize multi-target runs more and
    # more the more files are compared. Scale with target count, but stay
    # under a hard ceiling so we don't blow past Gemini's rate limits either.
    GAP_CHECK_MAX_WORKERS_CEILING = 6

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
        except Exception:
            pass
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return [str(x) for x in data]
            except Exception:
                pass
        return []

    def _parse_json_objects(self, raw: str) -> List[Dict[str, Any]]:
        """Same as _parse_json_list but for an array of objects."""
        text = re.sub(r'^```(json)?|```$', '', raw.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return []

    def extract_framework_items(
        self, reference_collection_ids: List[str], framework_name: str, language: ReplyLanguage = "id"
    ) -> Tuple[List[str], int]:
        """Ask the LLM to enumerate discrete requirement/control items from the
        reference collection(s). Generic — works for any standard/framework
        the user uploads, not just ISO. Returns (items, tokens_consumed) —
        the caller is responsible for metering/enforcement (MS-248), this
        function just reports what it spent."""
        reference_text = self._get_reference_text(reference_collection_ids)
        if not reference_text.strip():
            return [], 0

        # The example nudges the label language: an Indonesian example made
        # Gemini translate an English standard's item names into Indonesian.
        example = (
            '["A.5.1 Kebijakan keamanan informasi", "A.5.2 Peran dan tanggung jawab keamanan informasi"]'
            if language == "id"
            else '["A.5.1 Policies for information security", "A.5.2 Information security roles and responsibilities"]'
        )
        prompt = f"""Anda membaca dokumen standar/framework bernama "{framework_name}".

DOKUMEN:
{reference_text}

TUGAS: Daftar SEMUA item/klausul/kontrol/requirement yang disebutkan sebagai satuan terpisah di dokumen ini.

FORMAT WAJIB: kembalikan HANYA JSON array of string, tanpa markdown fence, tanpa penjelasan tambahan.
Contoh: {example}

JSON:"""

        llm, _ = self.get_llm(provider="gemini")
        raw = ""
        tokens_consumed = 0
        try:
            result = llm.invoke(prompt)
            raw = result.content.strip() if hasattr(result, 'content') else str(result).strip()
            tokens_consumed += _extract_total_tokens(result)
            items = self._parse_json_list(raw)
        except Exception as e:
            logger.error(f"extract_framework_items failed: {e}")
            items = []

        # Retry once on empty/malformed output — mirrors the retry already
        # done per-batch below, so a single bad generation doesn't sink the
        # whole run (this step has no batching to fall back on otherwise).
        if not items:
            logger.warning(
                f"extract_framework_items: first attempt returned no items "
                f"(raw response length={len(raw)}); retrying once"
            )
            try:
                retry_prompt = prompt + "\n\nPERHATIAN: balas HANYA dengan JSON array of string yang valid, tanpa teks lain, tanpa markdown fence."
                result = llm.invoke(retry_prompt)
                raw = result.content.strip() if hasattr(result, 'content') else str(result).strip()
                tokens_consumed += _extract_total_tokens(result)
                items = self._parse_json_list(raw)
            except Exception as e:
                logger.error(f"extract_framework_items retry failed: {e}")
                items = []
            if not items:
                logger.error(
                    f"extract_framework_items: retry also returned no items; "
                    f"raw response (first 500 chars): {raw[:500]!r}"
                )
        return items, tokens_consumed

    def _build_gap_check_batch_prompt(
        self, framework_name: str, batch_items: List[Dict[str, str]], language: ReplyLanguage = "id"
    ) -> str:
        """batch_items: list of {"label": ..., "target_context": ...}. Generic
        prompt — takes framework_name/context as parameters, no ISO-specific text.
        `language` only changes what `evidence`/`recommendation` are written
        in — status values and the JSON shape never change."""
        language_line = (
            ""
            if language == "id"
            else "\nPENTING: tulis nilai \"evidence\" dan \"recommendation\" dalam bahasa Inggris (English).\n"
        )
        items_block = "\n\n".join(
            f"ITEM {i+1}: {it['label']}\nBUKTI DARI DOKUMEN PERUSAHAAN:\n"
            f"{it['target_context'] or '(tidak ditemukan bukti terkait)'}"
            for i, it in enumerate(batch_items)
        )
        return f"""Anda adalah asisten audit compliance. Standar/framework yang dipakai: "{framework_name}".
Sumber standar ini kemungkinan berupa ringkasan/interpretasi pihak ketiga, BUKAN teks resmi berlisensi —
jangan berpura-pura mengutip klausul resmi kata demi kata.

Untuk setiap ITEM di bawah, tentukan apakah BUKTI dari dokumen perusahaan menunjukkan item itu sudah terpenuhi.

{items_block}

Status yang valid:
- "met": bukti jelas menunjukkan item terpenuhi
- "partial": ada bukti tapi tidak lengkap
- "not_met": tidak ada bukti relevan
- "unknown": tidak bisa disimpulkan dari bukti yang ada

FORMAT WAJIB: HANYA JSON array, tanpa markdown fence, satu object per item, urut sesuai urutan ITEM di atas:
[{{"label": "...", "status": "met|partial|not_met|unknown", "evidence": "kutipan singkat dari BUKTI atau kosong", "recommendation": "rekomendasi singkat kalau belum met, atau kosong"}}]
{language_line}
JSON:"""

    def run_compliance_gap_check(
        self,
        reference_collection_ids: List[str],
        target_collection_ids: List[str],
        framework_name: str,
    ) -> Tuple[List[Dict[str, Any]], str, int]:
        """Skill 1 — Compliance Gap Check. Generic map-per-batch orchestration
        (not an autonomous agent): extract reference items ONCE, then check
        each target collection independently against that same item list —
        so one guideline can be compared against several company files and
        each item comes back tagged with which file/collection it was
        checked against, instead of one merged verdict across all of them.
        Batches (item groups x target collections) run concurrently.
        Returns (items, disclaimer, tokens_consumed) — like
        extract_framework_items, metering/enforcement is the caller's job
        (MS-248); this just reports the total Gemini spend for the run."""
        # Gap Check has no chat question to detect a language from — the
        # findings are about the COMPANY's documents, so they're written in
        # that documents' language (no clear signal → English).
        language = detect_text_language(
            self._get_reference_text(target_collection_ids, max_chars=self.GAP_CHECK_LANGUAGE_SAMPLE_CHARS)
        )
        logger.info("run_compliance_gap_check: reply language=%s (from target documents)", language)
        items_labels, tokens_consumed = self.extract_framework_items(
            reference_collection_ids, framework_name, language
        )
        if language == "id":
            disclaimer = (
                f"Analisis ini berdasarkan dokumen \"{framework_name}\" yang diupload sebagai referensi "
                "— kemungkinan ringkasan/interpretasi pihak ketiga, bukan teks standar resmi berlisensi. "
                "Hasil ini bersifat bantuan awal, bukan audit/sertifikasi resmi."
            )
        else:
            disclaimer = (
                f"This analysis is based on the \"{framework_name}\" document uploaded as the reference "
                "— likely a third-party summary/interpretation, not the official licensed standard text. "
                "These results are an initial aid, not an official audit/certification."
            )
        if not items_labels:
            return [], disclaimer, tokens_consumed

        batches = [
            items_labels[i:i + self.GAP_CHECK_BATCH_SIZE]
            for i in range(0, len(items_labels), self.GAP_CHECK_BATCH_SIZE)
        ]

        def process_batch(batch_labels: List[str], target_collection_id: str) -> Tuple[List[Dict[str, Any]], int]:
            batch_tokens = 0
            batch_items = []
            for label in batch_labels:
                try:
                    results = self.search_across_collections(
                        label, collection_ids=[target_collection_id], top_k=3
                    )
                except Exception as e:
                    logger.warning(f"gap-check retrieval failed for item '{label}' in {target_collection_id}: {e}")
                    results = []
                target_context = "\n---\n".join(
                    doc.page_content[:1500] for doc in results[:3]
                )
                # Per-chunk "source" filename is already set at ingestion time
                # (services/ingest/documents.py) and preserved through retrieval — surface it here
                # so evidence points at an actual file, not just the collection id.
                source_files = []
                for doc in results[:3]:
                    src = doc.metadata.get("source")
                    if src and src not in source_files:
                        source_files.append(src)
                batch_items.append({
                    "label": label,
                    "target_context": target_context,
                    "source_files": source_files,
                })

            prompt = self._build_gap_check_batch_prompt(framework_name, batch_items, language)
            llm, _ = self.get_llm(provider="gemini")
            parsed: List[Dict[str, Any]] = []
            try:
                result = llm.invoke(prompt)
                raw = result.content.strip() if hasattr(result, 'content') else str(result).strip()
                batch_tokens += _extract_total_tokens(result)
                parsed = self._parse_json_objects(raw)
            except Exception as e:
                logger.error(f"gap-check batch LLM call failed: {e}")

            # Retry the batch once on parse failure / count mismatch — retry
            # per-batch, not per-run, so one bad batch doesn't cost a full re-run.
            if len(parsed) != len(batch_labels):
                try:
                    retry_prompt = prompt + "\n\nPERHATIAN: balas HANYA dengan JSON array yang valid, tanpa teks lain."
                    result = llm.invoke(retry_prompt)
                    raw = result.content.strip() if hasattr(result, 'content') else str(result).strip()
                    batch_tokens += _extract_total_tokens(result)
                    parsed = self._parse_json_objects(raw)
                except Exception as e:
                    logger.error(f"gap-check batch retry failed: {e}")

            out = []
            for i, label in enumerate(batch_labels):
                match = parsed[i] if i < len(parsed) else {}
                status = str(match.get("status", "unknown")).lower()
                if status not in ("met", "partial", "not_met", "unknown"):
                    status = "unknown"
                source_files = batch_items[i]["source_files"]
                citation = ", ".join(source_files) if source_files else f"target_collection:{target_collection_id}"
                out.append({
                    "label": match.get("label") or label,
                    "status": status,
                    "evidence": match.get("evidence") or None,
                    "source_citation": citation,
                    "recommendation": match.get("recommendation") or None,
                    "target_collection_id": target_collection_id,
                })
            return out, batch_tokens

        all_items: List[Dict[str, Any]] = []
        effective_workers = max(1, min(
            self.GAP_CHECK_MAX_WORKERS_CEILING,
            self.GAP_CHECK_MAX_WORKERS * len(target_collection_ids),
        ))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(process_batch, batch, target_collection_id)
                for target_collection_id in target_collection_ids
                for batch in batches
            ]
            for future in futures:
                try:
                    batch_items_out, batch_tokens = future.result()
                    all_items.extend(batch_items_out)
                    tokens_consumed += batch_tokens
                except Exception as e:
                    logger.error(f"gap-check batch future failed: {e}")

        return all_items, disclaimer, tokens_consumed
