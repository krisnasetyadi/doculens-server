"""Turning search results into an answer: prompts per question type, the LLM call, cleanup of its output, and the fallbacks when it fails.

Mixin of services.rag.processor.PDFQAProcessor; the methods were moved
verbatim from that class by MS-657."""
import logging
import re
from models import SourceInfo
from services.rag.conversation.language import ReplyLanguage, answer_language_instruction, detect_reply_language
from services.rag.conversation.messages import fallback_text
from services.rag.efficient_mode.compressor import ContextPart, compress_context, compress_memory
from services.rag.efficient_mode.token_estimate import estimate_tokens
from typing import Any, Dict, List, Optional, Tuple
from services.rag.shared import _extract_total_tokens, _off_topic_instruction

logger = logging.getLogger(__name__)


class AnswerMixin:
    def generate_hybrid_answer(
        self,
        hybrid_results: Dict[str, Any],
        question: str,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        memory: Optional[List[Dict[str, str]]] = None,
        skill_instruction: Optional[str] = None,
        efficient_mode: bool = False,
        language: Optional[ReplyLanguage] = None,
        task: str = "answer",
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Enhanced hybrid answer generation dengan conflict resolution and comprehensive metadata.

        `language` is the reply language (en/id); detected from the question
        (then memory, then English) when the caller doesn't pass one.
        `task="summarize"`: `hybrid_results` came from build_summary_results
        (chunks spread over whole documents) — every chunk goes into the
        context and the summary prompt is used."""
        is_summary = task == "summarize"

        language = language or detect_reply_language(question, memory)
        llm, model_id = self.get_llm(llm_provider, llm_model)
        
        intent_analysis = hybrid_results.get('search_analysis', {}).get('intent', {})
        merged_results = hybrid_results.get('merged_results', [])
        
        # Enhanced conflict detection
        conflicts = self.detect_conflicts(merged_results)
        
        # Track processing steps for observability
        processing_steps = ["Intent analysis completed", "Sources merged and ranked"]
        
        # Check if we have chat or DB results with good info - prioritize them over PDF
        question_lower = question.lower()
        # NOTE: "yang" was removed from this list — it's one of the most
        # common words in Indonesian (relative pronoun "that/which"), so it
        # matched most questions and mis-triggered person-query handling below.
        is_person_query = any(kw in question_lower for kw in ['siapa', 'who', 'handle', 'contact'])
        
        # Track exact matches and boost application
        exact_matches = []
        boost_applied = {"exact_match": 0.0, "person_query": 0.0, "brand_whitelist": 0.0}
        
        # Detect exact matches in results
        search_terms = hybrid_results.get('search_terms', [])
        for result in merged_results[:5]:
            content_lower = result.get('content', '').lower()
            for term in search_terms:
                if term.lower() in content_lower:
                    exact_matches.append({"term": term, "source": result.get('source', 'Unknown'), "type": result['type']})
                    # Check if boost was applied (from database exact match algorithm)
                    if result['type'] == 'database' and result.get('confidence', 0) > 0.9:
                        boost_applied["exact_match"] += 50.0  # +50 boost from services/rag/app_database.py
        
        processing_steps.append(f"Detected {len(exact_matches)} exact matches")
        
        # For person queries, prioritize chat and DB results
        if is_person_query and merged_results:
            # Reorder to prioritize chat/db, but keep every other type
            # (pdf, public_link, external_db, ...) instead of dropping them —
            # a plain equality filter here silently discarded any type it
            # didn't know about.
            chat_results = [r for r in merged_results if r['type'] == 'chat']
            db_results = [r for r in merged_results if r['type'] in ('database', 'external_db')]
            other_results = [r for r in merged_results if r['type'] not in ('chat', 'database', 'external_db')]
            merged_results = chat_results + db_results + other_results
            boost_applied["person_query"] = 1.8  # Person query boost multiplier
            processing_steps.append("Person query detected - prioritized chat/DB results")
        
        # Check if confidence is too low and no chat/db results - use direct extraction
        # Use a dynamic threshold: lower for smarter models like Gemini to leverage generative reasoning
        threshold = 0.12 if 'gemini' in model_id.lower() else 0.18
        if not is_summary and merged_results and merged_results[0]['confidence'] < threshold:
            if merged_results[0]['type'] == 'pdf' and not any(r['type'] in ['chat', 'database'] for r in merged_results[:3]):
                logger.warning(f"Low confidence ({merged_results[0]['confidence']:.2%}), using direct extraction")
                return self._extract_direct_answer(merged_results[0], question, language), model_id, {
                    "intent": "direct_extraction",
                    "sources_used": {"pdf": 1, "database": 0, "chat": 0},
                    "top_confidence": merged_results[0]['confidence'],
                    "conflicts_detected": False,
                    "extraction_method": "direct"
                }
        
        # Prepare context dengan prioritization - SHORTER for small models
        context_parts = []
        # MS-247 "Efficient Mode" — built alongside context_parts (same
        # loop, same data, no extra retrieval work) so the compressor
        # below has each chunk's header/body/confidence without needing
        # to re-derive them from the already-flattened strings. Unused
        # unless efficient_mode is on.
        context_records: List[ContextPart] = []
        source_breakdown = {"pdf": 0, "database": 0, "chat": 0, "public_link": 0, "external_db": 0}
        
        # Limit to top 3 results and shorter snippets for small models.
        # Large-context models (Gemini) get full chunks: truncating to a few
        # hundred chars discards the very passage retrieval worked to find.
        is_small_model = 'flan-t5' in model_id.lower()
        max_results = 3 if is_small_model else 8
        if is_summary:
            # build_summary_results already capped and sized these chunks.
            max_results = len(merged_results)
        base_max_content_len = 150 if is_small_model else 2000
        # Tabular sources need a bigger allowance than prose: a table chunk
        # (now sized to hold a whole small table, see external_db_chunk_size)
        # must not be re-truncated here, or the same "only some rows survive"
        # problem the bigger chunk size was meant to fix just reappears.
        structured_max_content_len = 150 if is_small_model else 6000

        for i, result in enumerate(merged_results[:max_results]):
            source_type = result['type']
            source_breakdown[source_type] += 1

            if is_small_model:
                # Small-context models need keyword-window extraction
                content_snippet = self._extract_relevant_snippet(result['content'], question)
            else:
                # Large-context models (Gemini) get the retrieved chunk as-is,
                # matching the research notebook pipeline: keyword-window
                # extraction on messy PDF text often clips out the relevant
                # passage that retrieval already found.
                content_snippet = result['content']

            max_content_len = (
                structured_max_content_len
                if source_type in ('database', 'external_db')
                else base_max_content_len
            )
            # Truncate if too long based on model
            if len(content_snippet) > max_content_len:
                content_snippet = content_snippet[:max_content_len] + "..."

            header = f"[Sumber {i+1} — {result['source']}]:"
            context_parts.append(f"{header}\n{content_snippet}")
            context_records.append(
                ContextPart(header=header, body=content_snippet, confidence=result.get('confidence', 0) or 0)
            )

        context = "\n\n---\n\n".join(context_parts)

        # MS-247 "Efficient Mode" — opt-in, isolated compression pass over
        # the assembled RAG context (see services/rag/efficient_mode/compressor.py).
        # Guarded entirely behind `efficient_mode` — including the
        # estimate_tokens() calls — so a caller that never opts in gets no
        # extra work done and no `efficiency` key at all in the metadata
        # (see the answer_metadata assembly below), not just an unused
        # no-op value. That matters beyond CPU cost: answer_metadata gets
        # persisted (services/agnostic.py -> log_token_usage), so leaving
        # this unconditional would write a raw/final-tokens row for every
        # query in the app, not just the ones actually testing the toggle.
        efficiency_meta: Optional[Dict[str, Any]] = None
        if efficient_mode:
            raw_tokens_est = estimate_tokens(context)
            compressed_context, compression_stats = compress_context(context_records, question)
            final_tokens_est = estimate_tokens(compressed_context)
            reduction_pct = (
                round(100 * (1 - final_tokens_est / raw_tokens_est), 1)
                if raw_tokens_est
                else 0.0
            )
            efficiency_meta = {
                "enabled": True,
                "raw_chars": len(context),
                "final_chars": len(compressed_context),
                "raw_tokens_est": raw_tokens_est,
                "final_tokens_est": final_tokens_est,
                "reduction_pct": reduction_pct,
                "parts_before": compression_stats.parts_before,
                "parts_after": compression_stats.parts_after,
                "deduplicated_chunks": compression_stats.deduplicated,
                "sections_pruned": compression_stats.sections_pruned,
            }
            context = compressed_context

        # Build enhanced prompt berdasarkan intent
        # Aggregation prompt demands numeric computation and treats "database"
        # as the source of truth — that's only correct when the request
        # explicitly scoped the search to database alone (source of truth =
        # what the user selected). If nothing/multiple sources were selected,
        # or the selection wasn't database-only, fall through to the general
        # prompt so words like "jumlah desimal" in a document/chat question
        # don't force a numeric-only answer style.
        source_weights = hybrid_results.get('search_analysis', {}).get('source_weights', {})
        selected_sources = {k for k, w in source_weights.items() if w and w > 0}
        db_is_sole_source = selected_sources in ({'db'}, {'external_db'})
        has_db_results = source_breakdown.get('database', 0) + source_breakdown.get('external_db', 0) > 0

        # Sole-source-type detection: when the user filtered the request down
        # to exactly one source type, the prompt wording should match that
        # source's shape ("percakapan chat" vs "dokumen" vs "data tabel")
        # instead of always saying "dokumen". None covers multi-source /
        # unfiltered requests, which must keep the original combined wording
        # unchanged — see _source_prompt_vocab.
        sole_source_type = None
        if selected_sources == {'pdf'}:
            sole_source_type = 'pdf'
        elif selected_sources == {'chat'}:
            sole_source_type = 'chat'
        elif selected_sources == {'public_link'}:
            sole_source_type = 'public_link'
        elif db_is_sole_source:
            sole_source_type = 'database'

        if is_summary:
            prompt = self._build_summary_prompt(context, question, language=language)
        elif intent_analysis.get('is_aggregation') and db_is_sole_source and has_db_results:
            prompt = self._build_aggregation_prompt(context, question, conflicts, language=language)
        elif intent_analysis.get('is_comparison'):
            prompt = self._build_comparison_prompt(context, question, conflicts, language=language)
        elif intent_analysis.get('is_explanation'):
            prompt = self._build_explanation_prompt(
                context, question, conflicts, sole_source_type=sole_source_type, language=language
            )
        else:
            prompt = self._build_general_prompt(
                context, question, conflicts, source_breakdown, sole_source_type=sole_source_type, language=language
            )

        # MS-237 poin 1: prepend the previous-5-chats window (a chat being
        # one question plus its answer) so a
        # follow-up like "ringkas semua di atas" or "yang tadi" has
        # something to resolve against. Prepended to the finished `prompt`
        # (not folded into `context` above) so it stands on its own instead
        # of reading as part of the "DOKUMEN:" section every builder wraps
        # context in. Caller (services/agnostic.py) has already clamped
        # `memory` to the last 5 chats — this only formats it.
        if memory:
            # MS-247 "Efficient Mode" — cleans/dedups memory turns (see
            # compress_memory's own docstring for why this is narrower
            # than compress_context's BM25-relevance pruning: dropping a
            # turn for scoring "irrelevant" against a vague follow-up
            # question is exactly the wrong call here). Folded into the
            # SAME efficiency_meta the context compression above already
            # built, so the popup reports one combined before/after
            # number rather than two separate ones.
            memory_for_prompt = memory
            if efficient_mode and efficiency_meta is not None:
                raw_memory_text = " ".join(str(t.get('content', '')) for t in memory)
                memory_for_prompt, memory_stats = compress_memory(memory)
                final_memory_text = " ".join(str(t.get('content', '')) for t in memory_for_prompt)
                raw_memory_tokens_est = estimate_tokens(raw_memory_text)
                final_memory_tokens_est = estimate_tokens(final_memory_text)
                efficiency_meta["raw_chars"] += len(raw_memory_text)
                efficiency_meta["final_chars"] += len(final_memory_text)
                efficiency_meta["raw_tokens_est"] += raw_memory_tokens_est
                efficiency_meta["final_tokens_est"] += final_memory_tokens_est
                efficiency_meta["memory_turns_deduplicated"] = memory_stats.deduplicated
                total_raw = efficiency_meta["raw_tokens_est"]
                efficiency_meta["reduction_pct"] = (
                    round(100 * (1 - efficiency_meta["final_tokens_est"] / total_raw), 1)
                    if total_raw
                    else 0.0
                )

            history_lines = [
                f"{'User' if turn.get('role') == 'user' else 'Asisten'}: {turn.get('content', '')}"
                for turn in memory_for_prompt
            ]
            history_block = (
                "RIWAYAT PERCAKAPAN (untuk memahami rujukan seperti \"itu\"/\"yang tadi\"):\n"
                + "\n".join(history_lines)
            )
            prompt = f"{history_block}\n\n{prompt}"

        # MS-252: one-shot skill invocation — prepended last (outermost) so
        # it reads before the conversation history and the task/document
        # prompt it wraps. Same orthogonal-prepend shape as `memory` above:
        # never folded into `context` or any `_build_*_prompt` builder, so
        # it applies regardless of detected intent. Caller (router/
        # agnostic.py) has already resolved + authorization-checked the
        # skill — this only formats it.
        if skill_instruction:
            skill_block = f"INSTRUKSI KHUSUS (skill aktif untuk pesan ini):\n{skill_instruction.strip()}"
            prompt = f"{skill_block}\n\n{prompt}"
            
        # MS-247 "Efficient Mode" — the actual, final prompt text this
        # request sends to the LLM, so a developer testing the toggle can
        # see exactly what survived compression (or didn't) instead of
        # only the char/token counts in the response. Only logged when
        # the toggle was actually on for this request, so this doesn't
        # add PDF/DB content to every request's logs by default.
        if efficient_mode:
            logger.info(
                "MS-247 Efficient Mode — final prompt sent to LLM (%d chars):\n%s",
                len(prompt), prompt,
            )

        try:
            result = llm.invoke(prompt)
            answer = result.content.strip() if hasattr(result, 'content') else str(result).strip()
            total_tokens = _extract_total_tokens(result)
            # Raw shape straight off the LLM response — proof this number comes
            # from Gemini's own usage_metadata (langchain_google_genai just
            # relays prompt_token_count/candidates_token_count/total_token_count
            # into this dict), not something we or LangChain compute ourselves.
            logger.info("LLM usage_metadata: %s", getattr(result, 'usage_metadata', None))

            # Validasi dan post-processing
            if not is_summary:
                # (A summary that opens with "Sorry," or is short is still the
                # summary — swapping in one raw chunk would be strictly worse.)
                answer = self._validate_and_clean_answer(answer, question, merged_results, language)
            processing_steps.append("Answer validated and cleaned")

            # Generate comprehensive metadata untuk response
            answer_metadata = {
                "confidence_score": merged_results[0]['confidence'] if merged_results else 0,
                "primary_intent": intent_analysis.get('primary_intent', 'data_retrieval'),
                "sources_used": source_breakdown,
                "search_strategy": "hybrid_multi_source",
                "conflicts_detected": len(conflicts) > 0,
                "exact_matches": exact_matches[:10],  # Top 10 exact matches
                "boost_applied": boost_applied,
                "ranking_algorithm": "weighted_scoring_with_intent",
                "processing_steps": processing_steps,
                "total_results_processed": len(merged_results),
                "conflict_details": conflicts if conflicts else [],
                "model_used": model_id,
                "is_person_query": is_person_query,
                "total_tokens": total_tokens,
            }
            if efficiency_meta:
                answer_metadata["efficiency"] = efficiency_meta

            return answer, model_id, answer_metadata
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            processing_steps.append(f"LLM generation failed: {str(e)[:100]}")
            return self._generate_fallback_answer(hybrid_results, question, language), model_id, {
                "error": str(e),
                "processing_steps": processing_steps,
                "sources_used": {"pdf": 0, "database": 0, "chat": 0},
                "confidence_score": 0,
                "conflicts_detected": False
            }

    # English "not found" replies, quoted verbatim by the model when the reply
    # language is English. Must not contain any failure_patterns substring
    # (see _validate_and_clean_answer) — same rule as the Indonesian ones.
    _NOT_FOUND_EN = {
        'pdf': 'This information was not found in the uploaded documents.',
        'public_link': 'This information was not found in the available public-link documents.',
        'chat': 'This information was not found in the available chat conversations.',
        'database': 'This information was not found in the available data.',
    }

    def _source_prompt_vocab(self, sole_source_type: Optional[str], language: ReplyLanguage = "id") -> Dict[str, str]:
        """Wording per jenis source, dipakai hanya kalau user memfilter request
        ke tepat satu source type (sole_source_type). None (multi-source atau
        tidak difilter) selalu jatuh ke wording 'pdf' — perilaku lama, tidak
        berubah — supaya jalur gabungan tetap identik dengan sebelumnya.
        Untuk language="en" hanya kalimat `not_found` yang diganti ke English
        (kalimat itu dikutip apa adanya ke jawaban); sisanya tetap instruksi."""
        if language == "en":
            key = sole_source_type if sole_source_type in self._NOT_FOUND_EN else 'pdf'
            entry = dict(self._source_prompt_vocab(sole_source_type, "id"))
            entry['not_found'] = self._NOT_FOUND_EN[key]
            return entry
        vocab = {
            'pdf': {
                'noun': 'dokumen',
                'label': 'DOKUMEN',
                'not_found': 'Informasi ini tidak ditemukan dalam dokumen yang diunggah.',
                'extra_instruction': '',
            },
            'public_link': {
                'noun': 'dokumen dari tautan publik',
                'label': 'DOKUMEN DARI TAUTAN PUBLIK',
                'not_found': 'Informasi ini tidak ditemukan dalam dokumen tautan publik yang tersedia.',
                'extra_instruction': '',
            },
            'chat': {
                'noun': 'percakapan chat',
                'label': 'PERCAKAPAN CHAT',
                'not_found': 'Informasi ini tidak ditemukan dalam percakapan chat yang tersedia.',
                'extra_instruction': '- Percakapan chat bisa informal/singkatan/emoji — tetap ekstrak maknanya, jangan tuntut struktur formal.',
            },
            'database': {
                'noun': 'data tabel',
                'label': 'DATA TABEL',
                'not_found': 'Informasi ini tidak ditemukan dalam data yang tersedia.',
                'extra_instruction': '- Jika data berupa angka/tabel, tampilkan persis apa adanya tanpa dibulatkan sendiri.',
            },
        }
        return vocab.get(sole_source_type, vocab['pdf'])

    def _build_aggregation_prompt(
        self, context: str, question: str, conflicts: List, language: ReplyLanguage = "id"
    ) -> str:
        """Build prompt untuk aggregation queries"""
        conflict_note = ""
        if conflicts:
            conflict_note = "\nPERHATIAN: Ditemukan perbedaan data antara sumber. Gunakan sumber database sebagai referensi utama."
        answer_format = (
            '"Berdasarkan data dari [sumber], [jawaban numerik]"'
            if language == "id"
            else '"Based on data from [source], [numeric answer]"'
        )

        return f"""Anda adalah asisten analisis data. Hitung dan berikan jawaban numerik berdasarkan data berikut.

    DATA YANG TERSEDIA (sudah diurutkan berdasarkan kepercayaan):
    {context}

    PERTANYAAN: {question}
    {conflict_note}

    PETUNJUK:
    1. Hitung nilai yang diminta (total, rata-rata, jumlah, dll)
    2. Gunakan format: {answer_format}
    3. Jika ada perbedaan data, sebutkan perbedaan tersebut
    4. Sertakan satuan jika relevan (Rp, unit, orang, dll)
    {answer_language_instruction(language, "5. Berikan jawaban dalam Bahasa Indonesia")}
    {_off_topic_instruction(language)}

    JAWABAN:"""

    def _build_general_prompt(
        self,
        context: str,
        question: str,
        conflicts: List,
        source_breakdown: Dict,
        sole_source_type: Optional[str] = None,
        language: ReplyLanguage = "id",
    ) -> str:
        """Build prompt untuk general queries"""
        vocab = self._source_prompt_vocab(sole_source_type, language)
        extra_instruction = f"\n{vocab['extra_instruction']}" if vocab['extra_instruction'] else ""

        return f"""Jawab pertanyaan berdasarkan {vocab['noun']} berikut SAJA.

{vocab['label']}:
{context}

Pertanyaan: {question}

INSTRUKSI PENTING:
- Jawab HANYA berdasarkan isi {vocab['noun']} di atas
- Jika {vocab['noun']} memuat informasi yang berkaitan sebagian dengan pertanyaan, jawab sebaik mungkin dari bagian yang relevan itu dan sebutkan bahwa informasinya terbatas
- Jika {vocab['noun']} mengandung angka, tabel, atau data — ekstrak dan tampilkan langsung dalam jawaban
- Katakan "{vocab['not_found']}" HANYA jika tidak ada satu pun bagian {vocab['noun']} yang berkaitan dengan pertanyaan
- JANGAN gunakan pengetahuan umum atau informasi dari luar {vocab['noun']}{extra_instruction}
{_off_topic_instruction(language)}
{answer_language_instruction(language, "- Jawab ringkas dan jelas dalam Bahasa Indonesia")}

Jawaban:"""

    def _build_comparison_prompt(
        self, context: str, question: str, conflicts: List, language: ReplyLanguage = "id"
    ) -> str:
        """Build prompt untuk comparison queries"""
        return f"""Anda adalah asisten analisis yang membantu membandingkan data.

DATA YANG TERSEDIA:
{context}

PERTANYAAN: {question}

INSTRUKSI:
1. Bandingkan data yang diminta
2. Highlight perbedaan utama
3. Gunakan format tabel atau bullet points jika sesuai
4. Berikan kesimpulan singkat
{answer_language_instruction(language, "5. Jawab dalam Bahasa Indonesia")}
{_off_topic_instruction(language)}

JAWABAN:"""

    def _build_summary_prompt(self, context: str, question: str, language: ReplyLanguage = "id") -> str:
        """Prompt for task="summarize" — the context is a sample of chunks
        spread over the whole document (build_summary_results), in order."""
        return f"""Anda adalah asisten yang meringkas dokumen milik user HANYA berdasarkan isi di bawah.

ISI DOKUMEN (potongan berurutan dari awal sampai akhir dokumen):
{context}

PERMINTAAN: {question}

INSTRUKSI PENTING:
- Buat ringkasan menyeluruh: tujuan/isi utama dokumen, lalu poin-poin penting (angka, nama, keputusan, status) dalam bullet points.
- Jika dokumen berupa tabel/data, ringkas strukturnya (kolom/kategori) dan angka atau temuan utamanya.
- JANGAN gunakan pengetahuan umum atau informasi dari luar dokumen, dan jangan mengarang isi yang tidak ada.
- Sebutkan nama file sumbernya.
{answer_language_instruction(language, "- Jawab dalam Bahasa Indonesia")}

RINGKASAN:"""

    def _build_explanation_prompt(
        self,
        context: str,
        question: str,
        conflicts: List,
        sole_source_type: Optional[str] = None,
        language: ReplyLanguage = "id",
    ) -> str:
        """Build prompt untuk explanation queries"""
        vocab = self._source_prompt_vocab(sole_source_type, language)
        extra_instruction = f"\n{vocab['extra_instruction']}" if vocab['extra_instruction'] else ""

        return f"""Anda adalah asisten yang menjawab HANYA berdasarkan {vocab['noun']} yang diberikan.

{vocab['label']} YANG TERSEDIA:
{context}

PERTANYAAN: {question}

INSTRUKSI PENTING:
- Jawab HANYA berdasarkan informasi di {vocab['noun']} yang sudah tersedia.
- Jika {vocab['noun']} memuat informasi yang berkaitan sebagian dengan pertanyaan, jawab sebaik mungkin dari bagian yang relevan itu dan sebutkan bahwa informasinya terbatas.
- Katakan "{vocab['not_found']}" HANYA jika tidak ada satu pun bagian {vocab['noun']} yang berkaitan dengan pertanyaan.
- JANGAN gunakan pengetahuan umum atau informasi dari luar {vocab['noun']}
- Gunakan bullet points atau daftar poin-poin HANYA jika sumber asli menyebutkan daftar terpisah atau beberapa item yang terdaftar secara terpisah. Jika sumber asli berisi narasi atau teks aslinya berupa penjelasan paragraf mengalir terus-menerus (seperti paragraf utuh tunggal), pertahankan sebagai paragraf mengalir/narasi utuh apa adanya tanpa dibuat menjadi poin-poin terpisah.{extra_instruction}
{_off_topic_instruction(language)}
{answer_language_instruction(language, "- Jawab dengan jelas dalam Bahasa Indonesia")}

JAWABAN:"""

    def _validate_and_clean_answer(
        self, answer: str, question: str, results: List[Dict], language: ReplyLanguage = "id"
    ) -> str:
        """Validate dan clean LLM answer"""

        # Check for common LLM failure patterns including prompt echo from small models.
        # The English entries mirror the Indonesian ones ("maaf," / "tidak tahu" /
        # "tidak menemukan") so an English answer is validated the same way.
        failure_patterns = [
            "maaf,", "tidak tahu", "tidak menemukan", "no information",
            "based on the context", "context:", "question:", "answer:",
            "- hanya berdasar", "jangan gunakan", "instruksi penting",
            "jawab hanya berdasarkan", "informasi tidak ditemukan dalam dokumen yang diunggah",
            "sorry,", "i don't know", "i do not know", "could not find", "couldn't find",
        ]

        if any(pattern in answer.lower() for pattern in failure_patterns):
            # Generate answer from best result
            if results:
                best_result = results[0]
                return self._format_result_as_answer(best_result, question, language)

        # Ensure answer is not too short
        if len(answer.split()) < 5:
            if results:
                best_result = results[0]
                return f"{fallback_text(language, 'info_from', source=best_result['source'])}\n{best_result['content']}"

        return answer

    def _format_result_as_answer(self, result: Dict, question: str, language: ReplyLanguage = "id") -> str:
        """Format single result as answer using context-aware snippet extraction"""
        source = result['source']
        confidence = result['confidence']

        if result['type'] == 'database':
            # Format database record
            prefix = fallback_text(language, 'based_on_db', source=source, confidence=f"{confidence:.0%}")
            return f"{prefix}\n\n{result['content']}"
        elif result['type'] == 'pdf':
            snippet = self._extract_relevant_snippet(result['content'], question)
            if not snippet:
                snippet = result['content']
            prefix = fallback_text(language, 'info_from_doc', source=source, confidence=f"{confidence:.0%}")
            return f"{prefix}\n\n{snippet}"
        else:
            return f"{fallback_text(language, 'from', source=source)}\n\n{result['content']}"

    def _extract_relevant_snippet(self, content: str, question: str) -> str:
        """Extract most relevant part of content based on question keywords"""
        # Get question keywords
        question_lower = question.lower()
        keywords = [w for w in question_lower.split() if len(w) > 2]
        
        # Add special keywords for person queries
        if any(kw in question_lower for kw in ['siapa', 'who', 'handle']):
            # Extract names (capitalized words, likely person names)
            import re
            name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            names = re.findall(name_pattern, content)
            if names:
                # Return sentences containing names
                for name in names[:3]:  # Top 3 names
                    for sent in content.split('.'):
                        if name in sent:
                            return sent.strip()[:300]
        
        # Default max length for snippets (increased to prevent truncation)
        max_len = 3000
        
        # For chat content, extract lines with keywords
        if '[' in content and ']' in content:  # Chat format detection
            lines = content.split('\n')
            best_lines = []
            for line in lines:
                line_lower = line.lower()
                score = sum(1 for kw in keywords if kw in line_lower)
                if score > 0:
                    best_lines.append((score, line))
            
            if best_lines:
                best_lines.sort(reverse=True, key=lambda x: x[0])
                return '\n'.join([line for _, line in best_lines[:15]])
        
        # Split content into sentences
        sentences = content.split('.')
        
        # Score each sentence based on keyword matches
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for kw in keywords if kw in sentence_lower)
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
        
        # If found relevant sentence, return it with context
        if best_sentence and best_score > 0:
            # Find position in original text
            start_idx = content.find(best_sentence)
            if start_idx >= 0:
                # Get some context before and after
                context_start = max(0, start_idx - 300)
                context_end = min(len(content), start_idx + len(best_sentence) + 1500)
                snippet = content[context_start:context_end].strip()
                if len(snippet) > max_len:
                    snippet = snippet[:max_len] + "..."
                return snippet
        
        # Fallback to first N characters
        return content[:max_len] + ("..." if len(content) > max_len else "")

    def _extract_direct_answer(self, result: Dict, question: str, language: ReplyLanguage = "id") -> str:
        """Extract direct answer from result for factoid questions"""
        content = result['content']
        source = result['source']
        
        # Try to find phone numbers
        phone_pattern = r'(\+?62[-\s]?\d{3}[-\s]?\d{4}[-\s]?\d{4}|\d{4}|ext\.\s*\d+)'
        phones = re.findall(phone_pattern, content, re.IGNORECASE)
        
        # Try to find emails
        email_pattern = r'[\w\.-]+@[\w\.-]+'
        emails = re.findall(email_pattern, content)
        
        # Detect question type
        question_lower = question.lower()
        
        based_on = fallback_text(language, 'based_on', source=source)

        if 'nomor' in question_lower or 'telepon' in question_lower or 'hp' in question_lower or 'kontak' in question_lower:
            if phones:
                phone_list = ", ".join(set(phones))
                return f"{based_on}\n\n{phone_list}\n\n{fallback_text(language, 'contacts')}\n{content}"

        if 'email' in question_lower:
            if emails:
                email_list = ", ".join(set(emails))
                return f"{based_on}\n\nEmail: {email_list}"

        # Fallback: return relevant snippet
        snippet = self._extract_relevant_snippet(content, question)
        return f"{based_on}\n\n{snippet}"

    def _is_garbled_output(self, text: str) -> bool:
        """Check if output looks garbled/reversed or unhelpful"""
        if not text or len(text) < 10:
            return True
        
        text_lower = text.lower()
        
        # Check for common garbled/unhelpful patterns from flan-t5
        unhelpful_patterns = [
            'pertanyaan pertanyaan',
            'berdasar konteks',
            'jawab pertanyaan',
            'answer the question',
            'based on context',
            'tidak ada informasi',
            'no information',
            'context:',
            'question:',
            'yang bersama yang bersama',  # repetitive garbled output
            'bersama yang bersama',
            'yang berkata terjadi',  # another garbled pattern
            'berkata tidak berkata',
            'data pertanyaan tahun',
            'data pertanyaan tersebut',
            'format yang bersah',
        ]
        
        for pattern in unhelpful_patterns:
            if pattern in text_lower:
                return True
        
        # Check for repetitive patterns (same word repeated 5+ times)
        import re
        words = text_lower.split()
        if len(words) > 5:
            from collections import Counter
            word_counts = Counter(words)
            most_common = word_counts.most_common(1)
            if most_common and most_common[0][1] >= 5 and most_common[0][1] / len(words) > 0.3:
                return True
        
        # Check for reversed text patterns (consonant clusters that don't make sense)
        consonant_pattern = re.compile(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{4,}')
        garbled_words = sum(1 for w in words if consonant_pattern.search(w))
        
        if len(words) > 0 and garbled_words / len(words) > 0.3:
            return True
        
        # If answer is too short and doesn't contain useful info
        if len(words) < 5:
            return True
        
        # Check if answer doesn't contain any expected content (names, data fields)
        # If context had database results but answer has none of the field values, likely garbled
        expected_markers = ['id:', 'name:', 'email:', '@', 'department:', 'position:', 'phone:']
        has_expected = any(marker in text_lower for marker in expected_markers)
        
        # If text looks like repeated gibberish without any data markers
        if not has_expected and 'yang' in text_lower and text_lower.count('yang') >= 3:
            return True
        
        return False

    def _generate_fallback_answer(
        self, hybrid_results: Dict[str, Any], question: str, language: ReplyLanguage = "id"
    ) -> str:
        """Generate a simpler fallback answer when LLM produces garbage"""
        pdf_docs = hybrid_results.get('pdf_documents', [])
        db_results = hybrid_results.get('database_results', {})
        chat_docs = hybrid_results.get('chat_documents', [])
        question_lower = question.lower()
        
        # Extract keywords from question
        keywords = []
        for word in question_lower.split():
            if len(word) > 3:
                keywords.append(word)
        
        # Priority 1: Database results (most structured)
        if db_results:
            for table_name, db_result in db_results.items():
                if db_result.record_count > 0:
                    # Format database results nicely
                    records_text = []
                    for record in db_result.data[:3]:  # Limit to 3 records
                        # Filter out internal fields
                        display_fields = {k: v for k, v in record.items() 
                                         if not k.startswith('_') and k not in ['search_vector', 'relevance_score', 'created_at']}
                        record_str = ", ".join(f"{k}: {v}" for k, v in display_fields.items())
                        records_text.append(f"• {record_str}")
                    
                    result_text = "\n".join(records_text)
                    return f"{fallback_text(language, 'based_on_table', table=table_name)}\n\n{result_text}"
        
        # Priority 2: PDF documents
        if pdf_docs:
            # Find the most relevant document based on question keywords
            best_doc = None
            best_score = 0
            
            for doc in pdf_docs:
                content_lower = doc.page_content.lower()
                # Count how many question keywords appear in the content
                matches = sum(1 for kw in keywords if kw in content_lower)
                doc_score = doc.metadata.get('similarity_score', 0) + (matches * 0.1)
                
                if doc_score > best_score:
                    best_score = doc_score
                    best_doc = doc
            
            if best_doc is None:
                best_doc = pdf_docs[0]
            
            source = best_doc.metadata.get('source', 'dokumen')
            page = best_doc.metadata.get('page', '?')
            
            # Try to extract relevant sentence containing keyword
            content = best_doc.page_content
            # Convert keywords list to question string for _extract_relevant_snippet
            keyword_query = " ".join(keywords)
            relevant_snippet = self._extract_relevant_snippet(content, keyword_query)
            
            prefix = fallback_text(language, 'based_on_page', source=source, page=page)
            if relevant_snippet:
                return f"{prefix}\n\n{relevant_snippet}"
            else:
                return f"{prefix}\n\n{content}"

        # Priority 3: Chat results
        if chat_docs:
            best_chat = chat_docs[0]
            source = best_chat.metadata.get('source', 'chat')
            platform = best_chat.metadata.get('platform', 'unknown')
            prefix = fallback_text(language, 'based_on_chat', source=source, platform=platform)
            return f"{prefix}\n\n{best_chat.page_content}"

        return fallback_text(language, 'no_valid_answer')

    def get_source_info(self, hybrid_results: Dict[str, Any]) -> List[SourceInfo]:
        """Extract source information for response"""
        sources = []

        #PDF sources
        for doc in hybrid_results.get('pdf_documents', []):
            source_info = SourceInfo(
                type="pdf",
                source=doc.metadata.get('source', 'Unknown'),
                confidence=doc.metadata.get('similarity_score', 0),
                preview=self.truncate_context(doc.page_content, max_tokens=50),
                metadata={
                    'page': doc.metadata.get('page', 'Unknown'),
                    'collection_id': doc.metadata.get('collection_id', 'Unknown')
                }
            )

            sources.append(source_info)

         # Database sources
        db_results = hybrid_results.get('database_results', {})
        for table_name, db_result in db_results.items():
            if db_result.record_count > 0:
                source_info = SourceInfo(
                    type="database",
                    source=f"Table: {table_name}",
                    confidence=1.0,
                    preview=f"Menemukan {db_result.reord_count} record",
                    metadata={
                        'table': table_name,
                        'record_count': db_result.record_count
                    }
                )

                sources.append(source_info)

        return sources
