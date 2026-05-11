"""
agnostic/evaluator.py
---------------------
Evaluator  — computes 8 quality metrics for a RAGResult.
EvalScore  — dataclass holding all metric values for one result.

Metrics:
    RR  — Retrieval Relevance    : cosine similarity between question embedding
                                   and mean chunk embedding
    AF  — Answer Faithfulness    : F1 token overlap between answer and context
    AC  — Answer Completeness    : keyword overlap between question and answer
    ROUGE-L                      : LCS-based F1 (answer vs reference/context)
    BLEU-1                       : unigram precision with brevity penalty
    P@K — Precision@K            : fraction of top-K chunks above threshold
    MRR — Mean Reciprocal Rank   : reciprocal rank of first relevant chunk
    CC  — Context Coverage       : unique source count / total chunks (diversity)

Reference modes:
    reference-free  (ground_truth=None)  → ROUGE-L and BLEU-1 computed vs context
    reference-based (ground_truth given) → ROUGE-L and BLEU-1 computed vs gold answer
"""

import math
import re as _re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from agnostic.config import config
from agnostic.indexer import RetrievedChunk


@dataclass
class EvalScore:
    """All metric values for a single evaluated RAGResult."""
    question:             str
    retrieval_relevance:  float
    answer_faithfulness:  float
    answer_completeness:  float
    rouge_l:              float
    bleu_1:               float
    precision_at_k:       float
    mrr:                  float
    context_coverage:     float
    avg_chunk_score:      float
    num_chunks:           int
    total_time:           float
    source_type:          str
    ground_truth:         Optional[str] = None

    @property
    def overall(self) -> float:
        """Average of the 5 primary metrics (RR, AF, AC, ROUGE-L, BLEU-1)."""
        return (
            self.retrieval_relevance
            + self.answer_faithfulness
            + self.answer_completeness
            + self.rouge_l
            + self.bleu_1
        ) / 5


class Evaluator:
    """
    Compute all 8 RAG quality metrics for a RAGResult.

    Usage:
        evaluator = Evaluator()
        score = evaluator.score(result)                     # reference-free
        score = evaluator.score(result, ground_truth="...") # reference-based
    """

    _STOPWORDS = {
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "adalah",
        "ini", "itu", "atau", "ada", "jika", "dalam", "pada", "oleh",
        "the", "is", "are", "of", "in", "to", "and", "a", "an",
        "for", "by", "with", "it", "this", "that", "be", "as", "at",
        "was", "were",
    }

    # ── Tokenization ─────────────────────────────────────────────────────────

    def _tok(self, text: str) -> List[str]:
        text = _re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in text.split()
                if t and t not in self._STOPWORDS and len(t) > 2]

    def _tok_set(self, text: str) -> set:
        return set(self._tok(text))

    # ── Helper computations ───────────────────────────────────────────────────

    def _f1_overlap(self, a: str, b: str) -> float:
        ta, tb = self._tok_set(a), self._tok_set(b)
        if not ta or not tb:
            return 0.0
        inter = ta & tb
        p = len(inter) / len(tb)
        r = len(inter) / len(ta)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def _cosine(self, v1: List[float], v2: List[float]) -> float:
        dot  = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))
        return dot / (mag1 * mag2) if mag1 and mag2 else 0.0

    # ── Individual metric computations ────────────────────────────────────────

    def _calc_retrieval_relevance(self, question: str, chunks: List[RetrievedChunk]) -> float:
        """RR: cosine similarity between question embedding and mean chunk embedding."""
        if not chunks:
            return 0.0
        try:
            from agnostic.indexer import EmbeddingModel
            emb    = EmbeddingModel.get()
            q_emb  = emb.embed_query(question)
            c_embs = emb.embed_documents([c.content for c in chunks])
            n      = len(c_embs[0])
            avg    = [sum(e[i] for e in c_embs) / len(c_embs) for i in range(n)]
            return max(0.0, min(1.0, self._cosine(q_emb, avg)))
        except Exception:
            # Fallback: average FAISS similarity scores
            return sum(c.score for c in chunks) / len(chunks)

    def _calc_answer_faithfulness(self, answer: str, chunks: List[RetrievedChunk]) -> float:
        """AF: F1 token overlap between answer and all context tokens."""
        if not chunks or not answer.strip():
            return 0.0
        context = " ".join(c.content for c in chunks)
        return self._f1_overlap(answer, context)

    def _calc_answer_completeness(self, question: str, answer: str) -> float:
        """AC: fraction of question keywords that appear in the answer."""
        if not answer.strip():
            return 0.0
        qt = self._tok_set(question)
        at = self._tok_set(answer)
        return len(qt & at) / len(qt) if qt else 0.0

    def _calc_precision_at_k(self, chunks: List[RetrievedChunk]) -> float:
        """P@K: fraction of top-K chunks scoring above similarity_threshold."""
        if not chunks:
            return 0.0
        above = sum(1 for c in chunks if c.score >= config.similarity_threshold)
        return above / config.top_k

    def _calc_mrr(self, chunks: List[RetrievedChunk]) -> float:
        """MRR: reciprocal rank of the first chunk above similarity_threshold."""
        for rank, c in enumerate(chunks, 1):
            if c.score >= config.similarity_threshold:
                return 1.0 / rank
        return 0.0

    def _calc_context_coverage(self, chunks: List[RetrievedChunk]) -> float:
        """CC: unique source count / total chunk count (source diversity)."""
        if not chunks:
            return 0.0
        unique_sources = len({c.source for c in chunks})
        return unique_sources / len(chunks)

    def _calc_rouge_l(self, hypothesis: str, reference: str) -> float:
        """ROUGE-L: LCS-based F1 between hypothesis and reference tokens."""
        h_tokens = self._tok(hypothesis)
        r_tokens = self._tok(reference)
        if not h_tokens or not r_tokens:
            return 0.0
        m, n = len(h_tokens), len(r_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if h_tokens[i - 1] == r_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        p = lcs / m
        r = lcs / n
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def _calc_bleu_1(self, hypothesis: str, reference: str) -> float:
        """BLEU-1: unigram precision with brevity penalty."""
        h_tokens = self._tok(hypothesis)
        r_tokens = self._tok(reference)
        if not h_tokens or not r_tokens:
            return 0.0
        ref_counts = Counter(r_tokens)
        hyp_counts = Counter(h_tokens)
        clipped    = sum(min(cnt, ref_counts[tok]) for tok, cnt in hyp_counts.items())
        precision  = clipped / len(h_tokens)
        bp = 1.0 if len(h_tokens) >= len(r_tokens) else math.exp(
            1 - len(r_tokens) / len(h_tokens)
        )
        return bp * precision

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, result, ground_truth: Optional[str] = None) -> EvalScore:
        """
        Compute all 8 metrics for a RAGResult.

        Args:
            result       : RAGResult from AgnosticRAGPipeline.ask()
            ground_truth : optional gold-standard answer string.
                           If None, ROUGE-L and BLEU-1 are computed against context.

        Returns:
            EvalScore dataclass with all metric values
        """
        ref = ground_truth if ground_truth else " ".join(
            c.content for c in result.retrieved_chunks
        )
        return EvalScore(
            question            = result.question,
            retrieval_relevance = self._calc_retrieval_relevance(
                result.question, result.retrieved_chunks),
            answer_faithfulness = self._calc_answer_faithfulness(
                result.answer, result.retrieved_chunks),
            answer_completeness = self._calc_answer_completeness(
                result.question, result.answer),
            rouge_l             = self._calc_rouge_l(result.answer, ref),
            bleu_1              = self._calc_bleu_1(result.answer, ref),
            precision_at_k      = self._calc_precision_at_k(result.retrieved_chunks),
            mrr                 = self._calc_mrr(result.retrieved_chunks),
            context_coverage    = self._calc_context_coverage(result.retrieved_chunks),
            avg_chunk_score     = (sum(c.score for c in result.retrieved_chunks) /
                                   len(result.retrieved_chunks))
                                  if result.retrieved_chunks else 0.0,
            num_chunks          = len(result.retrieved_chunks),
            total_time          = result.total_time,
            source_type         = result.metadata.get("source_type", "?"),
            ground_truth        = ground_truth,
        )

    def run_batch(
        self,
        pipeline,
        questions:        List[str],
        source:           str,
        ground_truths:    Optional[List[str]] = None,
        pg_tables:        Optional[List[str]] = None,
        pg_queries:       Optional[Dict[str, str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        verbose:          bool = True,
        delay_between:    float = 5.0,
    ):
        """
        Run the pipeline for each question and compute all metrics.

        Args:
            pipeline          : AgnosticRAGPipeline instance
            questions         : list of question strings
            source            : source string (folder path or postgresql:// URL)
            ground_truths     : optional list of gold answers (aligned with questions)
            pg_tables         : optional list of PostgreSQL table names to load
            pg_queries        : optional dict of custom SQL queries {label: sql}
            exclude_patterns  : substrings to skip in file names (FolderSourceAdapter)
            verbose           : print per-question metrics
            delay_between     : seconds to wait between questions (anti rate-limit)

        Returns:
            pandas.DataFrame with all metric columns
        """
        import time as _time
        import pandas as pd
        from agnostic.detector import SourceDetector

        rows = []
        print(f"\n{'='*68}")
        print(f"BATCH EVALUATION — {len(questions)} questions")
        mode = "reference-based" if ground_truths else "reference-free"
        print(f"   Source : {SourceDetector.describe(source)}")
        print(f"   Mode   : {mode}")
        if exclude_patterns:
            print(f"   Exclude: {exclude_patterns}")
        if delay_between > 0:
            print(f"   Delay  : {delay_between}s between questions (anti rate-limit)")
        print(f"{'='*68}")

        for i, q in enumerate(questions, 1):
            if i > 1 and delay_between > 0:
                print(f"   ⏳ Waiting {delay_between}s...", flush=True)
                _time.sleep(delay_between)

            gt = ground_truths[i - 1] if ground_truths and i - 1 < len(ground_truths) else None
            print(f"\n[{i}/{len(questions)}] {q[:70]}{'...' if len(q) > 70 else ''}")
            try:
                result = pipeline.ask(q, source=source,
                                      pg_tables=pg_tables, pg_queries=pg_queries,
                                      exclude_patterns=exclude_patterns,
                                      verbose=False)
                s = self.score(result, ground_truth=gt)
                rows.append({
                    "No":                   i,
                    "Question":             q[:55] + ("..." if len(q) > 55 else ""),
                    "Retrieval Relevance":  round(s.retrieval_relevance,  3),
                    "Answer Faithfulness":  round(s.answer_faithfulness,  3),
                    "Answer Completeness":  round(s.answer_completeness,  3),
                    "ROUGE-L":              round(s.rouge_l,              3),
                    "BLEU-1":               round(s.bleu_1,               3),
                    "Precision@K":          round(s.precision_at_k,       3),
                    "MRR":                  round(s.mrr,                  3),
                    "Context Coverage":     round(s.context_coverage,     3),
                    "Overall":              round(s.overall,              3),
                    "Avg Chunk Score":      round(s.avg_chunk_score,      3),
                    "Chunks":               s.num_chunks,
                    "Time (s)":             round(s.total_time,           2),
                    "Source Type":          s.source_type,
                    "Ground Truth":         gt or "",
                    "Answer":               result.answer,
                })
                if verbose:
                    print(f"   RR={s.retrieval_relevance:.3f} | AF={s.answer_faithfulness:.3f} | "
                          f"AC={s.answer_completeness:.3f} | ROUGE-L={s.rouge_l:.3f} | "
                          f"BLEU-1={s.bleu_1:.3f} | P@K={s.precision_at_k:.3f} | "
                          f"MRR={s.mrr:.3f} | Overall={s.overall:.3f}")
            except Exception as e:
                print(f"   Error: {e}")
                rows.append({
                    "No": i, "Question": q[:55],
                    "Retrieval Relevance": 0, "Answer Faithfulness": 0,
                    "Answer Completeness": 0, "ROUGE-L": 0, "BLEU-1": 0,
                    "Precision@K": 0, "MRR": 0, "Context Coverage": 0,
                    "Overall": 0, "Avg Chunk Score": 0, "Chunks": 0,
                    "Time (s)": 0, "Source Type": "error",
                    "Ground Truth": "", "Answer": f"ERROR: {e}",
                })

        df = pd.DataFrame(rows)
        cols_avg = ["Retrieval Relevance", "Answer Faithfulness", "Answer Completeness",
                    "ROUGE-L", "BLEU-1", "Precision@K", "MRR", "Overall"]
        print(f"\n{'-'*68}\nAVERAGES:")
        for col in cols_avg:
            print(f"   {col:<28}: {df[col].mean():.3f}")
        print(f"{'-'*68}")
        return df
