"""Context compression for Efficient Mode (MS-247).

Reimplements from scratch, in Python, the two techniques that actually
explain measurable token reduction in github.com/juliusbrussee/caveman's
public engine design (see engine/compressors/{text,tabular,redundancy}.go
in that repo) — NOT a port of any of caveman's BSL-licensed source, just
the same published ideas rebuilt independently:

1. Query-biased section pruning *within* one long retrieved chunk — a
   chunk keeps its first/last section, any heading, any section matching
   a small curated "important" keyword list (compliance-relevant terms,
   given this app's Gap Check feature), and any section whose BM25 score
   against the user's `question` clears a relative threshold. Everything
   else is replaced by a short "N bagian dihilangkan" marker. This is the
   piece the previous MVP (dedup/whitespace/filler-only) was missing —
   without it, a single-source answer (nothing to cross-chunk-dedup
   against) always measured 0% reduction, even though most of a long PDF
   passage is typically not relevant to any one question.
2. Cross-chunk redundancy — a chunk is dropped only when another
   *already-kept* chunk's vocabulary (words AND numbers, case-folded,
   digits always kept intact — an earlier version masked digit runs to
   "#" to catch duplicates differing only by an incidental value, but a
   code review found realistic content in this app's shape collides
   under that scheme: e.g. two DB records/PDF sentences describing the
   same entity at two different points in time, identical except for one
   price/date, easily clear the old word-count bar and were wrongly
   treated as the same content — see `_vocabulary`'s docstring) already
   covers >=90% of its own. Ties are broken by keeping whichever chunk
   retrieval scored higher (`confidence`, already computed upstream in
   processor.merge_and_rank_results — a real relevance-to-question
   signal we already have, not something reimplemented here), not just
   whichever came first.

Both are meaning-preserving in the sense that nothing is silently
reworded — content is either kept verbatim or elided with an explicit
marker naming what was dropped. The BM25 formula below is the standard
public Okapi BM25 (Robertson & Walker, 1994), not caveman's own
implementation.

A third function, compress_memory(), applies a deliberately narrower
transform to the conversation-memory window (see its own docstring for
why BM25-style relevance pruning is unsafe there specifically).

compress_context() and compress_memory() are both no-op-safe pure
functions: called only when a caller explicitly opts in
(efficient_mode=True in processor.generate_hybrid_answer), never on the
default path.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# --- cross-chunk redundancy -------------------------------------------------

_WORD_OR_DIGITS_RE = re.compile(r"[a-zA-Z]+|\d+")
# Share of a dropped chunk's vocabulary a kept chunk must already cover.
# Conservative on purpose: a missed dedup only costs tokens, a wrong one
# changes the answer.
_REDUNDANCY_CONTAINMENT = 0.9
# Caps the vocabulary built per chunk so one pathologically long chunk
# can't make the containment check quadratic.
_MAX_VOCAB_TOKENS = 128


# NOTE: two earlier versions of this file got this wrong, in two
# different ways a code review (see repo history) caught before this one
# shipped:
#
# v1 masked digit runs to a single "#" once a unit had >= 1 word token
# (caveman's own default — tuned for their target case of thousands of
# near-identical log/config lines, where an occasional false collapse is
# diluted and cheap). That collapsed realistic content in OUR shape (few,
# individually high-value chunks): "Harga produk itu adalah 1500000
# rupiah untuk kategori furniture kantor." vs the same sentence with
# 1750000 — both mask down to an IDENTICAL vocabulary once every digit
# run becomes "#", so one was silently dropped.
#
# v2 raised the masking bar to >= 6 word tokens and, separately, simply
# stopped masking digits at all — but that alone still isn't enough: with
# digits kept as literal tokens, the SAME example above produces two
# 10-token vocabularies differing in exactly one token (the price), and
# 9/10 shared tokens still clears a 90% containment bar. A single
# threshold on the WHOLE vocabulary can't protect a specific differing
# fact once the surrounding prose is long enough to dilute it.
#
# The actual fix: numbers are usually the load-bearing, information-
# carrying part of this app's content (a price, a date, a quantity — the
# answer itself), so they are never allowed to be "mostly the same" the
# way prose vocabulary can be. `_vocabulary` now returns WORDS and
# NUMBERS separately; `_represented_by` requires (a) >=90% containment on
# WORDS (still catches paraphrase-level near-duplicate prose) AND (b)
# every number in the candidate also appears in the kept text's numbers
# (a single differing number blocks the dedup outright, however long the
# surrounding sentence is). This only gives up catching duplicates that
# differ SOLELY by an incidental, non-informative number (e.g. a log line
# repeated with a different timestamp) — not this app's dominant
# redundancy case, which is the same fact restated verbatim (numbers
# included) across sources, still caught here since identical text
# produces identical word AND number sets either way.


def _vocabulary(text: str) -> Tuple[frozenset, frozenset]:
    """Reduce text to (words, numbers) — case-folded word tokens and the
    literal digit-runs it contains, kept as two separate sets so a
    containment check can treat them differently (see the note above)."""
    words: set = set()
    numbers: set = set()
    for m in _WORD_OR_DIGITS_RE.finditer(text.lower()):
        tok = m.group(0)
        target = numbers if tok.isdigit() else words
        target.add(tok)
        if len(words) >= _MAX_VOCAB_TOKENS and len(numbers) >= _MAX_VOCAB_TOKENS:
            break
    return frozenset(words), frozenset(numbers)


def _represented_by(profile: Tuple[frozenset, frozenset], classes: List[Tuple[frozenset, frozenset]]) -> bool:
    words, numbers = profile
    if not words and not numbers:
        return True  # nothing to lose
    need = math.ceil(_REDUNDANCY_CONTAINMENT * len(words)) if words else 0
    for cwords, cnumbers in classes:
        if numbers and not numbers.issubset(cnumbers):
            continue  # a differing number is decisive on its own
        hit = sum(1 for tok in words if tok in cwords)
        if hit >= need:
            return True
    return False


# --- BM25 (query-biased section retention) ----------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_scores(query: str, docs: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """Standard Okapi BM25 — used to score each SECTION of one retrieved
    chunk against the user's question, cheaply and locally (no LLM
    call, no external service)."""
    query_terms = _tokenize(query)
    if not query_terms or not docs:
        return [0.0] * len(docs)
    doc_tokens = [_tokenize(d) for d in docs]
    doc_lens = [len(t) for t in doc_tokens]
    avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
    n_docs = len(docs)
    unique_query_terms = set(query_terms)
    df = {term: sum(1 for tokens in doc_tokens if term in tokens) for term in unique_query_terms}
    idf = {term: math.log((n_docs - n + 0.5) / (n + 0.5) + 1) for term, n in df.items()}
    scores = []
    for tokens, dl in zip(doc_tokens, doc_lens):
        tf = Counter(tokens)
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            denom = f + k1 * (1 - b + b * (dl / avgdl if avgdl else 1.0))
            score += idf[term] * (f * (k1 + 1)) / denom
        scores.append(score)
    return scores


# --- section splitting / heading / importance -------------------------------

# Curated deliberately small — anything borderline is left alone rather
# than risk stripping something load-bearing. Mixed ID/EN since this
# app's documents/answers are bilingual, and several terms track this
# app's own Gap Check / compliance domain (sanksi/denda/wajib/dilarang).
_IMPORTANT_RE = re.compile(
    r"\b(ERROR|WARNING|IMPORTANT|NOTE|CATATAN|PENTING|PERHATIAN|WAJIB|"
    r"DILARANG|SANKSI|DENDA|KEPUTUSAN|KESIMPULAN|REKOMENDASI|DEADLINE|"
    r"BATAS\s+WAKTU)\b",
    re.IGNORECASE,
)

_MIN_PRUNABLE_LEN = 400
_QUERY_RELEVANCE_THRESHOLD = 0.25
_KEEP_HEAD = 1
_KEEP_TAIL = 1


def _split_sections(text: str) -> List[str]:
    if "\n\n" in text:
        parts = [s for s in text.split("\n\n") if s.strip()]
    else:
        parts = [s for s in text.split("\n") if s.strip()]
    return parts or [text]


def _is_heading(section: str) -> bool:
    s = section.strip()
    if s.startswith("#"):
        return True
    if len(s) > 96 or any(ch in s for ch in ".!?"):
        return False
    words = s.split()
    return 0 < len(words) <= 8


def _prune_sections(body: str, question: str) -> Tuple[str, bool]:
    """Shrink one long chunk's body down to its head/tail, headings,
    "important" sections, and sections that score well against
    `question` — the rest is replaced by a "N bagian dihilangkan"
    marker. No-op (returns the body unchanged, pruned=False) when the
    chunk is short, has too few sections to matter, or nothing would
    actually be dropped."""
    if len(body) < _MIN_PRUNABLE_LEN:
        return body, False
    sections = _split_sections(body)
    if len(sections) <= _KEEP_HEAD + _KEEP_TAIL + 1:
        return body, False

    keep = [False] * len(sections)
    for i, section in enumerate(sections):
        if i < _KEEP_HEAD or i >= len(sections) - _KEEP_TAIL:
            keep[i] = True
        elif _is_heading(section):
            keep[i] = True
        elif _IMPORTANT_RE.search(section):
            keep[i] = True

    if question:
        scores = _bm25_scores(question, sections)
        max_score = max(scores) if scores else 0.0
        if max_score > 0:
            for i, score in enumerate(scores):
                if not keep[i] and score / max_score >= _QUERY_RELEVANCE_THRESHOLD:
                    keep[i] = True

    if all(keep):
        return body, False

    out: List[str] = []
    dropped = 0
    for i, section in enumerate(sections):
        if keep[i]:
            if dropped:
                out.append(f"[...{dropped} bagian dihilangkan...]")
                dropped = 0
            out.append(section.strip())
        else:
            dropped += 1
    if dropped:
        out.append(f"[...{dropped} bagian dihilangkan...]")

    new_body = "\n\n".join(out)
    if len(new_body) >= len(body):
        return body, False
    return new_body, True


# --- public API --------------------------------------------------------------

@dataclass
class ContextPart:
    """One retrieved chunk, exactly as processor.py already has it in
    hand when it builds `context_parts` — just not yet flattened into a
    single formatted string, so the compressor still has each chunk's
    `confidence` (retrieval's own query-relevance score, from
    merge_and_rank_results) available."""

    header: str        # e.g. "[Sumber 1 — PDF: kontrak.pdf]:"
    body: str          # the content snippet itself
    confidence: float  # retrieval's relevance score for this chunk


@dataclass
class CompressionStats:
    parts_before: int
    parts_after: int
    deduplicated: int
    sections_pruned: int


def compress_context(parts: List[ContextPart], question: str = "") -> Tuple[str, CompressionStats]:
    """Compress a list of retrieved chunks into the same
    `"\\n\\n---\\n\\n"`-joined string processor.py already builds,
    applying (1) intra-chunk section pruning, independently per chunk,
    then (2) cross-chunk redundancy dedup, keeping the higher-confidence
    chunk of any near-duplicate pair. Output preserves original chunk
    order — confidence only decides which chunk of a duplicate PAIR
    survives, not where surviving chunks are placed (placement must stay
    deterministic/order-stable, or a later cache/replay would see a
    different prompt for the same inputs)."""
    n = len(parts)
    if n == 0:
        return "", CompressionStats(0, 0, 0, 0)

    pruned_bodies: List[str] = []
    sections_pruned = 0
    for part in parts:
        new_body, was_pruned = _prune_sections(part.body, question)
        pruned_bodies.append(new_body)
        if was_pruned:
            sections_pruned += 1

    # Decide which chunk of each near-duplicate class survives by walking
    # confidence-descending (ties broken by original position), but the
    # final assembly below stays in original order regardless.
    order = sorted(range(n), key=lambda i: (-parts[i].confidence, i))
    keep = [False] * n
    classes: List[Tuple[frozenset, frozenset]] = []
    for i in order:
        vocab = _vocabulary(pruned_bodies[i])
        if _represented_by(vocab, classes):
            keep[i] = False
        else:
            keep[i] = True
            classes.append(vocab)

    out_parts: List[str] = []
    deduplicated = 0
    for i, part in enumerate(parts):
        if keep[i]:
            out_parts.append(f"{part.header}\n{pruned_bodies[i]}")
        else:
            deduplicated += 1
    if deduplicated:
        out_parts.append(f"… {deduplicated} sumber duplikat dihilangkan (efficient mode) …")

    stats = CompressionStats(
        parts_before=n,
        parts_after=len(out_parts),
        deduplicated=deduplicated,
        sections_pruned=sections_pruned,
    )
    return "\n\n---\n\n".join(out_parts), stats


# --- conversation memory (last-5-chats window) --------------------------------
#
# Deliberately a *narrower* transform than compress_context above.
#
# Memory exists specifically so a follow-up like "yang tadi gimana?" can be
# resolved against recent turns — but that exact vagueness means a
# BM25-against-the-current-question score (the technique compress_context
# uses to decide what's relevant) would score the very turn that answers
# "yang tadi" the LOWEST, since a vague follow-up shares almost no
# vocabulary with what it refers to. Applying section-relevance pruning
# here would therefore risk dropping the one turn a reference needs to
# resolve — the opposite of what compress_context's BM25 pass is safe to
# do on retrieved documents. So this never drops a turn for being "less
# relevant": it only (1) cleans whitespace/boilerplate filler phrases, and
# (2) removes a turn that is a near-exact repeat of another, keeping
# whichever copy is more RECENT (recency, not confidence, is what matters
# for reference resolution — there is no retrieval confidence score for a
# conversation turn anyway).

_FILLER_PHRASES = [
    "perlu dicatat bahwa",
    "perlu diketahui bahwa",
    "seperti yang disebutkan sebelumnya",
    "seperti yang telah disebutkan",
    "sebagaimana telah dijelaskan",
    "sebagai catatan tambahan",
    "please note that",
    "it is important to note that",
    "it should be noted that",
    "as mentioned above",
    "as previously mentioned",
]
_FILLER_RE = re.compile("|".join(re.escape(p) for p in _FILLER_PHRASES), re.IGNORECASE)
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _clean_turn_text(text: str) -> str:
    text = _FILLER_RE.sub("", text)
    text = _TRAILING_SPACE_RE.sub("\n", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


@dataclass
class MemoryCompressionStats:
    turns_before: int
    turns_after: int
    deduplicated: int


def compress_memory(turns: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], MemoryCompressionStats]:
    """Clean and dedup conversation-memory turns (see module note above
    for why this is narrower than compress_context). `turns` is the same
    list of `{"role": ..., "content": ...}` dicts processor.py already
    formats into the RIWAYAT PERCAKAPAN block — this returns an
    equal-or-shorter list in the same shape and order."""
    n = len(turns)
    if n == 0:
        return [], MemoryCompressionStats(0, 0, 0)

    cleaned = [{**turn, "content": _clean_turn_text(str(turn.get("content", "")))} for turn in turns]

    # Most-recent-first so, of any near-duplicate pair, the later
    # (more recent) turn is the one kept as the class representative.
    order = list(range(n - 1, -1, -1))
    keep = [False] * n
    classes: List[Tuple[frozenset, frozenset]] = []
    for i in order:
        vocab = _vocabulary(cleaned[i]["content"])
        if _represented_by(vocab, classes):
            keep[i] = False
        else:
            keep[i] = True
            classes.append(vocab)

    out = [cleaned[i] for i in range(n) if keep[i]]
    deduplicated = n - len(out)
    return out, MemoryCompressionStats(turns_before=n, turns_after=len(out), deduplicated=deduplicated)
