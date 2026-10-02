"""
Reply-language detection — English or Indonesian only (Chinese and other
languages are out of scope for now; anything undetectable falls back to
English, never Indonesian).

Deliberately a marker-word heuristic instead of a library like langdetect:
chat questions are often 1-5 words ("hi", "ISO 27001?", "ok makasih"), which
statistical detectors misclassify badly, and adding a dependency to both
backends for two languages isn't worth it.
"""

import re
from typing import Dict, Iterable, List, Literal, Optional

ReplyLanguage = Literal["en", "id"]

DEFAULT_LANGUAGE: ReplyLanguage = "en"

# Words that only (or overwhelmingly) appear in Indonesian. Kept free of
# tokens that also occur in English text ("data", "file", "ok", "sop", "per")
# so a technical English question doesn't tip into Indonesian.
_ID_MARKERS = frozenset({
    # function words
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "pada",
    "dalam", "adalah", "akan", "sudah", "belum", "juga", "atau", "jika", "kalau",
    "karena", "tapi", "tetapi", "bisa", "dapat", "tidak", "bukan", "ada", "harus",
    "masih", "lagi", "saja", "aja", "sih", "dong", "deh", "nih", "kok", "ya", "yg",
    "gak", "ga", "nggak", "enggak", "tdk", "sama", "seperti", "secara", "oleh",
    "antara", "tentang", "mengenai", "bagi", "agar", "supaya", "lalu", "terus",
    # pronouns
    "saya", "aku", "kamu", "anda", "kita", "kami", "dia", "mereka", "gue", "gw",
    # question words
    "apa", "apakah", "siapa", "kapan", "mana", "dimana", "bagaimana", "gimana",
    "kenapa", "mengapa", "berapa", "kah",
    # common chat verbs
    "tolong", "mohon", "jelaskan", "ringkas", "ringkasan", "rangkum", "cari",
    "carikan", "tampilkan", "sebutkan", "buatkan", "berikan", "lihat", "mau",
    "ingin", "boleh", "pakai", "pake",
    # greetings/small talk
    "halo", "hai", "hallo", "makasih", "terima", "kasih", "trims", "selamat",
    "pagi", "siang", "sore", "malam", "oke", "sip", "siap", "baik", "kabar",
})

# Indonesian nouns that an English speaker may still quote from an
# Indonesian document ("what is pasal 5?") — they count half, so they can
# decide an otherwise markerless message ("isi dokumen") but can't outvote
# real English function words.
_ID_WEAK_MARKERS = frozenset({
    "isi", "dokumen", "berkas", "halaman", "pasal", "bagian", "semua", "setiap",
    "banyak", "jumlah", "fitur", "cara",
})

# Words that only (or overwhelmingly) appear in English.
_EN_MARKERS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "am", "do", "does",
    "did", "have", "has", "had", "can", "could", "would", "should", "will", "shall",
    "of", "to", "in", "on", "for", "with", "about", "from", "by", "at", "and", "or",
    "not", "this", "that", "these", "those", "there", "it", "its", "my", "your",
    "our", "their", "i", "me", "you", "we", "they", "he", "she",
    "what", "whats", "how", "who", "why", "when", "where", "which",
    "please", "explain", "summarize", "summarise", "summary", "find", "show",
    "tell", "list", "give", "describe", "compare", "document", "documents", "page",
    "clause", "section", "all", "any", "many", "much",
    "hello", "hi", "hey", "thanks", "thank", "thx", "bye", "goodbye", "good",
    "morning", "afternoon", "evening", "great", "nice", "okay",
})

_WORD_RE = re.compile(r"[a-zA-Z]+")

# English slang where "ya" means "you" ("see ya") — rewritten before scoring,
# since a bare "ya" is a strong Indonesian marker ("makasih ya").
_ENGLISH_YA = re.compile(r"\b(see|thank|love|catch|got|miss|told)\s+ya\b", re.IGNORECASE)

# English words Indonesian speakers use all the time as small talk ("hi",
# "thanks", "ok", "sorry"). A message made ONLY of these says nothing about
# which language the person actually speaks, so it defers to the
# conversation's language before it counts as English (see
# detect_reply_language).
_LOANWORDS = frozenset({
    "hi", "hii", "hello", "hey", "thanks", "thank", "you", "thx", "ty", "tq", "ok", "okay",
    "yes", "yep", "yup", "yeah", "sure", "please", "pls", "nice", "good", "great", "cool",
    "wow", "bye", "sorry", "noted", "done", "lol", "morning", "night", "so", "much", "a", "lot",
    "very", "guys", "all", "there", "bro", "sis",
})


def _score(text: str) -> Optional[ReplyLanguage]:
    """Decisive language of `text`, or None when there's no signal either way.

    Indonesian wins ties and code-mixed sentences ("tolong explain pasal 5"):
    Indonesian speakers routinely borrow English nouns/verbs, while English
    speakers essentially never borrow Indonesian function words — so a
    single Indonesian marker against up to twice as many English ones still
    reads as an Indonesian speaker."""
    words = [w.lower() for w in _WORD_RE.findall(_ENGLISH_YA.sub(r"\1 you", text))]
    if not words:
        return None
    id_hits = sum(
        1.0 if w in _ID_MARKERS or (len(w) > 5 and w.endswith("nya"))
        else 0.5 if w in _ID_WEAK_MARKERS
        else 0.0
        for w in words
    )
    en_hits = sum(1 for w in words if w in _EN_MARKERS)
    if id_hits == 0 and en_hits == 0:
        return None
    if id_hits > 0 and id_hits * 2 >= en_hits:
        return "id"
    return "en"


def detect_reply_language(
    question: str,
    memory: Optional[Iterable[Dict[str, str]]] = None,
) -> ReplyLanguage:
    """Language the answer to `question` should be written in.

    Order: the question itself → the most recent earlier user turn that has
    a clear language (so "ok" or "ISO 27001?" mid-conversation keeps the
    conversation's language) → English.

    Exception: a message made only of loanword small talk ("thanks", "hi",
    "ok sure") checks the conversation FIRST — an Indonesian user who types
    "thanks" mid-conversation still gets Indonesian; with no history it's
    English as usual."""
    words = [w.lower() for w in _WORD_RE.findall(question)]
    loanword_only = bool(words) and len(words) <= 4 and all(w in _LOANWORDS for w in words)

    def from_memory() -> Optional[ReplyLanguage]:
        for turn in reversed(list(memory or [])):
            if turn.get("role") != "user":
                continue
            found = _score(str(turn.get("content", "")))
            if found:
                return found
        return None

    if loanword_only:
        return from_memory() or _score(question) or DEFAULT_LANGUAGE
    return _score(question) or from_memory() or DEFAULT_LANGUAGE


def detect_text_language(text: str) -> ReplyLanguage:
    """Dominant language of a longer body of text (e.g. a company document
    for Gap Check). Same markers, compared as totals; no signal → English."""
    return _score(text) or DEFAULT_LANGUAGE


def answer_language_instruction(language: ReplyLanguage, id_line: str) -> str:
    """The prompt line that pins the answer language.

    `id_line` is each prompt builder's own original Indonesian wording, kept
    byte-for-byte so Indonesian questions behave exactly as before. The
    English line is explicit about the prompt itself being Indonesian —
    without that, Gemini tends to follow the instruction language. The
    English line reuses `id_line`'s own list marker ("- " or "5. ") so it
    still fits the surrounding numbered/bulleted instruction list."""
    if language == "id":
        return id_line
    marker = re.match(r"^\s*(\d+\.\s*|-\s*)?", id_line).group(0)
    return (
        f"{marker}IMPORTANT: Write the ENTIRE answer in English, even though these "
        "instructions and the source content may be in Indonesian."
    )


def language_name(language: ReplyLanguage) -> str:
    return "Indonesian (Bahasa Indonesia)" if language == "id" else "English"


def clamp_language(value: Optional[str]) -> ReplyLanguage:
    return "id" if value == "id" else "en"


__all__: List[str] = [
    "ReplyLanguage",
    "DEFAULT_LANGUAGE",
    "detect_reply_language",
    "detect_text_language",
    "answer_language_instruction",
    "language_name",
    "clamp_language",
]
