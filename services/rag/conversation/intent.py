"""
Chat intent routing — the deterministic half.

- rule_based_intent() settles the obvious cases for free: whole-message
  small talk and "how do I use DocuLens" questions are "conversation";
  messages that point at the user's own data are "retrieval" and go
  straight to the RAG pipeline. Everything else — including small talk,
  so a "yes" to an offer can still become a search — goes to the
  conversation router (processor.route_message): one LLM call that either
  replies or calls the search_documents function with structured
  arguments. RouteDecision is what that router returns.
- match_collection_titles() / is_summary_request() let both paths scope a
  search to the collection the user named and summarize it properly.
"""

import re
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional

ChatIntent = Literal["conversation", "retrieval"]

_EMOJI_AND_PUNCT = r"[\s!?.,~:;)(\-*^_'\"👋🙏😊😄😁🙂👍✨❤️🔥]*"

# Whole-message small talk. Anchored on purpose: "hi, apa isi dokumen X?"
# must NOT match, it's a retrieval question that happens to start politely.
_SMALL_TALK = {
    # "Yes, do it" — its own kind (checked before "ack") so a "yes" with no
    # pending offer gets its own canned fallback reply (messages.py).
    "affirm": (
        r"(ok(e|ay|ey)?|okk+|ya+|iya+|yes+|yep|yup|yeah|boleh|mau|sure|please|gas+|lanjut(kan)?|"
        r"silak?h?an|go\s+ahead|do\s+it|sounds\s+good|yes\s+please|ok(e)?\s+(boleh|mau|lanjut|gas)|"
        r"boleh\s+(deh|dong|banget)|mau\s+(dong|deh|banget))"
        r"(\s+(dong|deh|ya|please|kak|thanks?|makasih))?"
    ),
    "greeting": (
        r"(hi+|hai+|hay|hey+|halo+|hallo+|helo+|hello+|hola|yo|hei|p|ping|"
        r"(selamat\s+)?(pagi|siang|sore|malam)|good\s+(morning|afternoon|evening|day)|"
        r"assalamu'?alaikum|salam|permisi|misi|greetings)"
        r"(\s+(doculens|bot|kak|min|admin|bang|bro|sis|gan|there|all|semua|everyone|team))?"
    ),
    "thanks": (
        r"(thanks?(\s+(you|a\s+lot|so\s+much|banget|ya|yaa?))?|thank\s+you(\s+(so\s+much|very\s+much))?|"
        r"thx|tq|ty|makasih(\s+(banyak|ya|yaa?|kak|banget))?|terima\s*kasih(\s+(banyak|ya|kak))?|"
        r"trims|tengkyu|nuhun|matur\s+nuwun)"
    ),
    "ack": (
        r"(ok(e|ay|ey)?|okk+|sip+|siap|noted|baik(lah)?|mantap|mantul|keren|nice|great|cool|good|"
        r"perfect|awesome|got\s+it|i\s+see|understood|alright|paham|oh|ohh+|oalah|wow|wkwk+|haha+|hehe+|lol)"
        r"(\s+(thanks?|makasih|ya|deh|dong|sih))?"
    ),
    "farewell": (
        r"(bye+|goodbye|good\s+bye|see\s+(you|ya)(\s+later)?|dadah?|dah|sampai\s+jumpa|"
        r"selamat\s+tinggal|bye\s+bye|cya)"
    ),
    "how_are_you": (
        r"(how\s+are\s+(you|u)(\s+doing)?(\s+today)?|how('?s|\s+is)\s+it\s+going|what'?s\s+up|sup|wassup|"
        r"apa\s+kabar(nya)?|gimana\s+kabar(nya)?|kabar(nya)?\s+(gimana|baik)|apakabar)"
    ),
}
_SMALL_TALK_RE = {
    kind: re.compile(rf"^{_EMOJI_AND_PUNCT}{pattern}{_EMOJI_AND_PUNCT}$", re.IGNORECASE)
    for kind, pattern in _SMALL_TALK.items()
}

# Questions ABOUT DocuLens (features, commands, plans, quota, the assistant).
# Superset of processor.meta_help_patterns (which stays for the "/help"
# command's deterministic answer). Checked AFTER the retrieval signals, so
# "fitur apa saja yang ada di dokumen ini" still goes to retrieval.
_APP_QUESTION = re.compile(
    "|".join([
        r"apa\s+(saja\s+)?yang\s+bisa\s+(kamu|kau|anda|aplikasi\s+ini|doculens)",
        r"apa\s+yang\s+bisa\s+(saya|kita|aku)\s+lakukan(\s+di\s*sini)?",
        r"bisa\s+ngapain(\s+aja)?(\s+(di\s*sini|kamu|doculens))?\s*\??$",
        r"(kamu|doculens|aplikasi\s+ini)\s+bisa\s+(apa|ngapain)",
        r"cara\s+(pakai|pake|menggunakan|make|kerja)\s+(aplikasi|ini|doculens|fitur)",
        r"fitur\s+(apa|yang)\s+(saja\s+)?(yang\s+)?(ada|tersedia)",
        r"command\s+apa\s+(saja\s+)?(yang\s+)?(ada|tersedia)",
        r"apa\s+itu\s+(doculens|gap[\s-]*check|efficient\s+mode|skill|collection)",
        r"(siapa|apa)\s+(kamu|anda|kau)\s*\??$",
        r"(gap[\s-]*check|efficient\s+mode|skills?)\s+(itu\s+)?(apa|gimana|bagaimana)",
        r"\b(jelaskan|explain|tell\s+me\s+about|ceritakan|describe)\s+(tentang\s+|about\s+)?"
        r"(doculens|gap[\s-]*check|efficient\s+mode|skills?|fitur|features?|paket|plans?|commands?|slash\s+commands?)\b",
        r"what\s+(can|could)\s+(you|i|doculens|this\s+app)\s+do",
        r"what\s+(is|are)\s+(doculens|gap[\s-]*check|efficient\s+mode|skills?|your\s+features)",
        r"who\s+are\s+you",
        r"what\s+features",
        r"how\s+does\s+(doculens|this\s+app|gap[\s-]*check|efficient\s+mode)\s+work",
        r"(which|what)\s+commands",
        # plans / pricing / quota — "plan"/"paket"/"upgrade" are everyday words
        # ("project plan", "paket pengadaan", "upgrade the firmware"), so they
        # only count next to a DocuLens plan name or a pricing word.
        r"\b(paket|plan|plans|langganan|subscription)\s+(doculens|free|gratis|individual|team|tim|enterprise|berbayar|langganan)\b",
        r"\b(harga|biaya|price|prices|pricing|cost)\s+(paket|plan|plans|langganan|subscription|doculens|berlangganan)\b",
        r"\b(paket|plan|plans|langganan|subscription)\s+(apa|mana|what|which)\s+(saja\s+)?(yang\s+)?(ada|tersedia|available)?",
        r"\b(which|what)\s+(subscription\s+)?plans?\s+(are|is|do|does)\b",
        r"\b(beda|perbedaan|difference|compare)\s+(antara\s+|between\s+)?(paket|plans?)\b",
        r"\bpricing\b",
        r"\b(upgrade|berlangganan|subscribe)\s+(ke\s+|to\s+)?(paket|plan|langganan|subscription|individual|team|tim|enterprise|akun|account|doculens)\b",
        r"^\s*(cara|how\s+(do|can)\s+i|how\s+to)\s+(upgrade|berlangganan|subscribe)\s*\??\s*$",
        r"\b(kuota|quota|token)\s+(saya|aku|ku|my|kami|tim)\b",
        r"\b(sisa|remaining|left)\s+(kuota|quota|token)",
        r"\bberapa\s+(token|kuota)\b",
        r"\b(how\s+many|how\s+much)\s+(tokens?|quota)",
        r"\b(my|our)\s+(usage|quota|plan|tokens?)\b",
        r"\b(rate\s+limit|limit\s+token|batas\s+(token|kuota|pemakaian))\b",
    ]),
    re.IGNORECASE,
)

# "How do I use feature X" — about DocuLens even when it names a data noun
# ("cara upload dokumen", "how do I upload a file"), so checked BEFORE the
# data-reference rule below. Kept narrow on purpose: a company SOP is full of
# "cara membuat purchase order" / "how do I create a vendor account" — those
# are retrieval questions. So only (a) actions that only make sense inside
# DocuLens (upload, connect, invite, ...) or (b) any verb applied to a
# DocuLens object (collection, source, skill, gap check, ...) count — and
# neither counts when the action points at an outside system ("cara upload
# invoice ke SAP", "how do I connect to the VPN"), see _is_app_howto.
_HOWTO_LEAD = r"\b(((gimana|bagaimana)\s+)?cara|how\s+(do|can|should)\s+(i|we)|how\s+to)\s+"
_APP_ACTION_VERB = (
    r"(upload|unggah|mengunggah|connect|konek|hubungkan|menghubungkan|sambungkan|menyambungkan|"
    r"invite|undang|mengundang|sync|sinkron|sinkronkan)"
)
_APP_OBJECT = (
    r"(collections?|koleksi|sources?|sumber|skills?|folders?|sesi|sessions?|gap[\s-]*check|telegram|"
    r"members?|anggota|workspace|public\s+links?|tautan\s+publik|doculens|efficient\s+mode|token|kuota|quota)"
)
_APP_HOWTO = re.compile(
    "|".join([
        rf"{_HOWTO_LEAD}{_APP_ACTION_VERB}\b",
        rf"{_HOWTO_LEAD}(\w+\s+){{0,3}}?{_APP_OBJECT}\b",
        r"\b(apa|what)\b.*\b(bisa|can|could)\b.*\b(doculens|dilakukan\s+di\s*sini)\b",
    ]),
    re.IGNORECASE,
)
# A how-to aimed at another system ("... ke SAP", "... to the VPN", "... di
# SharePoint") is about that system, not DocuLens.
# Destination prepositions only — "upload dari laptop" / "upload in bulk"
# describe the upload itself, not another system.
_EXTERNAL_TARGET = re.compile(
    r"\b(ke|to|into|di)\s+(the\s+|my\s+|our\s+|a\s+)?"
    r"(?!doculens\b|sini\b|sources?\b|sumber\b|workspace\b|panel\b|sidebar\b|chat\b|composer\b|"
    r"collections?\b|koleksi\b|folders?\b|here\b|this\b|ini\b|database\b|db\b|telegram\b|"
    r"google\b|drive\b|team\b|tim\b)[a-z0-9]",
    re.IGNORECASE,
)


def _is_app_howto(question: str) -> bool:
    if not _APP_HOWTO.search(question):
        return False
    return not _EXTERNAL_TARGET.search(question)

# The message points at the user's own content — always retrieval, even if
# it also mentions a feature ("gap check dokumen ini").
_DATA_REFERENCE = re.compile(
    "|".join([
        r"\b(menurut|according\s+to|berdasarkan|based\s+on)\b",
        r"\b(dokumen|document|documents|file|files|berkas|pdf|halaman|pasal|clause|section|bab|chapter|"
        r"tabel|table|chat\s*log|percakapan|isi\s+dari|contents?\s+of)\b",
    ]),
    re.IGNORECASE,
)

# Verbs/question shapes that ask for information — retrieval unless the
# message was already recognized as a question about DocuLens itself.
_RETRIEVAL_VERB = re.compile(
    "|".join([
        r"\b(ringkas|ringkaskan|rangkum|rangkuman|ringkasan|summari[sz]e|summary|tl;?dr)\b",
        r"\b(cari|carikan|temukan|search|find|look\s+up|lookup)\b",
        r"\b(jelaskan|explain|describe|uraikan|elaborate)\b",
        r"\b(bandingkan|compare|perbandingan)\b",
        r"\b(siapa\s+yang|who\s+(is|handles?|was)|berapa\s+(total|jumlah|harga|banyak)|how\s+many|total\s+\w+)\b",
        r"\b(list|daftar|sebutkan|tampilkan)\b",
    ]),
    re.IGNORECASE,
)


def small_talk_kind(question: str) -> Optional[str]:
    """affirm / greeting / thanks / ack / farewell / how_are_you, or None."""
    q = question.strip()
    if not q or len(q) > 60:
        return None
    for kind, pattern in _SMALL_TALK_RE.items():
        if pattern.match(q):
            return kind
    return None


def is_app_question(question: str) -> bool:
    return bool(_is_app_howto(question) or _APP_QUESTION.search(question))


def rule_based_intent(question: str, collection_titles: Iterable[str] = ()) -> Optional[ChatIntent]:
    """Deterministic routing, or None when the conversation router decides.

    Order matters: whole-message small talk → "how do I use DocuLens"
    questions → pointing at the user's own content (incl. naming one of
    their collections) → other questions about DocuLens → other
    information-seeking verbs."""
    q = question.strip()
    if small_talk_kind(q):
        return "conversation"
    if _is_app_howto(q):
        return "conversation"
    if _DATA_REFERENCE.search(q):
        return "retrieval"
    q_lower = q.lower()
    for title in collection_titles:
        title_lower = (title or "").strip().lower()
        if len(title_lower) >= 4 and title_lower in q_lower:
            return "retrieval"
    if is_app_question(q):
        return "conversation"
    if _RETRIEVAL_VERB.search(q):
        return "retrieval"
    return None


_SUMMARY_REQUEST = re.compile(
    r"\b(ringkas|ringkaskan|rangkum|rangkumkan|rangkuman|ringkasan|summari[sz]e|summary|overview|tl;?dr|"
    r"intisari|garis\s+besar)\b",
    re.IGNORECASE,
)


def is_summary_request(question: str) -> bool:
    """"ringkas X" / "summarize X" — a whole-document overview, which
    retrieval answers from chunks spread across the document rather than
    from the top similarity hits for the word "summarize"."""
    return bool(_SUMMARY_REQUEST.search(question))


# A specific PART of a document ("pasal 5.2", "the access control section",
# "ringkas … tentang cuti") — the user wants that part found, not a
# whole-document overview.
_DOCUMENT_STRUCTURE = re.compile(
    r"\b(pasal|ayat|bagian|bab|klausul|klausa|butir|poin|paragraf|halaman|lampiran|"
    r"section|sections|chapter|clause|clauses|article|paragraph|page|pages|appendix|annex)\b"
    r"|\b(\d+(\.\d+)+|[a-z]\.\d+(\.\d+)*)\b",
    re.IGNORECASE,
)
# "… tentang cuti" / "… about leave" — a topic inside the document. Only
# meaningful in the USER's own words: a model-written search query says
# "summary about the report" for a whole-document summary too.
_DOCUMENT_TOPIC = re.compile(
    r"\b(tentang|mengenai|terkait|soal|perihal|about|regarding|related\s+to|on\s+the\s+topic)\b",
    re.IGNORECASE,
)

# The message is ABOUT more than the one collection it names ("bandingkan X
# dengan dokumen lain", "X vs the SOP") — scoping to X would drop the rest.
_OTHER_DOCUMENTS = re.compile(
    r"\b(bandingkan|perbandingan|dibandingkan|banding|compare|comparison|versus|vs\.?|beda(nya)?|perbedaan|"
    r"difference|differences)\b"
    r"|\b(lain|lainnya|other|others|another|semua|seluruh|all)\s*(dokumen|document|documents|file|files|berkas|collection|collections)?\b"
    r"|\b(dokumen|document|documents|file|files|berkas|collection|collections)\s+(lain|lainnya|other)\b",
    re.IGNORECASE,
)


def mentions_document_part(text: str, include_topics: bool = True) -> bool:
    """A specific part of a document: structure ("pasal 5.2", "section 3")
    and, unless include_topics=False, a topic ("tentang cuti")."""
    text = text or ""
    return bool(_DOCUMENT_STRUCTURE.search(text) or (include_topics and _DOCUMENT_TOPIC.search(text)))


def is_whole_document_summary(question: str, collection_titles: Iterable[str] = ()) -> bool:
    """A summary of a WHOLE document ("ringkas ISO 27001 Policy") — not of
    one part of it ("ringkas pasal 5.2 di ISO 27001 Policy"), which is a
    normal search scoped to that document. The named titles are removed
    first, so a title that itself contains "Section 3" or "v1.2" doesn't
    read as a part of the document."""
    if not is_summary_request(question):
        return False
    remainder = question
    for title in match_collection_titles(question, collection_titles):
        remainder = re.sub(re.escape(title), " ", remainder, flags=re.IGNORECASE)
    return not mentions_document_part(remainder)


def scope_titles_for(question: str, collection_titles: Iterable[str]) -> List[str]:
    """Collections a deterministic-path search should be limited to: the
    ones the message names — unless it's also about other documents
    (comparisons, "dokumen lain", "all files"), where narrowing to the named
    one would leave out the very documents being compared against. Naming
    two or more titles still scopes to exactly those."""
    titles = match_collection_titles(question, collection_titles)
    if len(titles) == 1 and _OTHER_DOCUMENTS.search(question):
        return []
    return titles


def match_collection_titles(question: str, collection_titles: Iterable[str]) -> List[str]:
    """Titles of the user's collections that the message names verbatim
    (case-insensitive, at least 4 characters), longest first — so
    "ringkas ISO 27001 Policy" can be scoped to that one collection."""
    q = question.lower()
    found = {t for t in collection_titles if t and len(t.strip()) >= 4 and t.strip().lower() in q}
    return sorted(found, key=len, reverse=True)


@dataclass
class RouteDecision:
    """What processor.route_message decided for one message."""
    action: Literal["reply", "search"]
    answer: str = ""                          # action == "reply"
    search_query: str = ""                    # action == "search": the function call's args
    search_collection: Optional[str] = None
    search_task: Literal["answer", "summarize"] = "answer"
    model_id: str = ""
    total_tokens: int = 0
    fallback: bool = False
    guard_replaced: bool = False
