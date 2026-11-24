"""
Lens Guide context questions (MS-1182). The server decides when to ask and
what: the role, then the part of the system the user holds, each only when
the folder's documents treat them differently. A model only reads those names
out of the documents (RoutingMixin.extract_facets); it never decides whether
to ask, which it did unreliably (probe, 24 Nov 2026).

The conversation is the only memory (cards are not saved), so the state is
read back from the text every turn: the folder from its path, the request from
the guidance message, the answers from the replies to the server's own
questions. Scenarios: ~/doculens/docs/MS-1182-SCENARIOS.md.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from models import GuideContext
from services.lens_guide.match import collection_ids_in, find_mentioned_folders, folder_label
from services.lens_guide.scope import Scope, ScopeFolder

# The one small search in the folder that the names are read from (plan K6).
BRIEF_QUERY = (
    "peran dan tanggung jawab tim, bagian atau modul sistem, arsitektur, urutan baca; "
    "team roles and responsibilities, system parts, modules, architecture, reading order"
)
BRIEF_PASSAGES = 5
BRIEF_PASSAGE_CHARS = 700
BRIEF_TITLES = 30

MAX_OPTIONS = 5
MAX_OPTION_CHARS = 60

ROLE, PART = "role", "part"
LABELS = {"id": {ROLE: "Peran", PART: "Bagian"}, "en": {ROLE: "Role", PART: "Part"}}
QUESTIONS = {
    "id": {ROLE: "Kamu perannya apa di project {name}?", PART: "Kamu pegang bagian yang mana di project {name}?"},
    "en": {ROLE: "What is your role in {name}?", PART: "Which part of {name} do you work on?"},
}
# Each question as a pattern, any folder name, to find the server's own
# questions in the conversation (also after a reload, when only text is left).
_ASKED = {
    kind: re.compile("|".join(re.escape(q[kind]).replace(r"\{name\}", ".+?") for q in QUESTIONS.values()), re.I)
    for kind in (ROLE, PART)
}

# The messages that ask to be guided (the Home starters, MS-1181, and their
# usual wordings). shortcut: a fixed list, so an unusual wording is answered
# without questions; widen it from MS-1184's conversation set.
# A bare noun ("dokumen onboarding", "jadwal belajar", "what's new in") is a
# question about documents, not a request to be guided.
_GUIDANCE = re.compile(
    r"\b(baru (masuk|di|gabung|join|bergabung|pindah)|pindah ke (project|proyek|modul|tim)|"
    r"(mau|ingin|perlu|minta|butuh) (onboarding|belajar)|mempelajari|pelajari|"
    r"mau (mengerjakan|ngerjain|kerjain)|mulai dari mana|harus baca|baca apa|"
    r"just joined|i'?m new|i am new|new to (this|the) (project|team)|onboard me|"
    r"(want|need) to learn|where (do|should) i start|what should i read)\b",
    re.I,
)
# What the folder cards send (UI folder-part.tsx), and what "Ubah" sends (context-line.tsx).
_PICK = re.compile(r"^(?:ya, )?pakai folder .+$", re.I)
# "Ubah" names the folder too, so it still works once the card is out of the clamped history.
_CHANGE = re.compile(r"^ganti (.+?) jadi (.+?)(?: di folder .+)?$", re.I)
_REJECT = "bukan folder ini"
# Words too common to tell one option from another ("QA Engineer" vs "Data Engineer").
_GENERIC = {"engineer", "manager", "project", "tim", "team", "lead", "senior", "junior", "staff", "bagian", "part"}


@dataclass
class Facets:
    roles: List[str] = field(default_factory=list)
    parts: List[str] = field(default_factory=list)

    def options(self, kind: str) -> List[str]:
        return self.roles if kind == ROLE else self.parts


def is_guidance(text: str) -> bool:
    return bool(_GUIDANCE.search(text or ""))


def is_folder_pick(text: str) -> bool:
    return bool(_PICK.match((text or "").strip()))


def active_folder(question: str, memory: Optional[List[Dict[str, str]]], scope: Scope) -> Optional[ScopeFolder]:
    """The folder of the newest turn (this message first) that names one of
    the scope's folder paths the way the conversation does: "**path**" in a
    reply, "folder path" from the user (what the folder chips send). The
    longest path when a turn names several. None when this message turns the
    folder down."""
    if question.strip().lower().rstrip(".!") == _REJECT:
        return None
    by_label = sorted(((folder_label(f).lower(), f) for f in scope.folders), key=lambda p: -len(p[0]))
    turns = [("user", question)] + [(t.get("role"), t.get("content", "")) for t in reversed(memory or [])]
    for role, text in turns:
        lowered = (text or "").lower()
        for label, folder in by_label:
            if (f"**{label}**" if role == "assistant" else f"folder {label}") in lowered:
                return folder
    return None


def conversation_folder(
    question: str, memory: Optional[List[Dict[str, str]]], scope: Scope,
) -> Tuple[Optional[ScopeFolder], bool]:
    """(folder, confirmed). A message naming exactly one folder by its whole
    name, not already the conversation's folder or shown by the last reply,
    gives that folder unconfirmed: it is confirmed together with the first
    question. Otherwise the newest folder in the conversation, as confirmed."""
    folder = active_folder(question, memory, scope)
    named = find_mentioned_folders(question, scope)
    last_reply = next((t.get("content", "") for t in reversed(memory or []) if t.get("role") == "assistant"), "")
    if len(named) == 1 and named[0] is not folder and folder_label(named[0]).lower() not in (last_reply or "").lower():
        return named[0], False
    return folder, folder is not None


def _turns(question: str, memory: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    return list(memory or []) + [{"role": "user", "content": question}]


def _asked_before(turns: List[Dict[str, str]], i: int) -> Optional[str]:
    """Which of the server's questions the assistant turn right before turn i asked."""
    previous = next((t.get("content") or "" for t in reversed(turns[:i]) if t.get("role") == "assistant"), "")
    return next((kind for kind, pattern in _ASKED.items() if pattern.search(previous)), None)


def guidance_request(question: str, memory: Optional[List[Dict[str, str]]]) -> Optional[int]:
    """Index (in memory + this message) of the message that asked to be
    guided, when this message belongs to that guidance: it is one itself, a
    folder pick, an answer to the server's question, or an "Ubah" change.
    None otherwise: the message goes to the router as before."""
    turns = _turns(question, memory)
    last = len(turns) - 1
    # Only a guided answer offers "Ubah", and only the guidance asks these questions.
    guided_only = bool(_CHANGE.match(question.strip())) or _asked_before(turns, last) is not None
    if not (guided_only or is_guidance(question) or is_folder_pick(question)):
        return None
    found = next((i for i in range(last, -1, -1)
                  if turns[i].get("role") == "user" and is_guidance(turns[i].get("content") or "")), None)
    # The history is clamped to the last 5 questions (MS-237): after a few
    # follow-ups the request is gone, but the guidance still goes on.
    return 0 if found is None and guided_only else found


def request_text(question: str, memory: Optional[List[Dict[str, str]]], start: int, folder: ScopeFolder, language: str) -> str:
    """The guidance request at `start`, or, when it is no longer in the
    clamped history, one written from the folder's name."""
    text = (_turns(question, memory)[start].get("content") or "").strip()
    if is_guidance(text):
        return text
    return (f"Guide me to get started in {folder.path[-1]}" if language == "en"
            else f"Arahkan saya untuk mulai di project {folder.path[-1]}")


def match_option(text: str, options: List[str]) -> Optional[str]:
    """The one option `text` names: as a whole first, else by a telling word
    ("aku developer" -> "Software Developer"). None when none or several."""
    words = set(re.findall(r"[^\W_]+", (text or "").lower()))
    whole = [o for o in options if re.search(rf"(?<!\w){re.escape(o.lower())}(?!\w)", (text or "").lower())]
    if whole:
        return whole[0] if len(whole) == 1 else None
    by_word = [o for o in options if (set(re.findall(r"[^\W_]+", o.lower())) - _GENERIC) & words]
    return by_word[0] if len(by_word) == 1 else None


def known_answers(
    facets: Facets, question: str, memory: Optional[List[Dict[str, str]]], start: int,
) -> Dict[str, str]:
    """What the user has said since the request, newest first: an "Ubah"
    change, a reply to the server's question (taken as written when it names
    no option, e.g. "Belum tahu"), or an option named anywhere."""
    turns = _turns(question, memory)
    known: Dict[str, str] = {}
    for i in range(len(turns) - 1, start - 1, -1):
        if turns[i].get("role") != "user":
            continue
        text = (turns[i].get("content") or "").strip()
        change = _CHANGE.match(text)
        asked = _asked_before(turns, i)
        for kind in (ROLE, PART):
            if kind in known:
                continue
            options = facets.options(kind)
            if change and change.group(1).strip().lower() in {LABELS[lang][kind].lower() for lang in LABELS}:
                value = change.group(2).strip()
                known[kind] = match_option(value, options) or value[:MAX_OPTION_CHARS]
            elif asked == kind and not is_folder_pick(text):
                known[kind] = match_option(text, options) or text[:MAX_OPTION_CHARS]
            elif options and match_option(text, options):
                known[kind] = match_option(text, options)
    return known


def next_question(facets: Facets, known: Dict[str, str]) -> Optional[str]:
    """The role first, then the part; only what the documents tell apart."""
    return next((kind for kind in (ROLE, PART) if len(facets.options(kind)) >= 2 and kind not in known), None)


def question_text(kind: str, folder: ScopeFolder, language: str) -> str:
    return QUESTIONS["en" if language == "en" else "id"][kind].format(name=folder.path[-1])


def guide_context(folder: ScopeFolder, facets: Facets, known: Dict[str, str], language: str) -> List[GuideContext]:
    """The "Konteks yang dipakai" line: the folder's full path (which "Ubah"
    sends back), then each answer with the other choices "Ubah" offers. The
    folder has none: naming another folder changes it."""
    labels = LABELS["en" if language == "en" else "id"]
    items = [GuideContext(label="Project", value=folder_label(folder))]
    items += [GuideContext(label=labels[k], value=known[k], options=facets.options(k)) for k in (ROLE, PART) if k in known]
    return items


# The request alone ("Saya baru di project A.") is no question: searched and
# answered as it is, the answer was "not found" (real GPT, 24 Nov 2026).
_ASK = {
    "id": ("Dari dokumen di folder ini, apa yang perlu saya baca atau lakukan dulu, berurutan, dan kenapa?",
           {ROLE: "Peran saya: {}.", PART: "Bagian yang saya pegang: {}."}),
    "en": ("From the documents in this folder, what should I read or do first, in order, and why?",
           {ROLE: "My role: {}.", PART: "The part I work on: {}."}),
}


def search_question(request: str, known: Dict[str, str], language: str) -> str:
    """What is searched and answered once nothing is left to ask: the request
    as a guidance question, with the answers given."""
    ask, said = _ASK["en" if language == "en" else "id"]
    return " ".join([request.strip().rstrip(".") + ".", ask] + [said[k].format(known[k]) for k in (ROLE, PART) if k in known])


def folder_brief(folder: ScopeFolder, scope: Scope, passages: List[Any]) -> str:
    """What the names are read from: the folder's document titles and the top
    passages of the brief search (LangChain documents)."""
    ids = set(collection_ids_in([folder], scope))
    titles = {c.collection_id: c.title for c in scope.collections if c.collection_id in ids}
    lines = [f"FOLDER: {folder_label(folder)}", "Documents in it and its subfolders:"]
    lines += [f"- {t}" for t in list(titles.values())[:BRIEF_TITLES]] or ["- (none)"]
    for doc in passages[:BRIEF_PASSAGES]:
        text = re.sub(r"\s+", " ", str(getattr(doc, "page_content", "") or "")).strip()[:BRIEF_PASSAGE_CHARS]
        title = titles.get((getattr(doc, "metadata", None) or {}).get("collection_id"), "")
        if text:
            lines.append(f"- [{title}] {text}" if title else f"- {text}")
    return "\n".join(lines)


def clean_options(raw: Any) -> List[str]:
    """Names from the model: short distinct strings, at most MAX_OPTIONS."""
    seen, options = set(), []
    for item in raw if isinstance(raw, list) else []:
        text = re.sub(r"\s+", " ", str(item or "")).strip()[:MAX_OPTION_CHARS]
        if text and text.lower() not in seen:
            seen.add(text.lower())
            options.append(text)
    return options[:MAX_OPTIONS]
