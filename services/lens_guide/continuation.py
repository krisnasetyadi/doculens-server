"""MS-1183: continue a reading list from chat text, without stored progress."""
import base64
import json
import re
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urlparse

from models import Guide, GuideStep
from services.rag.conversation.limits import MAX_MEMORY_CHATS, MAX_MEMORY_MESSAGES
from services.lens_guide.answer import NEAREST_WHY, casual_to_plain, is_guidance, location, source_detail, step_text

_STEP = re.compile(r"^([a-z]|\d+)\. \*\*(.+?)\*\* \((.+)\): (.+)$", re.M)
_LINK = re.compile(r"\[[^\]]+\]\(([^\s)]+)\)")
_FILE_PATH = re.compile(r"/api/v1/files/([^/]+)/([^/]+)")
_PAGE = re.compile(r"page=(\d+)")
_MINUTES = re.compile(r"± (\d+) (?:min|mnt)")
_ORIGIN = re.compile(r"; (?:dari|from) (.+)$")
# Matched against casual_to_plain(text): lower case, words only, chat spellings
# replaced ("udah" -> "sudah", "kubaca" -> "ku baca", "&" -> "dan").
_NEXT = re.compile(
    r"^(?:(?:ok|oke|okay|sip|baik|yuk)\s+)?(?:lanjut|lanjutkan|next|terus|berikutnya)"
    r"(?:\s+(?:dong|ya|yuk|kak|please|aja|lagi))*$"
)
_READ = re.compile(
    r"^(?:(?:aku|saya|i)\s+)?(?:(?:ve|have)\s+)?(?:already\s+)?"
    r"(?:(?:sudah|selesai)\s+(?:(?:ku|aku|saya)\s+)?(?:baca|membaca)|read|finished|done\s+with)\s+(.+)$"
)
# "sudah" alone: the step the last reply showed is done. Right after the whole
# list it could mean anything, so it is not guessed there.
_DONE = re.compile(r"^(?:(?:ok|oke|sip)\s+)?(?:sudah|selesai|done|beres)(?:\s+(?:baca|dong|ya|kak|nih))*$")
_KEY = re.compile(r"\b(?:[a-z]|\d+)\b")
_ALL = re.compile(r"\b(?:semua|semuanya|all|everything)\b")
_RANGE = re.compile(r"\b([a-z]|\d+)\s*(?:sampai|hingga|through|to|-)\s*([a-z]|\d+)\b", re.I)
_LOCATION_REFERENCE = re.compile(
    r"\b(?:halaman|hlm|hal|page|p|bab|slide|sheet|bagian)\b\s*[.:]?\s*"
    r"(?:[a-z]|\d+)\b(?:\s*(?:sampai|hingga|through|to|-)\s*(?:[a-z]|\d+)\b)?", re.I,
)
# A step the server showed on its own ("Berikutnya: **c.") or on request ("Langkah **a.").
_SHOWN = re.compile(r"^(?:(?:Berikutnya|Next): |Langkah |Step )\*\*([a-z]|\d+)\.", re.M)

# The history the client sends is the last five questions, so a list shown long ago is gone from it.
# Every reply of this module therefore carries the list and the keys already done as a markdown
# reference definition, which renders as nothing. It is client text like the history itself: the
# list is re-read with the same file checks, and keys outside it are ignored.
_STATE = re.compile(r'^\[lens-list\]: # "([A-Za-z0-9_-]+)"[ \t]*$', re.M)
_STATE_LINE = re.compile(r"^\[lens-list\]:[^\n]*(?:\n|$)", re.M)
MAX_STATE_CHARS = 12000
MAX_CTX_CHARS = 60  # the longest role or part the server accepts as an answer (clarify.MAX_OPTION_CHARS)
_CTX_KEYS = ("folder", "role", "part")


def _recent_turns(memory) -> List[Dict[str, str]]:
    """The history window of every chat (last five questions), plus, for a Lens Guide chat, the newest
    reply carrying a list even when it is older: side questions must not make the list disappear."""
    turns = [{"role": m.role, "content": m.content} for m in memory or []]
    questions = [i for i, t in enumerate(turns) if t["role"] == "user"]
    start = questions[-MAX_MEMORY_CHATS] if len(questions) > MAX_MEMORY_CHATS else 0
    recent = turns[start:][-MAX_MEMORY_MESSAGES:]
    older = turns[:len(turns) - len(recent)]
    pinned = next((t for t in reversed(older) if t["role"] == "assistant" and _STATE_LINE.search(t["content"])), None)
    return ([pinned] if pinned else []) + recent


def strip_list_state(text: str) -> str:
    """The reply without its carried state: what a model prompt and Copy should see."""
    return _STATE_LINE.sub("", text or "").rstrip()


def carries_list_state(text: str) -> bool:
    return bool(_STATE_LINE.search(text or ""))


def _carry(lines: List[str], done: Set[str], ctx: Optional[Dict[str, str]] = None) -> str:
    """The state line to append to a reply ("" when the list is too large to carry). `ctx` is the
    folder, role and part the list was made for (what guide_turn would otherwise ask again)."""
    raw = json.dumps({"v": 1, "list": "\n".join(lines), "done": sorted(done), "ctx": ctx or {}},
                     ensure_ascii=False, separators=(",", ":"))
    token = base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")
    return f'\n\n[lens-list]: # "{token}"' if len(token) <= MAX_STATE_CHARS else ""


def _read_state(content: str) -> Optional[Tuple[str, Set[str], Dict[str, str]]]:
    """(list text, keys done, context) a reply carried; None when absent, malformed or from another version."""
    found = _STATE.search(content)
    if not found or len(found[1]) > MAX_STATE_CHARS:
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(found[1] + "=" * (-len(found[1]) % 4)))
        text, done = data["list"], data["done"]
        if data["v"] != 1 or not isinstance(text, str) or not isinstance(done, list) or not all(isinstance(k, str) for k in done):
            return None
        raw_ctx = data.get("ctx")
        ctx = {k: v for k, v in (raw_ctx if isinstance(raw_ctx, dict) else {}).items()
               if k in _CTX_KEYS and isinstance(v, str) and len(v) <= MAX_CTX_CHARS}
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return text, set(done), ctx


def _real_items(text: str):
    """The list items of a text, none when they are only nearest-document pointers (older answers
    wrote those as items; they are not a list to work through)."""
    matches = list(_STEP.finditer(text))
    return [] if all(m[4].strip() in NEAREST_WHY.values() for m in matches) else matches


def _latest_list(turns):
    """(items, index, done, context): the newest reading list, steps or stages a reply showed, in its
    text or as carried by a later reply, the keys that reply says are done, and its folder, role, part."""
    for i in range(len(turns) - 1, -1, -1):
        if turns[i]["role"] != "assistant":
            continue
        text, done, ctx = _read_state(turns[i]["content"]) or (turns[i]["content"], set(), {})
        matches = _real_items(text)
        if matches:
            return matches, i, done, ctx
    return [], None, set(), {}


def carry_list(answer: str, ctx: Optional[Dict[str, str]] = None) -> str:
    """A list answer with its state, so a later "lanjut" finds it however many questions came between."""
    matches = _real_items(answer)
    return answer + _carry([m[0] for m in matches], set(), ctx) if matches else answer


def list_context(guide: Guide) -> Dict[str, str]:
    """The folder, role and part of a guide answer, from its "Konteks yang dipakai" line."""
    labels = {"project": "folder", "peran": "role", "role": "role", "bagian": "part", "part": "part"}
    return {labels[c.label.casefold()]: c.value for c in guide.context or [] if c.label.casefold() in labels}


def carried_answers(memory, folder: str) -> Dict[str, str]:
    """The role and part given earlier in this guidance, for `folder`, when the messages that
    carried them have left the history. Only the latest list's context, and only for its own folder."""
    ctx = _latest_list(_recent_turns(memory))[3]
    return {k: ctx[k] for k in ("role", "part") if ctx.get("folder") == folder and k in ctx}


def has_step_list(memory) -> bool:
    """Whether "continue" can mean anything in this conversation (the router only gets the tool then)."""
    return _latest_list(_recent_turns(memory))[1] is not None


def _step_from_text(match, allowed: Dict[str, Set[str]], base_url: str, language: str) -> Optional[GuideStep]:
    """One list item as the reply wrote it, linked only to files the user may still open."""
    sources = []
    for url in _LINK.findall(match[3]):
        parsed = urlparse(url)
        path = _FILE_PATH.fullmatch(parsed.path)
        if not path:
            continue
        collection_id, name = unquote(path[1]), unquote(path[2])
        if name not in allowed.get(collection_id, set()):
            continue
        page = _PAGE.fullmatch(parsed.fragment)
        sources.append(source_detail(name, collection_id, page[1] if page else None, base_url))
    if not sources:
        # Never turn an inaccessible or ambiguous file name into a link.
        return None
    minutes = _MINUTES.search(match[3])
    origin = _ORIGIN.search(match[3])
    return GuideStep(key=match[1], title=match[2], why=match[4],
                     file="; ".join(dict.fromkeys(s.file_name for s in sources)),
                     where="; ".join(location(s, language) for s in sources),
                     source=sources[0], sources=sources if len(sources) > 1 else None,
                     minutes=int(minutes[1]) if minutes else None,
                     origin=origin[1] if origin else None)


def _written_keys(text: str) -> Set[str]:
    """Step keys a message names. File locations ("hlm 2") describe what was read, not keys."""
    return set(_KEY.findall(_LOCATION_REFERENCE.sub(" ", text)))


def _read_keys(text: str, keys: List[str]) -> Set[str]:
    """Keys a "sudah baca ..." message reports read: named, "semua", or a range ("a sampai c")."""
    text = _LOCATION_REFERENCE.sub(" ", text)
    done = set(keys) if _ALL.search(text) else set()
    done.update(key for key in _KEY.findall(text) if key in keys)
    for first, last in _RANGE.findall(text):
        if first in keys and last in keys and keys.index(first) <= keys.index(last):
            done.update(keys[keys.index(first):keys.index(last) + 1])
    return done


def _one_step_reply(label: str, step: GuideStep, language: str, carry: str) -> Tuple[str, Guide]:
    """A single item, written unlike the full list so a later "lanjut" does not mistake it for the list."""
    lead = f"{label}**{step.key}. {step.title}**"
    detail = step_text(step, language).split("** (", 1)[1]
    return f"{lead} ({detail}{carry}", Guide(lead=lead, steps=[step])


def _list_gone(turns, language: str) -> Optional[Tuple[str, Guide]]:
    """A step was just shown but the list it came from is no longer in the history: say so
    (the model would invent the next step)."""
    last = next((t["content"] for t in reversed(turns) if t["role"] == "assistant"), "")
    if not _SHOWN.search(last):
        return None
    text = ("The reading list is no longer in this conversation's history, so I do not know the next step. "
            "Tell me the project and your role again and I will build it again." if language == "en" else
            "Daftar bacanya sudah tidak ada di riwayat percakapan ini, jadi saya tidak tahu langkah berikutnya. "
            "Sebutkan lagi project dan perannya, nanti saya susun ulang.")
    return text, Guide(lead=text)


def continuation(
    question: str, memory, scope, base_url: str, language: str,
    read: Optional[List[str]] = None, show: Optional[str] = None, forced: bool = False,
) -> Optional[Tuple[str, Guide]]:
    """The next item of the newest list. `forced`: the router read the message as
    progress (continue_list), with the keys it reported read and one to show again."""
    english = language == "en"
    plain = casual_to_plain(question)
    bare_done = bool(_DONE.match(plain)) and not forced
    if not (forced or _NEXT.match(plain) or _READ.match(plain) or bare_done):
        return None
    turns = _recent_turns(memory)
    found, index, carried_done, ctx = _latest_list(turns)
    if index is None:
        return _list_gone(turns, language)
    after_list = turns[index + 1:]
    if bare_done:
        # The reply that carries the list may itself be the step just shown.
        last_reply = next((t["content"] for t in reversed(turns[index:]) if t["role"] == "assistant"), "")
        if not _SHOWN.search(last_reply):
            return None
    # A new guidance request after this list starts another project/task.
    if any(t["role"] == "user" and is_guidance(t["content"]) for t in after_list):
        return None
    allowed = {c.collection_id: set(c.file_names) for c in scope.collections}
    steps = []
    for match in found:
        step = _step_from_text(match, allowed, base_url, language)
        if step is None:
            return None
        steps.append(step)
    keys = [s.key for s in steps]
    # The model may add keys the user never wrote ("abis itu apa?" came back as read a, b, c):
    # only keys in the message itself count, or every key for "semua" / "all".
    written = _written_keys(plain)
    if _ALL.search(plain):
        written |= set(keys)
    read = [key for key in read or [] if key in written]
    show = show if show in written else None
    done = {key for key in read if key in keys} | (carried_done & set(keys))
    for turn in turns[index:] + [{"role": "user", "content": question}]:
        if turn["role"] == "assistant":
            done.update(_SHOWN.findall(turn["content"]))
            continue
        said = _READ.match(casual_to_plain(turn["content"]))
        if said:
            done |= _read_keys(said[1], keys)
    carry = _carry([match[0] for match in found], done, ctx)
    if show in keys:
        return _one_step_reply("Step " if english else "Langkah ", steps[keys.index(show)], language, carry)
    step = next((s for s in steps if s.key not in done), None)
    if step is None:
        text = ("You have read all the steps. Would you like to ask about a particular part?" if english else
                "Semua langkah sudah kamu baca. Mau tanya bagian tertentu?")
        return text + carry, Guide(lead=text)
    return _one_step_reply("Next: " if english else "Berikutnya: ", step, language, carry)
