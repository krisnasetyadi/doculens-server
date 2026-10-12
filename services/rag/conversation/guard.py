"""
Guard against questions about how DocuLens itself is built (tech stack,
AI model/provider, database, hosting, prompts, source code) and against
prompt-injection attempts.

Three layers, because a prompt instruction alone is easy to talk around:
1. is_internal_question() — regex BEFORE any LLM call; a hit gets a canned
   refusal (messages.internal_refusal) and costs nothing.
2. The conversation prompt itself forbids the topic (processor).
3. leaks_internal_details() — scans the conversation-mode answer AFTER the
   LLM; a hit replaces the whole answer with the same canned refusal.

Scoped to questions whose SUBJECT is DocuLens/the assistant, and tuned to
let feature questions through: "can DocuLens connect to my database?" or
"can you find which server crashed?" are legitimate, while "DocuLens pakai
database apa?" or "what model are you?" are not. Words that are also
features ("database", "server", "model", "AI", "prompt") therefore only
count when the question asks WHICH one is used; unambiguous implementation
words ("framework", "tech stack", "LLM", "source code") count on their own.

Only DocuLens BY NAME (or the assistant addressed directly: "kamu", "your
model") counts as the subject. Demonstratives like "aplikasi ini" / "sistem
ini" / "this system" are left alone: in a document Q&A app they usually mean
the system described in the user's own document ("jelaskan arsitektur
aplikasi ini"), and if such a question does land in conversation mode,
layer 3 still stops a leak.
"""

import re
from typing import Iterable

# Only ever about implementation.
_STRICT = (
    r"(tech\s*stack|techstack|technology\s+stack|teknologi|technolog(y|ies)|frameworks?|librar(y|ies)|"
    r"bahasa\s+pemrograman|programming\s+languages?|backend|back-end|frontend|front-end|"
    r"\bllms?\b|(ai|language)\s+models?|model\s+(ai|llm|bahasa)|"
    r"\bgpt(-?\d+)?\b|chatgpt|gemini|openai|claude|anthropic|llama|mistral|langchain|"
    r"vector\s*(db|database|store)|faiss|embeddings?|\brag\b|system\s+prompt|"
    r"source\s*code|kode\s+sumber|codebase|repositor(y|i)|github|api\s*keys?|secret\s+keys?|"
    r"kredensial|credentials?|arsitektur|architecture|infrastruktur|infrastructure|"
    r"hosting|cloud\s+provider)"
)

# Also user-facing feature words — only count when asking which one is used.
_AMBIGUOUS = r"(database|\bdb\b|server|models?|\bai\b|prompts?|instruksi|instructions?|deploy(ment)?|cloud|stack)"

_USE_VERB = r"(use|uses|using|used|pakai|pake|pakek|menggunakan|gunakan|digunakan|dipakai|run|runs|running|jalan|berjalan)"
_BUILD_VERB = (
    r"(built|made|developed|created|coded|written|powered|trained|hosted|"
    r"dibuat|dibangun|dikembangkan|ditulis|dilatih)"
)

_WHICH_AMBIGUOUS = (
    rf"(\b{_USE_VERB}\s+(\w+\s+)?{_AMBIGUOUS}\s+(apa|what)\b|"
    rf"\b(what|which|apa)\s+(kind\s+of\s+)?{_AMBIGUOUS}\b.*\b{_USE_VERB}\b|"
    rf"{_AMBIGUOUS}\s+apa\s+(yang\s+)?{_USE_VERB}\b)"
)

# DocuLens itself named explicitly (or the chatbot itself, which can only be
# this assistant — unlike "aplikasi ini"/"this system", see module docstring).
_APP_REFERENCE = re.compile(
    r"\b(doculens|docu\s+lens|bot\s+ini|chatbot\s+ini|asisten\s+ini|"
    r"this\s+(bot|chatbot|assistant))\b",
    re.IGNORECASE,
)
_APP_INTERNAL = re.compile(rf"{_STRICT}|\b{_BUILD_VERB}\b|{_WHICH_AMBIGUOUS}", re.IGNORECASE)

_PRONOUN = r"(you|u|kamu|kau|anda|lu|lo|elu)"
_SELF_THING = rf"({_STRICT}|models?|prompts?|instruksi|instructions?|\bai\b)"

# The assistant addressed by pronoun — only "about yourself" shapes.
_ASSISTANT_SELF = re.compile(
    "|".join([
        rf"\b(your|yours)\s+(own\s+)?{_SELF_THING}",                     # "your model", "your tech stack"
        rf"{_SELF_THING}\s*(mu|kamu|anda|lu|lo)\b",                       # "framework kamu", "promptmu"
        rf"\b{_PRONOUN}\s+(punya\s+)?{_SELF_THING}\s+(apa|mana)\b",       # "kamu punya model apa"
        rf"\b{_PRONOUN}\s+(were\s+|are\s+|itu\s+)?({_USE_VERB}|{_BUILD_VERB})\b.*({_STRICT}|{_AMBIGUOUS}\s+(apa|what)\b)",
        rf"({_STRICT}|{_AMBIGUOUS}).*\b(do|did|does|are|were)\s+{_PRONOUN}\s+({_USE_VERB}|{_BUILD_VERB})\b",
        rf"({_STRICT}|{_AMBIGUOUS})\s+apa\s+(yang\s+)?{_PRONOUN}\s+{_USE_VERB}\b",
        rf"\b(are|r)\s+{_PRONOUN}\s+(a\s+|an\s+)?(gpt|chatgpt|gemini|claude|llama|mistral|openai|llm|language\s+model)\b",
        rf"\b{_PRONOUN}\s+(itu\s+|ini\s+)?(gpt|chatgpt|gemini|claude|llama|mistral|openai|llm)\b",
        rf"\b(what|which)\s+(ai\s+|language\s+)?(model|llm)\s+(are|r)\s+{_PRONOUN}\b",
        rf"\bhow\s+(were|are|was)\s+{_PRONOUN}\s+{_BUILD_VERB}\b",
        rf"\b{_PRONOUN}\s+{_BUILD_VERB}\s+(pakai|pake|dengan|menggunakan|oleh|gimana|bagaimana|with|using|by)\b",
    ]),
    re.IGNORECASE,
)

# Prompt-injection / jailbreak attempts — refused regardless of subject.
_INJECTION = re.compile(
    r"(ignore\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)|"
    r"disregard\s+(all\s+|the\s+|your\s+)?(previous|prior|above)\s+(instructions?|rules?)|"
    r"abaikan\s+(semua\s+)?(instruksi|perintah|aturan)|lupakan\s+(semua\s+)?(instruksi|aturan)|"
    r"system\s+prompt|reveal\s+(your\s+)?(prompt|instructions?)|show\s+(me\s+)?your\s+(prompt|instructions?)|"
    r"tampilkan\s+(prompt|instruksi)\s+(kamu|anda|sistem)|jailbreak|developer\s+mode|"
    r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(unrestricted|unfiltered))",
    re.IGNORECASE,
)

# The user is pointing at THEIR OWN content — not asking about DocuLens.
_DOCUMENT_REFERENCE = re.compile(
    r"\b(dokumen|document|documents|file|files|berkas|pdf|collection|koleksi|tabel|table|chat\s+log|"
    r"menurut|according\s+to|isi\s+dari|uploaded|diunggah|diupload|"
    r"data\s+(saya|kami|kita)|(my|our)\s+data)\b",
    re.IGNORECASE,
)

# Implementation names that must never appear in a conversation-mode answer.
# No database brands (PostgreSQL, MySQL, ...): connecting the user's own
# database is a feature, so the answer may legitimately name one.
_LEAK_TERMS = re.compile(
    r"\b(react|next\.?js|vite|tailwind|tanstack|zustand|fastapi|uvicorn|python|"
    r"supabase|faiss|bm25|langchain|hugging\s*face|huggingface|"
    r"flan-?t5|gemini|openai|gpt(-?\d+)?|chatgpt|claude|anthropic|llama|mistral|"
    r"docker|vercel|hf\s+space|system\s+prompt|vector\s+store|embeddings?)\b",
    re.IGNORECASE,
)


def is_internal_question(question: str) -> bool:
    """True when the question asks how DocuLens/the assistant is built, or
    tries to override/extract its instructions."""
    q = question.strip()
    if _INJECTION.search(q):
        return True
    if _DOCUMENT_REFERENCE.search(q):
        return False
    if _APP_REFERENCE.search(q) and _APP_INTERNAL.search(q):
        return True
    return bool(_ASSISTANT_SELF.search(q))


def leaks_internal_details(answer: str, user_text: Iterable[str] = ()) -> bool:
    """True when a conversation-mode answer names an implementation detail —
    the caller then swaps in the canned refusal instead of trusting it.
    Only applied to conversation mode: a RAG answer quoting the user's own
    document may legitimately say "PostgreSQL" or "React".

    `user_text` is the user's own collection titles: a term that appears
    there isn't a leak, the model is just naming their file ("want me to
    summarize your *React Training Handbook*?"). The question itself is
    deliberately NOT allowed — otherwise "does DocuLens use React?" would
    let "Yes, it uses React" through."""
    allowed = " ".join(user_text).lower()
    return any(
        match.group(0).lower() not in allowed
        for match in _LEAK_TERMS.finditer(answer)
    )
