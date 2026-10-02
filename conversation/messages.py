"""
Canned bilingual (en/id) replies that never go through the LLM: refusals,
system answers, conversation fallbacks, and the source-prefix wording used
by processor's non-LLM fallback answers. The "id" strings are the exact
wording the app used before, so Indonesian users see no change.
"""

from typing import Dict

from conversation.language import ReplyLanguage

_INTERNAL_REFUSAL = {
    "id": (
        "Maaf, saya tidak bisa membahas detail teknis tentang cara DocuLens dibuat. "
        "Saya di sini untuk membantu kamu memakai DocuLens — misalnya bertanya tentang "
        "dokumen kamu atau melihat fitur dan paket yang tersedia."
    ),
    "en": (
        "Sorry, I can't share technical details about how DocuLens is built. "
        "I'm here to help you use DocuLens — for example asking about your documents "
        "or exploring the available features and plans."
    ),
}

_NO_SOURCE_SELECTED = {
    "id": (
        "Pilih dulu minimal satu sumber (PDF, Database, Chat, atau Drive) "
        "sebelum bertanya, biar jawabannya bisa saya dasarkan dari data kamu."
    ),
    "en": (
        "Please turn on at least one source (PDF, Database, Chat, or Drive) "
        "before asking, so I can base the answer on your data."
    ),
}

# Used only when the conversation-mode LLM call fails (or is unavailable) —
# keyed by the small-talk kind from intent.small_talk_kind().
_SMALL_TALK_FALLBACK: Dict[str, Dict[str, str]] = {
    "greeting": {
        "id": "Halo! 👋 Ada yang bisa saya bantu? Kamu bisa tanya isi dokumen kamu, minta ringkasan, atau ketik `/` untuk lihat semua command.",
        "en": "Hi! 👋 How can I help? You can ask about your documents, request a summary, or type `/` to see all commands.",
    },
    "thanks": {
        "id": "Sama-sama! 😊 Kalau ada lagi yang mau ditanyakan soal dokumen kamu, langsung aja.",
        "en": "You're welcome! 😊 Just ask if there's anything else about your documents.",
    },
    "ack": {
        "id": "Siap! Ada lagi yang bisa saya bantu?",
        "en": "Got it! Anything else I can help with?",
    },
    # A "yes" when the router's LLM call isn't available to act on it.
    "affirm": {
        "id": "Siap! Mau tanya apa soal dokumen kamu? Langsung ketik aja pertanyaannya.",
        "en": "Sure! What would you like to know about your documents? Just type your question.",
    },
    "farewell": {
        "id": "Sampai jumpa! 👋 Kapan pun butuh, saya ada di sini.",
        "en": "Bye! 👋 I'm here whenever you need me.",
    },
    "how_are_you": {
        "id": "Saya baik, terima kasih! 😊 Ada yang bisa saya bantu dengan dokumen kamu hari ini?",
        "en": "I'm doing well, thanks! 😊 Anything I can help you with in your documents today?",
    },
}

# Prefixes for processor's non-LLM fallback answers (direct extraction,
# validator replacement, LLM-failure fallback).
FALLBACK_TEXT: Dict[ReplyLanguage, Dict[str, str]] = {
    "id": {
        "based_on": "Berdasarkan {source}:",
        "based_on_page": "Berdasarkan {source} (halaman {page}):",
        "based_on_table": "Berdasarkan data dari tabel {table}:",
        "based_on_db": "Berdasarkan data dari {source} (akurasi: {confidence}):",
        "based_on_chat": "Berdasarkan percakapan dari {source} ({platform}):",
        "info_from": "Informasi dari {source}:",
        "info_from_doc": "Informasi dari dokumen {source} (relevansi: {confidence}):",
        "from": "Dari {source}:",
        "contacts": "Kontak lengkap:",
        "no_valid_answer": "Maaf, sistem tidak dapat menghasilkan jawaban yang valid. Silakan coba pertanyaan yang lebih spesifik.",
    },
    "en": {
        "based_on": "Based on {source}:",
        "based_on_page": "Based on {source} (page {page}):",
        "based_on_table": "Based on data from table {table}:",
        "based_on_db": "Based on data from {source} (accuracy: {confidence}):",
        "based_on_chat": "Based on the conversation from {source} ({platform}):",
        "info_from": "Information from {source}:",
        "info_from_doc": "Information from document {source} (relevance: {confidence}):",
        "from": "From {source}:",
        "contacts": "Full contact details:",
        "no_valid_answer": "The system couldn't produce a valid answer. Please try a more specific question.",
    },
}


def internal_refusal(language: ReplyLanguage) -> str:
    return _INTERNAL_REFUSAL[language]


def no_source_selected(language: ReplyLanguage) -> str:
    return _NO_SOURCE_SELECTED[language]


def small_talk_fallback(kind: str, language: ReplyLanguage) -> str:
    return _SMALL_TALK_FALLBACK.get(kind, _SMALL_TALK_FALLBACK["greeting"])[language]


def fallback_text(language: ReplyLanguage, key: str, **kwargs) -> str:
    return FALLBACK_TEXT[language][key].format(**kwargs)
