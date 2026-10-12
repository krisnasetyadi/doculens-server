"""Constants and helpers shared by the services/rag mixins (moved from processor.py by MS-657)."""
from services.rag.conversation.language import ReplyLanguage

# Off-topic scope guard: shown verbatim when a question is unrelated to both
# the retrieved context and the platform itself (e.g. vacation recommendations,
# coding help, trivia). Kept out of the failure_patterns list in
# _validate_and_clean_answer() on purpose — must not overlap those substrings
# or this canned reply gets silently replaced by a forced document answer.
OFF_TOPIC_REDIRECT_ID = (
    "Saya dirancang untuk menjawab pertanyaan seputar dokumen, database, dan "
    "chat log Anda — bukan topik umum. Coba tanyakan misalnya "
    "\"ringkas dokumen saya\" atau \"apa isi chat log saya?\""
)

OFF_TOPIC_INSTRUCTION_ID = (
    f"- Jika PERTANYAAN adalah permintaan pengetahuan umum yang tidak berkaitan "
    f"dengan dokumen di atas maupun dengan platform ini (misalnya rekomendasi "
    f"liburan, bantuan coding, trivia, atau saran pribadi), JANGAN mencoba "
    f"menjawabnya. Balas persis dengan: \"{OFF_TOPIC_REDIRECT_ID}\""
)

# English counterpart, same rule: must not contain any failure_patterns
# substring ("sorry,", "no information", "could not find", ...).
OFF_TOPIC_REDIRECT_EN = (
    "I'm designed to answer questions about your documents, databases, and "
    "chat logs — not general topics. Try asking, for example, "
    "\"summarize my document\" or \"what's in my chat log?\""
)

OFF_TOPIC_INSTRUCTION_EN = (
    f"- Jika PERTANYAAN adalah permintaan pengetahuan umum yang tidak berkaitan "
    f"dengan dokumen di atas maupun dengan platform ini (misalnya rekomendasi "
    f"liburan, bantuan coding, trivia, atau saran pribadi), JANGAN mencoba "
    f"menjawabnya. Balas persis dengan: \"{OFF_TOPIC_REDIRECT_EN}\""
)

def _off_topic_instruction(language: ReplyLanguage) -> str:
    return OFF_TOPIC_INSTRUCTION_ID if language == "id" else OFF_TOPIC_INSTRUCTION_EN

def _extract_total_tokens(llm_result) -> int:
    """Gemini's usage_metadata rides along on the returned AIMessage for free
    (langchain_google_genai maps it from the raw API response) — pull the
    total out of it so callers can meter consumption without a second round
    trip. Local/free fallback models (HuggingFace) don't set this, so this
    returns 0 for them, which is correct — there's no API cost to log."""
    usage_metadata = getattr(llm_result, 'usage_metadata', None) or {}
    return usage_metadata.get('total_tokens', 0) if isinstance(usage_metadata, dict) else 0
