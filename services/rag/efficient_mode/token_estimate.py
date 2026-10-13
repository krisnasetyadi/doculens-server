"""Local, dependency-free token estimation for the Efficient Mode
before/after comparison. This is a reporting metric only — it never
feeds back into what's actually sent to the LLM, so it doesn't need to
match any provider's real tokenizer exactly (Gemini and HuggingFace
would need two different tokenizers anyway). ~4 chars/token is the
standard rule-of-thumb approximation for both GPT- and Gemini-style
tokenizers on English/Indonesian prose.
"""

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)
