"""
agnostic/generator.py
---------------------
AnswerGenerator — LLM wrapper with automatic retry + model fallback chain.

Primary model : Gemini 2.5-flash
Fallback chain:
    attempt 0-1 → gemini-2.5-flash   (primary)
    attempt 2-3 → gemini-2.0-flash   (first fallback)
    attempt 4-5 → gemini-1.5-flash   (second fallback)
    attempt 6   → return error

Retry triggers:
    HTTP 429 RESOURCE_EXHAUSTED  → exponential backoff: 15→30→60→120→240s
    HTTP 503 UNAVAILABLE         → same backoff + model fallback at attempt 2/4

Also supports HuggingFace (flan-t5 family) as an alternative provider.
"""

import time
import logging
from typing import Dict, Optional

from agnostic.config import config

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Generate answers from an LLM using retrieved context chunks.
    Zero-shot: no fine-tuning, model used as-is.

    Usage:
        gen = AnswerGenerator()
        gen.load_gemini()          # or gen.load_huggingface()
        answer = gen.generate(question, context)
    """

    _FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash"]

    def __init__(self):
        self._llm      = None
        self._provider = None
        self._model    = None
        self._cache: Dict[str, object] = {}

    # ── Loaders ──────────────────────────────────────────────────────────────

    def load_gemini(self, model: Optional[str] = None) -> bool:
        """Load a Gemini model via langchain-google-genai."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            print("langchain_google_genai not installed.")
            return False

        m   = model or config.llm_model
        key = f"gemini/{m}"
        if key in self._cache:
            self._llm, self._provider, self._model = self._cache[key], "gemini", m
            print(f"Gemini from cache: {m}")
            return True
        try:
            print(f"Loading Gemini {m}...", end="", flush=True)
            llm = ChatGoogleGenerativeAI(
                model=m,
                google_api_key=config.gemini_api_key,
                temperature=0.3,
                max_output_tokens=2048,
            )
            llm.invoke("test")
            self._cache[key] = llm
            self._llm, self._provider, self._model = llm, "gemini", m
            print(" done")
            return True
        except Exception as e:
            print(f" error: {e}")
            return False

    def load_huggingface(self, model: Optional[str] = None) -> bool:
        """Load a HuggingFace seq2seq model (flan-t5 family)."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            from langchain_community.llms import HuggingFacePipeline
        except ImportError:
            print("transformers not installed.")
            return False

        m   = model or config.hf_model
        key = f"hf/{m}"
        if key in self._cache:
            self._llm, self._provider, self._model = self._cache[key], "huggingface", m
            print(f"HuggingFace from cache: {m}")
            return True
        try:
            print(f"Loading HuggingFace {m}...", end="", flush=True)
            tok  = AutoTokenizer.from_pretrained(m)
            mdl  = AutoModelForSeq2SeqLM.from_pretrained(m)
            pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok,
                            max_new_tokens=512, temperature=0.3, do_sample=True)
            llm  = HuggingFacePipeline(pipeline=pipe)
            self._cache[key] = llm
            self._llm, self._provider, self._model = llm, "huggingface", m
            print(" done")
            return True
        except Exception as e:
            print(f" error: {e}")
            return False

    # ── Generation ───────────────────────────────────────────────────────────

    def _load_gemini_instance(self, model: str):
        """Create/retrieve a Gemini instance for fallback without changing self._llm."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = f"gemini/{model}"
        if key in self._cache:
            return self._cache[key]
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=config.gemini_api_key,
            temperature=0.3,
            max_output_tokens=2048,
        )
        self._cache[key] = llm
        return llm

    def generate(self, question: str, context: str,
                 max_retries: int = 6, base_delay: float = 15.0) -> str:
        """
        Generate an answer with automatic retry + model fallback.

        Retry schedule per attempt:
            attempt 0 → primary model (2.5-flash),  no wait
            attempt 1 → primary model,               wait 15s
            attempt 2 → fallback gemini-2.0-flash,   wait 30s
            attempt 3 → fallback gemini-2.0-flash,   wait 60s
            attempt 4 → fallback gemini-1.5-flash,   wait 120s
            attempt 5 → fallback gemini-1.5-flash,   wait 240s
            attempt 6 → return error message

        Args:
            question    : user question string
            context     : formatted context from QueryProcessor.build_context()
            max_retries : total retry limit (default 6)
            base_delay  : initial wait in seconds, doubles each attempt (default 15)

        Returns:
            str — generated answer, or error message string on failure
        """
        if not self._llm:
            return "LLM not loaded. Call load_gemini() or load_huggingface() first."

        APP_DESCRIPTION = (
            "DocuLens is an AI-powered document intelligence platform. "
            "Users can upload PDF documents and chat logs (WhatsApp, Telegram, etc.), "
            "then ask questions in natural language. The system retrieves relevant passages "
            "from those sources and generates precise, grounded answers. "
            "Key capabilities: multi-PDF Q&A, chat log analysis, hybrid search across documents "
            "and databases, source citations with page references, dark mode, conversation history, "
            "and support for multiple AI models (Gemini 2.5-flash, etc.)."
        )

        OFF_TOPIC_REDIRECT = (
            "I'm built to answer questions about your documents, databases, and chat "
            "logs — not general topics. Try asking something like \"summarize my PDFs\" "
            "or \"what's in my chat logs?\" instead."
        )

        META_KEYWORDS = (
            "what is", "how does", "how do i", "how to use", "what can", "what are",
            "tell me about", "explain", "help me", "guide", "tutorial", "getting started",
            "fitur", "cara", "apa itu", "bagaimana", "panduan", "mulai"
        )
        q_lower = question.lower().strip()
        is_meta = any(q_lower.startswith(kw) for kw in META_KEYWORDS)

        # No context at all — answer as app assistant
        if not context.strip():
            prompt = (
                f"You are the DocuLens AI assistant. {APP_DESCRIPTION}\n\n"
                "The user has not selected any document sources yet, or their question is about the app itself.\n"
                "Answer helpfully as an onboarding assistant. Be concise and friendly.\n"
                "If the question is a general-knowledge request unrelated to DocuLens or the user's own "
                "data — e.g. travel/vacation recommendations, coding help, trivia, personal advice, or any "
                "other topic not about this platform or the user's documents — do NOT answer it. Instead "
                f"reply with exactly this redirect: \"{OFF_TOPIC_REDIRECT}\"\n\n"
                f"USER: {question}\n\nASSISTANT:"
            )
        elif is_meta and len(context.strip()) < 200:
            # Very thin context + meta question → blend app knowledge with context
            prompt = (
                f"You are the DocuLens AI assistant. {APP_DESCRIPTION}\n\n"
                f"CONTEXT FROM DOCUMENTS:\n{context}\n\n"
                f"USER: {question}\n\nASSISTANT:"
            )
        else:
            prompt = (
                "You are DocuLens AI, an assistant that answers questions based strictly on the provided context.\n"
                f"About this platform: {APP_DESCRIPTION}\n\n"
                "Instructions:\n"
                "1. Answer using only the information available in the CONTEXT below.\n"
                "2. If the context contains partial information, provide the partial answer and note what is missing.\n"
                "3. If the context contains numbers, tables, or data — extract and display them directly.\n"
                "4. Only say 'Information not found in the available data sources.' if the context contains "
                "ABSOLUTELY NO information relevant to the question.\n"
                "5. Do not fabricate facts outside the context.\n"
                "6. If the user asks about the app itself (what it does, how to use it), answer from your app knowledge.\n"
                "7. If the QUESTION is a general-knowledge request unrelated to both the CONTEXT and the "
                "platform itself — e.g. travel/vacation recommendations, coding help, trivia, personal "
                f"advice — do NOT attempt an answer from general knowledge. Reply with exactly this "
                f"redirect instead: \"{OFF_TOPIC_REDIRECT}\"\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}\n\n"
                "ANSWER (based on the context above):"
            )

        def _get_llm(attempt: int):
            if attempt >= 4 and len(self._FALLBACK_MODELS) >= 2:
                fb = self._FALLBACK_MODELS[1]   # gemini-1.5-flash
                print(f"   ↪ Fallback to {fb}", flush=True)
                return self._load_gemini_instance(fb)
            elif attempt >= 2 and len(self._FALLBACK_MODELS) >= 1:
                fb = self._FALLBACK_MODELS[0]   # gemini-2.0-flash
                print(f"   ↪ Fallback to {fb}", flush=True)
                return self._load_gemini_instance(fb)
            return self._llm

        for attempt in range(max_retries + 1):
            llm_to_use = _get_llm(attempt)
            try:
                resp = llm_to_use.invoke(prompt)
                if attempt > 0:
                    print(f"   ✓ Succeeded on attempt {attempt + 1}", flush=True)
                return resp.content if hasattr(resp, "content") else str(resp)
            except Exception as e:
                err_str        = str(e)
                is_rate_limit  = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                is_unavailable = "503" in err_str or "UNAVAILABLE"        in err_str

                if attempt >= max_retries:
                    return f"Generate error: {e}"

                if is_rate_limit or is_unavailable:
                    wait      = base_delay * (2 ** attempt)   # 15→30→60→120→240→480s
                    err_label = "429 rate-limit" if is_rate_limit else "503 unavailable"
                    print(f"   ⚠ Gemini {err_label} — waiting {wait:.0f}s then retry "
                          f"({attempt + 1}/{max_retries})...", flush=True)
                    time.sleep(wait)
                else:
                    return f"Generate error: {e}"

        return "Generate error: max retries exceeded."

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._llm is not None

    @property
    def info(self) -> str:
        return f"{self._provider}/{self._model}" if self._provider else "Not loaded"
