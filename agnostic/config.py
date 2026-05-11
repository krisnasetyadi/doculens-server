"""
agnostic/config.py
------------------
Global configuration dataclass for the Agnostic Multi-Source RAG pipeline.
Set config.gemini_api_key before instantiating any pipeline objects.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Embedding
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # LLM
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    hf_model: str = "google/flan-t5-base"
    gemini_api_key: str = ""

    # Chunking
    chunk_size: int = 2000
    chunk_overlap: int = 300

    # Retrieval
    top_k: int = 8
    similarity_threshold: float = 0.2

    # Database
    max_db_rows: int = 1000

    # Index cache
    use_session_cache: bool = True

    # Supported file extensions (FolderSourceAdapter)
    # DOCX is disabled — uncomment to re-enable (requires python-docx)
    doc_extensions: List[str] = field(default_factory=lambda: [
        ".pdf",
        ".txt",
        ".md",
        ".log",
        # ".docx",
        # ".doc",
    ])


# Singleton config instance — import and mutate this directly
config = Config()

# Auto-load API key from environment if available
if not config.gemini_api_key:
    config.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
