"""
agnostic/indexer.py
-------------------
EmbeddingModel      — singleton HuggingFaceEmbeddings (MiniLM-L12-v2, 384-dim)
UniversalTextSplitter — splits List[RawDocument] → List[LCDocument]
RuntimeIndexBuilder  — builds FAISS index in-memory with session cache
RetrievedChunk       — dataclass for a single retrieval result
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agnostic.config import config
from agnostic.adapters import RawDocument


@dataclass
class RetrievedChunk:
    """A single chunk returned by QueryProcessor after similarity search."""
    content:  str
    source:   str
    doc_type: str
    score:    float
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingModel:
    """
    Singleton HuggingFaceEmbeddings.
    Model is loaded once per kernel session — no re-loading.

    Model: paraphrase-multilingual-MiniLM-L12-v2
    Dim:   384
    Device: CPU
    Normalization: enabled (cosine-friendly)
    """
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print(f"Loading embedding model: {config.embedding_model}...", end="", flush=True)
            cls._instance = HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print(" done")
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None
        print("Embedding model reset.")


class UniversalTextSplitter:
    """
    Wraps RecursiveCharacterTextSplitter to process List[RawDocument]
    and produce List[LCDocument] ready for embedding.

    chunk_size and chunk_overlap are read from config at split() time
    so they always reflect the latest config values.

    Parameters (from config):
        chunk_size    : 2000 characters
        chunk_overlap : 300 characters
        separators    : ["\n\n", "\n", ". ", " ", ""]
    """

    def split(self, raw_docs: List[RawDocument]) -> List:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document as LCDocument

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        result = []
        for rd in raw_docs:
            if not rd.content.strip():
                continue
            chunks = splitter.split_text(rd.content)
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                meta = {
                    "source":   rd.source,
                    "doc_type": rd.doc_type,
                    "chunk_i":  i,
                    **rd.metadata,
                }
                result.append(LCDocument(page_content=chunk, metadata=meta))
        return result


class RuntimeIndexBuilder:
    """
    Build a FAISS vector index from List[LCDocument] entirely in-memory.
    Session cache prevents re-indexing the same source within one session.

    Cache key: source string passed to build().
    Use clear_cache() to force re-indexing.
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def build(self, docs: List, source_key: str = "") -> Any:
        from langchain_community.vectorstores import FAISS

        if config.use_session_cache and source_key and source_key in self._cache:
            print(f"  Index from cache: {source_key[:60]}")
            return self._cache[source_key]

        embedder = EmbeddingModel.get()
        print(f"  Building FAISS index ({len(docs)} chunks)...", end="", flush=True)
        vs = FAISS.from_documents(docs, embedder)
        print(" done")

        if config.use_session_cache and source_key:
            self._cache[source_key] = vs
        return vs

    def clear_cache(self, source_key: Optional[str] = None):
        if source_key:
            self._cache.pop(source_key, None)
            print(f"Cache '{source_key}' cleared.")
        else:
            self._cache.clear()
            print("All index cache cleared.")
