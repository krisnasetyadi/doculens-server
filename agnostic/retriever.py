"""
agnostic/retriever.py
---------------------
QueryProcessor — embeds the user question, runs similarity search in FAISS,
                 filters by similarity_threshold, and returns the top-K chunks.

Retrieval formula:
    similarity = 1 / (1 + L2_distance)   (converts FAISS L2 → similarity score)
    threshold  : config.similarity_threshold  (default 0.20)
    top_k      : config.top_k                 (default 8)
"""

from pathlib import Path
from typing import Any, List

from agnostic.config import config
from agnostic.indexer import RetrievedChunk


class QueryProcessor:
    """
    Embed the question → similarity search in FAISS index →
    filter + sort → return List[RetrievedChunk].

    The question is vectorized using the same EmbeddingModel singleton
    as the document chunks (paraphrase-multilingual-MiniLM-L12-v2, 384-dim).
    This is the "query vectorization" path shown crossing the architecture diagram.
    """

    def retrieve(self, question: str, vectorstore: Any) -> List[RetrievedChunk]:
        """
        Retrieve top-K relevant chunks for the question.

        Args:
            question    : user query string
            vectorstore : FAISS vectorstore built by RuntimeIndexBuilder

        Returns:
            List[RetrievedChunk] sorted by score descending, max config.top_k items
        """
        raw = vectorstore.similarity_search_with_score(
            question, k=config.top_k * 2
        )
        results: List[RetrievedChunk] = []
        for doc, l2_dist in raw:
            score = 1.0 / (1.0 + float(l2_dist))
            if score < config.similarity_threshold:
                continue
            results.append(RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                doc_type=doc.metadata.get("doc_type", "unknown"),
                score=round(score, 4),
                metadata=doc.metadata,
            ))
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:config.top_k]

    def build_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a numbered context block for the LLM prompt.
        Source label format: "[Source N — filename (doc_type)]:"
        """
        if not chunks:
            return ""
        parts = []
        for i, c in enumerate(chunks, 1):
            if c.source.startswith(("table:", "query:")):
                src_label = c.source
            else:
                src_label = Path(c.source).name
            parts.append(
                f"[Source {i} — {src_label} ({c.doc_type})]:\n{c.content}"
            )
        return "\n\n---\n\n".join(parts)
