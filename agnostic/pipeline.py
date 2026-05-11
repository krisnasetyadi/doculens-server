"""
agnostic/pipeline.py
--------------------
RAGResult           — dataclass for the full output of one pipeline run
AgnosticRAGPipeline — top-level orchestrator wiring all pipeline stages

Pipeline stages:
    1. Load       : SourceFactory → adapter.load() → List[RawDocument]
    2. Split      : UniversalTextSplitter → List[LCDocument]
    3. Index      : RuntimeIndexBuilder  → FAISS vectorstore (session cached)
    4a. Retrieve  : QueryProcessor       → List[RetrievedChunk]
    4b. Generate  : AnswerGenerator      → answer string
    (Evaluate)    : Evaluator.score()    → EvalScore  [optional, separate call]

Usage:
    from agnostic.pipeline import AgnosticRAGPipeline
    from agnostic.config import config
    from agnostic.generator import AnswerGenerator

    config.gemini_api_key = "YOUR_KEY"

    gen = AnswerGenerator()
    gen.load_gemini()

    pipeline = AgnosticRAGPipeline(generator=gen)

    # Folder (local):
    result = pipeline.ask("your question", source="/path/to/docs")

    # PostgreSQL:
    result = pipeline.ask("your question", source="postgresql://user:pass@host/db")

    # Hybrid (Folder + PostgreSQL):
    result = pipeline.ask("your question", source="/path/to/docs|postgresql://...")

    result.display()
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agnostic.adapters import RawDocument
from agnostic.config import config
from agnostic.detector import SourceDetector, SourceFactory
from agnostic.generator import AnswerGenerator
from agnostic.indexer import RetrievedChunk, RuntimeIndexBuilder, UniversalTextSplitter
from agnostic.retriever import QueryProcessor


@dataclass
class RAGResult:
    """Full output of one AgnosticRAGPipeline.ask() call."""
    question:         str
    answer:           str
    retrieved_chunks: List[RetrievedChunk]
    timing:           Dict[str, float]
    metadata:         Dict[str, Any]

    @property
    def total_time(self) -> float:
        return sum(self.timing.values())

    def display(self):
        print("\n" + "=" * 62)
        print(f"QUESTION:\n   {self.question}")
        print("-" * 62)
        print("ANSWER:")
        for line in self.answer.strip().split("\n"):
            print(f"   {line}")
        print("-" * 62)
        print(f"CHUNKS ({len(self.retrieved_chunks)}):")
        for i, c in enumerate(self.retrieved_chunks[:3], 1):
            src = (Path(c.source).name
                   if not c.source.startswith(("table:", "query:")) else c.source)
            print(f"   [{i}] score={c.score:.3f} | {c.doc_type} | {src}")
            print(f"       {c.content[:110].replace(chr(10), ' ')}...")
        if len(self.retrieved_chunks) > 3:
            print(f"   ... +{len(self.retrieved_chunks) - 3} more chunks")
        print("-" * 62)
        print("TIMING:")
        for step, t in self.timing.items():
            bar = "#" * max(1, int(t * 8))
            print(f"   {step:<22} {t:.3f}s  {bar}")
        print(f"   {'TOTAL':<22} {self.total_time:.3f}s")
        print("-" * 62)
        print(f"Source: {self.metadata.get('source_type', '?')} | "
              f"{self.metadata.get('raw_docs', '?')} docs | "
              f"{self.metadata.get('total_chunks', '?')} chunks | "
              f"LLM: {self.metadata.get('llm', '?')}")
        print("=" * 62)


class AgnosticRAGPipeline:
    """
    Universal RAG pipeline — agnostic to data source type.

    Accepts any combination of:
      - Local folder (PDF / TXT / MD / LOG)
      - PostgreSQL database (any tables or custom SQL)
      - Hybrid (both, separated by "|")

    Args:
        generator : AnswerGenerator instance (must have load_gemini() called)
                    If None, creates a new instance (still needs load_gemini()).
    """

    def __init__(self, generator: Optional[AnswerGenerator] = None):
        self._splitter      = UniversalTextSplitter()
        self._index_builder = RuntimeIndexBuilder()
        self._query_proc    = QueryProcessor()
        self._generator     = generator or AnswerGenerator()
        self._history: List[RAGResult] = []

    def ask(
        self,
        question: str,
        source: str,
        pg_tables: Optional[List[str]] = None,
        pg_queries: Optional[Dict[str, str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> RAGResult:
        """
        Run the full RAG pipeline for one question.

        Args:
            question         : user question string (non-empty)
            source           : data source — folder path, postgresql:// URL, or
                               "folder_path|postgresql://..." for hybrid
            pg_tables        : specific PostgreSQL tables to load (None = all)
            pg_queries       : custom SQL queries dict {label: sql}
            exclude_patterns : file name substrings to skip (FolderSourceAdapter)
            verbose          : print progress per stage

        Returns:
            RAGResult with answer, retrieved chunks, timing, and metadata
        """
        if not question.strip():
            raise ValueError("question must not be empty.")
        if not source.strip():
            raise ValueError("source must not be empty.")

        timing: Dict[str, float] = {}

        if verbose:
            print(f"\nAgnosticRAG — starting...")
            print(f"   Source   : {SourceDetector.describe(source)}")
            print(f"   Question : {question[:80]}{'...' if len(question) > 80 else ''}")

        # Stage 1 — Load
        t = time.time()
        if verbose:
            print("\n[1/4] Loading documents from source...")
        adapter  = SourceFactory.create(source, tables=pg_tables,
                                        custom_queries=pg_queries,
                                        exclude_patterns=exclude_patterns)
        raw_docs = adapter.load()
        timing["1_load"] = time.time() - t
        if verbose:
            print(f"  {len(raw_docs)} raw documents ({timing['1_load']:.2f}s)")

        if not raw_docs:
            return RAGResult(
                question=question,
                answer="No documents found in the specified source.",
                retrieved_chunks=[],
                timing=timing,
                metadata={
                    "source": source,
                    "source_type": SourceDetector.describe(source),
                    "raw_docs": 0, "total_chunks": 0,
                    "llm": self._generator.info,
                }
            )

        # Stage 2 — Split
        t = time.time()
        if verbose:
            print(f"\n[2/4] Splitting documents...")
        chunks = self._splitter.split(raw_docs)
        timing["2_split"] = time.time() - t
        if verbose:
            print(f"  {len(chunks)} chunks ({timing['2_split']:.2f}s)")

        # Stage 3 — Index
        t = time.time()
        if verbose:
            print(f"\n[3/4] Building FAISS index...")
        vectorstore = self._index_builder.build(chunks, source_key=source)
        timing["3_index"] = time.time() - t

        # Stage 4a — Retrieve
        t = time.time()
        if verbose:
            print(f"\n[4a/4] Retrieving relevant chunks...")
        retrieved = self._query_proc.retrieve(question, vectorstore)
        timing["4a_retrieve"] = time.time() - t
        if verbose:
            print(f"  {len(retrieved)} relevant chunks ({timing['4a_retrieve']:.2f}s)")

        context = self._query_proc.build_context(retrieved)

        # Stage 4b — Generate
        t = time.time()
        if verbose:
            print(f"\n[4b/4] Generating answer ({self._generator.info})...")
        answer = self._generator.generate(question, context)
        timing["4b_generate"] = time.time() - t

        result = RAGResult(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved,
            timing=timing,
            metadata={
                "source":       source,
                "source_type":  SourceDetector.describe(source),
                "raw_docs":     len(raw_docs),
                "total_chunks": len(chunks),
                "retrieved":    len(retrieved),
                "llm":          self._generator.info,
                "timestamp":    datetime.now().isoformat(),
            }
        )
        self._history.append(result)
        return result

    @property
    def history(self) -> List[RAGResult]:
        return self._history

    def clear_history(self):
        self._history.clear()
        print("History cleared.")

    def show_history(self):
        if not self._history:
            print("History is empty.")
            return
        print(f"\nHISTORY ({len(self._history)} questions):")
        for i, r in enumerate(self._history, 1):
            src = r.metadata.get("source", "?")[:40]
            print(f"  [{i}] {r.question[:55]:55} | {r.total_time:.2f}s | {src}")

    def clear_index_cache(self, source_key: Optional[str] = None):
        """Clear FAISS session cache to force re-indexing."""
        self._index_builder.clear_cache(source_key)
