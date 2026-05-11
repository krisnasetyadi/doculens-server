"""
Agnostic Multi-Source RAG Pipeline
===================================
Extracted from QA_RAG_AgnosticSource.ipynb for standalone use.

Usage:
    from agnostic.pipeline import AgnosticRAGPipeline
    from agnostic.config import config

    config.gemini_api_key = "YOUR_KEY"
    pipeline = AgnosticRAGPipeline()
    result = pipeline.ask("your question", source="/path/to/docs")
    result.display()
"""

from agnostic.config import Config, config
from agnostic.detector import SourceType, SourceDetector, SourceFactory
from agnostic.adapters import RawDocument, FolderSourceAdapter, PostgreSQLAdapter, MultiSourceAdapter
from agnostic.indexer import RetrievedChunk, EmbeddingModel, UniversalTextSplitter, RuntimeIndexBuilder
from agnostic.retriever import QueryProcessor
from agnostic.generator import AnswerGenerator
from agnostic.evaluator import EvalScore, Evaluator
from agnostic.pipeline import RAGResult, AgnosticRAGPipeline

__all__ = [
    "Config", "config",
    "SourceType", "SourceDetector", "SourceFactory",
    "RawDocument", "FolderSourceAdapter", "PostgreSQLAdapter", "MultiSourceAdapter",
    "RetrievedChunk", "EmbeddingModel", "UniversalTextSplitter", "RuntimeIndexBuilder",
    "QueryProcessor",
    "AnswerGenerator",
    "EvalScore", "Evaluator",
    "RAGResult", "AgnosticRAGPipeline",
]
