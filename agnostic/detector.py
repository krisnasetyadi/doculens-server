"""
agnostic/detector.py
--------------------
SourceDetector  — auto-classifies a source string into FOLDER / POSTGRES / HYBRID.
SourceFactory   — instantiates the correct adapter from a source string.
"""

import re
from enum import Enum, auto
from typing import Dict, List, Optional


class SourceType(Enum):
    FOLDER   = auto()
    POSTGRES = auto()
    HYBRID   = auto()   # Folder + PostgreSQL, separated by "|"


class SourceDetector:
    """
    Auto-detect the data source type from a single string.

    Rules (evaluated in order):
      1. Contains "|"            → HYBRID  (multi-source)
      2. Starts with postgres:// → POSTGRES
      3. Looks like a file path  → FOLDER
      4. Default fallback        → FOLDER
    """

    _POSTGRES_PREFIXES = ("postgresql://", "postgres://")
    _FOLDER_PATTERNS   = (r"^/", r"^\./", r"^\.\./", r"^[A-Za-z]:[/\\]", r"^~")

    @classmethod
    def detect(cls, source: str) -> SourceType:
        s = source.strip()
        if "|" in s:
            return SourceType.HYBRID
        if any(s.startswith(p) for p in cls._POSTGRES_PREFIXES):
            return SourceType.POSTGRES
        if any(re.match(p, s) for p in cls._FOLDER_PATTERNS):
            return SourceType.FOLDER
        return SourceType.FOLDER

    @classmethod
    def describe(cls, source: str) -> str:
        labels = {
            SourceType.FOLDER:   "Folder (local / Google Drive)",
            SourceType.POSTGRES: "PostgreSQL Database",
            SourceType.HYBRID:   "Hybrid (Folder + PostgreSQL)",
        }
        return labels[cls.detect(source)]


class SourceFactory:
    """
    Create the correct adapter from a source string.
    Supports "|" separator for multi-source (Hybrid).

    Example:
        adapter = SourceFactory.create("/data/docs|postgresql://user:pass@host/db")
    """

    @staticmethod
    def create(
        source: str,
        tables: Optional[List[str]] = None,
        custom_queries: Optional[Dict[str, str]] = None,
        max_depth: Optional[int] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        # Import here to avoid circular imports
        from agnostic.adapters import FolderSourceAdapter, PostgreSQLAdapter, MultiSourceAdapter

        stype = SourceDetector.detect(source)

        if stype == SourceType.HYBRID:
            parts = [s.strip() for s in source.split("|") if s.strip()]
            adapters = []
            for part in parts:
                sub = SourceDetector.detect(part)
                if sub == SourceType.FOLDER:
                    adapters.append(FolderSourceAdapter(
                        part, max_depth=max_depth, exclude_patterns=exclude_patterns
                    ))
                elif sub == SourceType.POSTGRES:
                    adapters.append(PostgreSQLAdapter(
                        part, tables=tables, custom_queries=custom_queries
                    ))
            return MultiSourceAdapter(adapters)

        elif stype == SourceType.FOLDER:
            from agnostic.adapters import FolderSourceAdapter
            return FolderSourceAdapter(source, max_depth=max_depth,
                                       exclude_patterns=exclude_patterns)

        elif stype == SourceType.POSTGRES:
            from agnostic.adapters import PostgreSQLAdapter
            return PostgreSQLAdapter(source, tables=tables, custom_queries=custom_queries)

        raise ValueError(f"Unknown source type for: {source}")
