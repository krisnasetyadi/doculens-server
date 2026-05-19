"""
agnostic/adapters.py
--------------------
Source adapters for the Agnostic Multi-Source RAG pipeline.

  RawDocument        — dataclass for a single loaded document (pre-split)
  BaseSourceAdapter  — abstract contract for all adapters
  FolderSourceAdapter — loads PDF / TXT / MD / LOG from a local folder recursively
  PostgreSQLAdapter   — loads tables or custom SQL from a PostgreSQL database
  MultiSourceAdapter  — composite adapter (merges multiple adapters into one)
"""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from agnostic.config import config

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    """
    A single loaded document before splitting.
    Produced by adapters, consumed by UniversalTextSplitter.
    """
    content:  str
    source:   str
    doc_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSourceAdapter(ABC):
    """Abstract contract for all source adapters."""

    @abstractmethod
    def load(self) -> List[RawDocument]:
        ...

    @abstractmethod
    def describe(self) -> str:
        ...


class FolderSourceAdapter(BaseSourceAdapter):
    """
    Recursively load all supported files from a folder.

    Active formats (unstructured data):
      - PDF  (.pdf)           → text from all pages via pypdf
      - Text (.txt, .md, .log) → raw read

    Disabled formats (uncomment to re-enable):
      - Word (.docx, .doc)    → python-docx  [disabled: only PDF & TXT in use]

    Args:
        folder_path       : root folder to scan
        max_depth         : max subfolder depth (None = unlimited)
        exclude_patterns  : substrings — files whose name contains any of these
                            are skipped (case-insensitive).
                            Example: ["copy", "agreement", "dummy"]
    """

    def __init__(self, folder_path: str,
                 max_depth: Optional[int] = None,
                 exclude_patterns: Optional[List[str]] = None):
        self.folder_path      = Path(folder_path).expanduser()
        self.max_depth        = max_depth
        self.exclude_patterns = [p.lower() for p in (exclude_patterns or [])]

    def describe(self) -> str:
        depth_info = f", max_depth={self.max_depth}" if self.max_depth else ""
        excl_info  = f", exclude={self.exclude_patterns}" if self.exclude_patterns else ""
        return f"Folder: {self.folder_path}{depth_info}{excl_info}"

    def _is_excluded(self, fp: Path) -> bool:
        name_lower = fp.name.lower()
        return any(pat in name_lower for pat in self.exclude_patterns)

    def load(self) -> List[RawDocument]:
        if not self.folder_path.exists():
            self._try_mount_drive()

        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")

        docs: List[RawDocument] = []
        all_files = list(self.folder_path.rglob("*"))
        eligible, skipped_excl = [], []

        for f in all_files:
            if not f.is_file():
                continue
            if f.suffix.lower() not in config.doc_extensions:
                continue
            if self.max_depth is not None:
                depth = len(f.relative_to(self.folder_path).parts)
                if depth > self.max_depth:
                    continue
            if self._is_excluded(f):
                skipped_excl.append(f.name)
                continue
            eligible.append(f)

        depth_label = f" (max depth: {self.max_depth})" if self.max_depth else " (all levels)"
        print(f"  {self.folder_path}{depth_label}")
        print(f"  {len(eligible)} eligible files out of {len(all_files)} total")
        if skipped_excl:
            print(f"  {len(skipped_excl)} skipped (exclude_patterns): {skipped_excl}")

        for fp in eligible:
            try:
                raw = self._load_file(fp)
                if raw:
                    docs.append(raw)
                    rel = fp.relative_to(self.folder_path)
                    print(f"     [OK] {rel} [{raw.doc_type}]")
            except Exception as e:
                logger.warning(f"Skip {fp.name}: {e}")
                print(f"     [SKIP] {fp.name}: {e}")

        return docs

    def _try_mount_drive(self):
        """Mount Google Drive if running on Colab."""
        try:
            from google.colab import drive
            print("  Mounting Google Drive...", end="", flush=True)
            drive.mount("/content/drive", force_remount=False)
            print(" done")
        except ImportError:
            pass

    def _load_file(self, fp: Path) -> Optional[RawDocument]:
        ext = fp.suffix.lower()
        if ext == ".pdf":
            return self._load_pdf(fp)
        # elif ext in (".docx", ".doc"):    # DOCX disabled — only PDF & TXT in use
        #     return self._load_docx(fp)
        elif ext in (".txt", ".md", ".log"):
            return self._load_text(fp)
        return None

    def _load_pdf(self, fp: Path) -> Optional[RawDocument]:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(fp))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            if not text:
                return None
            return RawDocument(content=text, source=str(fp), doc_type="pdf",
                               metadata={"pages": len(reader.pages)})
        except Exception as e:
            logger.warning(f"PDF failed {fp.name}: {e}")
            return None

    # -------------------------------------------------------------------------
    # _load_docx — DISABLED (only PDF & TXT in use)
    # To re-enable:
    #   1. Uncomment this method
    #   2. Uncomment routing in _load_file (elif ext in (".docx", ".doc"))
    #   3. Add ".docx", ".doc" to config.doc_extensions
    #   4. pip install python-docx
    # -------------------------------------------------------------------------
    # def _load_docx(self, fp: Path) -> Optional[RawDocument]:
    #     try:
    #         import docx
    #         doc  = docx.Document(str(fp))
    #         text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    #         if not text:
    #             return None
    #         return RawDocument(content=text, source=str(fp), doc_type="docx")
    #     except Exception as e:
    #         logger.warning(f"DOCX failed {fp.name}: {e}")
    #         return None

    def _load_text(self, fp: Path) -> Optional[RawDocument]:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                return None
            ext   = fp.suffix.lower()
            dtype = "markdown" if ext == ".md" else ("log" if ext == ".log" else "txt")
            return RawDocument(content=text, source=str(fp), doc_type=dtype)
        except Exception as e:
            logger.warning(f"Text failed {fp.name}: {e}")
            return None


class PostgreSQLAdapter(BaseSourceAdapter):
    """
    Load data from PostgreSQL — each table/query becomes one RawDocument.

    Connection string format: postgresql://user:password@host:port/dbname
    Data is queried at pipeline.ask() time, not at startup.
    """

    def __init__(self, connection_string: str,
                 tables: Optional[List[str]] = None,
                 custom_queries: Optional[Dict[str, str]] = None):
        self.conn_str       = connection_string
        self.tables         = tables
        self.custom_queries = custom_queries or {}
        self._engine        = None

    def describe(self) -> str:
        safe = re.sub(r":[^@/]+@", ":***@", self.conn_str)
        return f"PostgreSQL: {safe}"

    def _get_engine(self):
        """Lazy-init SQLAlchemy engine with connection pool_pre_ping."""
        if self._engine is None:
            try:
                from sqlalchemy import create_engine, text
                self._engine = create_engine(self.conn_str, pool_pre_ping=True)
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                print("  PostgreSQL connection successful")
            except Exception as e:
                raise ConnectionError(f"PostgreSQL connection failed: {e}")
        return self._engine

    def _list_tables(self) -> List[str]:
        from sqlalchemy import inspect
        inspector = inspect(self._get_engine())
        return inspector.get_table_names(schema="public")

    def load(self) -> List[RawDocument]:
        from sqlalchemy import text as sa_text
        engine = self._get_engine()
        docs: List[RawDocument] = []

        for label, sql in self.custom_queries.items():
            try:
                with engine.connect() as conn:
                    df = pd.read_sql(sa_text(sql), conn)
                if df.empty:
                    print(f"  Query '{label}': empty, skipped")
                    continue
                docs.append(RawDocument(
                    content=self._df_to_text(df, label),
                    source=f"query:{label}",
                    doc_type="db_query",
                    metadata={"rows": len(df), "cols": len(df.columns), "sql": sql}
                ))
                print(f"  [OK] Query '{label}': {len(df)} rows × {len(df.columns)} cols")
            except Exception as e:
                logger.warning(f"Query '{label}' failed: {e}")
                print(f"  [ERROR] Query '{label}': {e}")

        target_tables = self.tables if self.tables else self._list_tables()
        print(f"  Loading {len(target_tables)} tables from PostgreSQL...")

        for table in target_tables:
            try:
                with engine.connect() as conn:
                    df = pd.read_sql(
                        sa_text(f'SELECT * FROM "{table}" LIMIT {config.max_db_rows}'),
                        conn
                    )
                if df.empty:
                    print(f"  Table '{table}': empty, skipped")
                    continue
                docs.append(RawDocument(
                    content=self._df_to_text(df, table),
                    source=f"table:{table}",
                    doc_type="db_table",
                    metadata={"table": table, "rows": len(df), "cols": len(df.columns),
                              "columns": list(df.columns)}
                ))
                print(f"  [OK] Table '{table}': {len(df)} rows × {len(df.columns)} cols")
            except Exception as e:
                logger.warning(f"Table '{table}' failed: {e}")
                print(f"  [ERROR] Table '{table}': {e}")

        return docs

    @staticmethod
    def _df_to_text(df: pd.DataFrame, label: str) -> str:
        """Convert a DataFrame to descriptive text that can be embedded by RAG."""
        header = (
            f"=== {label.upper()} ===\n"
            f"Columns: {', '.join(df.columns)}\n"
            f"Row count: {len(df)}\n"
        )
        rows = df.head(config.max_db_rows).to_string(index=False)
        return f"{header}\n{rows}"


class MultiSourceAdapter(BaseSourceAdapter):
    """
    Composite adapter — merges documents from multiple adapters into one list.

    Used for HYBRID mode (Scenario D: Cross-Paradigm):
      - FolderSourceAdapter  → unstructured docs (PDF press releases, TXT chat logs)
      - PostgreSQLAdapter    → structured tables (BOND_SYS relational data)
    Both are merged into a single shared FAISS index, enabling single-query
    retrieval across both data paradigms.

    Usage:
        source = "/data/docs|postgresql://user:pass@host/db"
        SourceFactory.create(source)  → MultiSourceAdapter automatically
    """

    def __init__(self, adapters: List[BaseSourceAdapter]):
        self.adapters = adapters

    def describe(self) -> str:
        return "Hybrid: " + " + ".join(a.describe() for a in self.adapters)

    def load(self) -> List[RawDocument]:
        all_docs: List[RawDocument] = []
        for i, adapter in enumerate(self.adapters, 1):
            print(f"\n  [MultiSource {i}/{len(self.adapters)}] {adapter.describe()}")
            docs = adapter.load()
            all_docs.extend(docs)
            print(f"  [MultiSource {i}/{len(self.adapters)}] → {len(docs)} documents loaded")
        print(f"\n  [MultiSource] Combined total: {len(all_docs)} documents "
              f"from {len(self.adapters)} adapters")
        return all_docs
