"""Structured data: the app's own database and the user's external databases, searched per request.

Mixin of services.rag.processor.PDFQAProcessor; the methods were moved
verbatim from that class by MS-657."""
import logging
from config import config
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models import DatabaseResult, SearchType
from services.rag.app_database import db_manager
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StructuredDataMixin:
    def initialize_database(self):
        """Initialize database connection"""
        try:
            self.db_manager = db_manager
            self._db_initialized = True
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self._db_initialized = False

    def expand_query_for_db(self, query: str) -> List[str]:
        """Expand query with database-related terms"""
        expanded_queries = [query]

        # Add lowercase version
        expanded_queries.append(query.lower())

        # Add synonyms for database-related terms
        for term, synonyms in self.db_query_expansion_terms.items():
            if term in query.lower():
                for synonym in synonyms:
                    expanded_query = query.lower().replace(term, synonym)
                    expanded_queries.append(expanded_query)

        return list(set(expanded_queries))

    def analyze_question_type(self, question: str) -> Dict[str, Any]:
        """Analyze question to determine optimal search strategy with smart table routing"""
        question_lower = question.lower()

        #Database-related keywords
        db_keywords = [
            'user', 'profile', 'customer', 'product', 'order', 'price', 
            'jumlah', 'total', 'data', 'tabel', 'table', 'database', 'sql',
            'nama', 'email', 'alamat', 'tanggal', 'date', 'harga', 'stock',
            'karyawan', 'transaksi', 'pesanan', 'siapa', 'pegawai', 'staff',
            'anggota', 'member', 'pelanggan', 'department', 'departemen'
        ]

        # Use pdf_keywords from instance (includes LPDU document terms)
        pdf_terms = getattr(self, 'pdf_keywords', [
            'dokumen', 'pdf', 'file', 'laporan', 'report', 'handbook',
            'kebijakan', 'policy', 'prosedur', 'pedoman', 'guideline',
            'kontrak', 'agreement', 'proposal'
        ])

        is_db_question = any(keyword in question_lower for keyword in db_keywords)
        is_pdf_question = any(keyword in question_lower for keyword in pdf_terms)

        # Smart table routing - determine which tables to search
        target_tables = self.get_target_tables(question_lower)
        
        # Log detection results
        logger.info(f"📊 Question analysis: is_db={is_db_question}, is_pdf={is_pdf_question}")

        # if both or unclear, use hybrid
        if (is_db_question and is_pdf_question) or (not is_db_question and not is_pdf_question):
            recommended_type = SearchType.HYBRID
            logger.info(f"🔄 Using HYBRID search (both sources)")
        elif is_db_question:
            recommended_type = SearchType.STRUCTURED
            logger.info(f"🗄️ Using STRUCTURED search (database)")
        else:
            recommended_type = SearchType.UNSTRUCTURED
            logger.info(f"📄 Using UNSTRUCTURED search (PDF)")

        return {
            "recommended_type": recommended_type,
            "is_db_question": is_db_question,
            "is_pdf_question": is_pdf_question,
            "search_terms": self.extract_search_terms(question),
            "target_tables": target_tables  # NEW: specific tables to search
        }

    def get_target_tables(self, question_lower: str) -> List[str]:
        """Determine which database tables to search based on question content"""
        target_tables = []
        table_scores: Dict[str, int] = {}
        
        # Check for person-related questions (routes to user_profiles)
        is_person_question = any(pattern in question_lower for pattern in self.person_question_patterns)
        if is_person_question:
            table_scores["user_profiles"] = table_scores.get("user_profiles", 0) + 2  # Higher weight
        
        # Score based on keywords
        for table_name, keywords in self.table_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                table_scores[table_name] = table_scores.get(table_name, 0) + score
        
        if table_scores:
            # Sort by score and return tables with matches
            sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
            target_tables = [table for table, score in sorted_tables]
            logger.info(f"Smart routing: targeting tables {target_tables} (scores: {table_scores})")
        else:
            # No specific match, search all tables (fallback)
            target_tables = list(self.table_keywords.keys())
            logger.info(f"No specific table match, searching all: {target_tables}")
        
        return target_tables

    def extract_search_terms(self, question: str) -> Dict[str, Any]:
        """Enhanced search terms extraction dengan entity recognition"""
        import re
        
        # Entity patterns - UPDATED to include more software/brand names
        patterns = {
            'product_names': r'\b(Laptop|Smartphone|Chair|Software|Mouse|ThinkPad|Galaxy|Ergonomic|Project|JetBrains|SAP|Jira|Confluence|AWS|Azure|Datadog|SonarQube|GitLab|Jenkins|Docker|Kubernetes)\b',
            'departments': r'\b(IT|HR|Finance|Marketing|Sales|Operations)\b',
            'positions': r'\b(Engineer|Manager|Analyst|Specialist|Administrator|Director)\b',
            'currencies': r'Rp\s*\d+[.,]\d+|\d+\s*(juta|ribu)',
            'numbers': r'\b\d+\b',
            'dates': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}',
        }
        
        entities = {}
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        # Extract keywords (improved version)
        stop_words = {'apa', 'siapa', 'dimana', 'kapan', 'berapa', 'bagaimana', 
                    'yang', 'dan', 'atau', 'di', 'ke', 'dari', 'dalam', 'untuk'}
        
        question_lower = question.lower()
        words = re.findall(r'\b[\w+]+\b', question_lower)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Detect if query contains specific field references
        field_refs = []
        field_mapping = {
            'harga': 'price', 'nama': 'name', 'email': 'email', 
            'departemen': 'department', 'posisi': 'position',
            'jumlah': 'quantity', 'stok': 'stock_quantity', 'status': 'status'
        }
        
        for idn_field, eng_field in field_mapping.items():
            if idn_field in question_lower:
                field_refs.append(eng_field)
        
        return {
            "keywords": list(set(keywords)),
            "entities": entities,
            "field_references": field_refs,
            "original_terms": words
        }

    def query_structured_data(self, search_terms: List[str], target_tables: Optional[List[str]] = None) -> Dict[str, DatabaseResult]:
        """Query structured data from database with query expansion and smart routing"""
        logger.info(f"📊 query_structured_data called with terms: {search_terms}, tables: {target_tables}")
        logger.info(f"📊 _db_initialized: {self._db_initialized}")
        
        if not self._db_initialized:
            logger.warning("⚠️ Database not initialized, returning empty results")
            return {}

        try:
            # Apply query expansion to search terms
            expanded_terms = self.expand_search_terms_for_db(search_terms)
            logger.info(f"🔍 Original terms: {search_terms} -> Expanded: {expanded_terms}")
            
            # Use target tables if provided, otherwise search all
            tables_to_search = target_tables if target_tables else config.db_tables
            
            db_results = self.db_manager.search_in_specific_tables(
                expanded_terms, 
                tables_to_search,
                limit=config.db_result_limit
            )

            formatted_results = {}
            for table_name, records in db_results.items():
                formatted_results[table_name] = DatabaseResult(
                    table=table_name,
                    data=records,
                    record_count=len(records)
                )

            return formatted_results
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return {}

    def expand_search_terms_for_db(self, search_terms: List[str]) -> List[str]:
        """Expand search terms with synonyms and stemming for better DB matches"""
        expanded = set(search_terms)
        
        # Brand/product names that should NOT be stemmed
        brand_whitelist = {
            'jetbrains', 'jira', 'confluence', 'aws', 'azure', 'datadog', 
            'sonarqube', 'gitlab', 'jenkins', 'docker', 'kubernetes', 'sap'
        }
        
        for term in search_terms:
            term_lower = term.lower()
            
            # Add synonyms from expansion dictionary
            for key, synonyms in self.db_query_expansion_terms.items():
                if key in term_lower or term_lower in synonyms:
                    expanded.add(key)
                    expanded.update(synonyms)
            
            # Apply simple Indonesian stemming ONLY if not a brand name
            if term_lower not in brand_whitelist:
                stemmed = self.simple_indonesian_stem(term_lower)
                if stemmed != term_lower:
                    expanded.add(stemmed)
        
        return list(expanded)

    def simple_indonesian_stem(self, word: str) -> str:
        """Simple Indonesian stemming - remove common affixes"""
        word = word.lower().strip()
        
        # Common Indonesian suffixes
        suffixes = ['kan', 'an', 'i', 'nya', 'lah', 'kah']
        prefixes = ['me', 'di', 'ke', 'se', 'ber', 'ter', 'pe']
        
        # Remove suffixes first
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break
        
        # Then remove prefixes
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                word = word[len(prefix):]
                break
        
        return word

    # ── External (user-connected) database retrieval ────────────────────────
    # This is intentionally separate from self.db_manager / query_structured_data
    # above: that path is a single hardcoded connection used for THIS app's own
    # data. This path opens a fresh connection per active connection the user
    # set up in Sources > Database, schema-agnostic (no ALLOWED_TABLES
    # whitelist) — mirrors PostgreSQLAdapter's "all tables" mode from the
    # research notebook.

    def _serialize_table_rows(self, table_name: str, columns: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> str:
        # Markdown table, not "col: value | col: value" prose lines. LLMs are
        # heavily trained on markdown tables and parse/attend to them far
        # more reliably than ad-hoc key-value dumps — verified empirically:
        # the pipe-delimited format put the exact right row in context and
        # the model still answered "not found".
        col_names = [c["name"] for c in columns]

        def cell(value: Any) -> str:
            text = "" if value is None else str(value)
            return text.replace("|", "\\|").replace("\n", " ")

        lines = [
            f"Tabel: {table_name}",
            "",
            "| " + " | ".join(col_names) + " |",
            "|" + "|".join(["---"] * len(col_names)) + "|",
        ]
        for row in rows:
            lines.append("| " + " | ".join(cell(row.get(k)) for k in col_names) + " |")
        return "\n".join(lines)

    def _load_external_db_documents(self, db_connections: List[Dict[str, Any]]) -> List[Document]:
        from services.database_connections import open_external_connection, list_tables_with_columns

        documents: List[Document] = []

        for conn_info in db_connections:
            connection_id = conn_info.get("connection_id", "")
            label = conn_info.get("label") or connection_id or "Database"
            url = conn_info.get("url", "")
            if not url:
                continue

            try:
                ext_conn = open_external_connection(url)
            except Exception as exc:
                logger.warning("External DB connect failed for %s: %s", label, exc)
                continue

            try:
                tables = list_tables_with_columns(ext_conn, max_tables=config.external_db_max_tables)
                for table in tables:
                    table_name = table["name"]
                    columns = table.get("columns") or []
                    if not columns:
                        continue
                    safe_name = '"' + table_name.replace('"', '""') + '"'
                    try:
                        with ext_conn.cursor() as cur:
                            cur.execute(
                                f"SELECT * FROM {safe_name} LIMIT %s",
                                (config.external_db_max_rows_per_table,),
                            )
                            rows = cur.fetchall() or []
                    except Exception as exc:
                        logger.warning("External DB table read failed (%s.%s): %s", label, table_name, exc)
                        continue

                    if not rows:
                        continue

                    text = self._serialize_table_rows(table_name, columns, rows)
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": table_name,
                            "source_kind": "external_db",
                            "external_db_connection_id": connection_id,
                            "external_db_label": label,
                            "collection_id": f"external-db:{connection_id}" if connection_id else "external-db",
                            "row_count": len(rows),
                        },
                    ))
            finally:
                ext_conn.close()

        return documents

    def search_external_db_realtime(
        self,
        query: str,
        db_connections: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
    ) -> List[Document]:
        if not db_connections:
            return []

        documents = self._load_external_db_documents(db_connections)
        if not documents:
            logger.info("No readable tables found for active database connections")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.external_db_chunk_size,
            chunk_overlap=config.external_db_chunk_overlap,
        )
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            return []

        vector_store = FAISS.from_documents(chunks, self.embeddings)
        k_faiss = min(len(chunks), max(top_k * 3, 10))
        results_with_score = vector_store.similarity_search_with_score(query, k=k_faiss)

        faiss_ranked: List[Tuple[Document, float]] = []
        for doc, distance in results_with_score:
            faiss_ranked.append((doc, 1.0 / (1.0 + float(distance))))

        bm25 = self._build_bm25(chunks)
        fused = self._rrf_fuse(query, faiss_ranked, bm25, chunks, top_k=max(top_k, 1))

        realtime_results: List[Document] = []
        for doc, similarity_score in fused:
            if similarity_score <= 0.05:
                continue
            doc.metadata["similarity_score"] = similarity_score
            doc.metadata.setdefault("source_kind", "external_db")
            realtime_results.append(doc)

        return realtime_results[: config.total_k_results]
