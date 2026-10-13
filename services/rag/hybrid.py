"""hybrid_search: every selected source searched and ranked together.

Mixin of services.rag.processor.PDFQAProcessor; the methods were moved
verbatim from that class by MS-657."""
import logging
from concurrent.futures import ThreadPoolExecutor
from config import config
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HybridSearchMixin:
    # services/rag/processor.py - perbaiki hybrid_search
    def hybrid_search(
        self,
        question: str,
        collection_ids: Optional[List[str]] = None,
        include_chat: bool = True,
        include_pdf: bool = True,
        include_db: bool = True,
        chat_collection_ids: Optional[List[str]] = None,
        include_public_links: bool = False,
        public_link_sources: Optional[List[Dict[str, Any]]] = None,
        include_external_db: bool = False,
        external_db_connections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Enhanced hybrid search dengan context-aware aggregation"""
        
        # Initialize database if needed
        if include_db and not self._db_initialized:
            logger.info("🔧 Initializing database connection...")
            self.initialize_database()
        
        # Step 1: Deep analysis
        intent_analysis = self.analyze_intent(question)
        search_terms_info = self.extract_search_terms(question)
        search_terms = search_terms_info["keywords"]
        
        logger.info(f"🎯 Intent: {intent_analysis['primary_intent']}")
        logger.info(f"🔍 Search terms: {search_terms}")
        logger.info(f"🏷️ Entities: {search_terms_info.get('entities', {})}")
        
        # Step 2: Smart source prioritization berdasarkan intent
        if intent_analysis["is_aggregation"]:
            # Untuk aggregasi, prioritaskan database
            db_weight = 1.0
            pdf_weight = 0.3
            chat_weight = 0.2
        elif intent_analysis["is_explanation"]:
            # Untuk penjelasan, prioritaskan PDF
            pdf_weight = 1.0
            db_weight = 0.4
            chat_weight = 0.5
        elif intent_analysis["is_data_retrieval"]:
            # Untuk data retrieval, balance semua
            db_weight = 0.7
            pdf_weight = 0.6
            chat_weight = 0.5
        else:
            # Default weights
            db_weight = 0.6
            pdf_weight = 0.6
            chat_weight = 0.5
        public_link_weight = pdf_weight
        external_db_weight = db_weight

        results: Dict[str, Any] = {}

        # Step 3: Parallel search dengan weights.
        # Each source is independent I/O/CPU work, so the sub-searches run
        # concurrently in a small thread pool instead of one after another.
        # hybrid_search already executes off the event-loop thread (called via
        # asyncio.to_thread), so this is a plain ThreadPoolExecutor, not asyncio.
        def _search_pdf() -> tuple:
            logger.info(f"📄 Searching PDF (weight: {pdf_weight})...")
            pdf_docs = self.search_across_collections(
                question,
                collection_ids=collection_ids,
                top_k=int(config.k_per_collection * pdf_weight)
            )
            for doc in pdf_docs:
                if 'similarity_score' in doc.metadata:
                    doc.metadata['similarity_score'] *= pdf_weight
            return 'pdf_documents', pdf_docs

        def _search_db() -> tuple:
            logger.info(f"🗄️ Searching DB (weight: {db_weight})...")
            target_tables = self.get_target_tables(question.lower())
            db_results = self.query_structured_data(search_terms, target_tables)
            for db_result in db_results.values():
                if hasattr(db_result, 'data'):
                    for record in db_result.data:
                        if 'relevance_score' in record:
                            record['relevance_score'] *= db_weight
            return 'database_results', db_results

        def _search_chat() -> tuple:
            logger.info(f"💬 Searching chat (weight: {chat_weight})...")
            file_filter = self.extract_file_reference(question)
            # Floor of 6: chat chunks are tiny (chat_chunk_size=300), so a
            # weighted k of 1-2 chunks gives the LLM almost no conversation
            # context to work with.
            chat_docs = self.search_across_chat_collections(
                question,
                collection_ids=chat_collection_ids,
                file_filter=file_filter,
                top_k=max(int(config.k_per_collection * chat_weight), 6)
            )
            for doc in chat_docs:
                if 'similarity_score' in doc.metadata:
                    doc.metadata['similarity_score'] *= chat_weight
            return 'chat_documents', chat_docs

        def _search_public_links() -> tuple:
            logger.info(f"🌐 Searching active public links in realtime (weight: {public_link_weight})...")
            # Wider top_k than k_per_collection: public links span many files,
            # and the answer stage can consume up to 8 snippets.
            public_link_docs = self.search_public_links_realtime(
                question,
                public_link_sources=public_link_sources,
                top_k=max(int(config.k_per_collection * public_link_weight), 8),
            )
            for doc in public_link_docs:
                if 'similarity_score' in doc.metadata:
                    doc.metadata['similarity_score'] *= public_link_weight
            return 'public_link_documents', public_link_docs

        def _search_external_db() -> tuple:
            logger.info(f"🗄️ Searching active external database connections in realtime (weight: {external_db_weight})...")
            external_db_docs = self.search_external_db_realtime(
                question,
                db_connections=external_db_connections,
                top_k=max(int(config.k_per_collection * external_db_weight), 8),
            )
            for doc in external_db_docs:
                if 'similarity_score' in doc.metadata:
                    doc.metadata['similarity_score'] *= external_db_weight
            return 'external_db_documents', external_db_docs

        search_tasks = []
        if include_pdf and pdf_weight > 0:
            search_tasks.append(_search_pdf)
        if include_db and db_weight > 0:
            search_tasks.append(_search_db)
        if include_chat and chat_weight > 0:
            search_tasks.append(_search_chat)
        if include_public_links and public_link_weight > 0 and public_link_sources:
            search_tasks.append(_search_public_links)
        if include_external_db and external_db_weight > 0 and external_db_connections:
            search_tasks.append(_search_external_db)

        if search_tasks:
            with ThreadPoolExecutor(max_workers=len(search_tasks)) as executor:
                futures = [executor.submit(task) for task in search_tasks]
                for future in futures:
                    key, value = future.result()
                    results[key] = value

        # Step 4: Cross-source deduplication and ranking
        all_results = self.merge_and_rank_results(results, intent_analysis)

        return {
            **results,
            "search_analysis": {
                "intent": intent_analysis,
                "search_terms": search_terms_info,
                "source_weights": {
                    "pdf": pdf_weight if include_pdf else 0,
                    "db": db_weight if include_db else 0,
                    "chat": chat_weight if include_chat else 0,
                    "public_link": public_link_weight if include_public_links else 0,
                    "external_db": external_db_weight if include_external_db else 0,
                }
            },
            "search_terms": search_terms,
            "target_tables": self.get_target_tables(question.lower()) if include_db else [],
            "merged_results": all_results,
            "has_conflicts": self.detect_conflicts(all_results if isinstance(all_results, list) else [])
        }

    def merge_and_rank_results(self, results: Dict[str, Any], intent_analysis: Dict) -> List[Dict]:
        """Merge results from all sources dengan ranking yang cerdas"""
        merged = []
        
        # Extract all results dengan unified format
        if 'pdf_documents' in results:
            for doc in results['pdf_documents']:
                merged.append({
                    'type': 'pdf',
                    'content': doc.page_content,
                    'score': doc.metadata.get('similarity_score', 0),
                    'metadata': doc.metadata,
                    'source': f"PDF: {doc.metadata.get('source', 'Unknown')}",
                    'confidence': doc.metadata.get('similarity_score', 0) * 0.8  # PDF confidence factor
                })
        
        if 'database_results' in results:
            for table_name, db_result in results['database_results'].items():
                if hasattr(db_result, 'data'):
                    for record in db_result.data:
                        # Create readable content from record
                        content_parts = []
                        for key, value in record.items():
                            if key not in ['search_vector', 'relevance_score']:
                                content_parts.append(f"{key}: {value}")
                        
                        merged.append({
                            'type': 'database',
                            'content': "\n".join(content_parts),
                            'score': record.get('relevance_score', 0),
                            'metadata': {'table': table_name, 'record': record},
                            'source': f"DB: {table_name}",
                            'confidence': record.get('relevance_score', 0) * 0.9  # DB confidence factor
                        })
        
        if 'chat_documents' in results:
            for doc in results['chat_documents']:
                merged.append({
                    'type': 'chat',
                    'content': doc.page_content,
                    'score': doc.metadata.get('similarity_score', 0),
                    'metadata': doc.metadata,
                    'source': f"Chat: {doc.metadata.get('source', 'Unknown')}",
                    'confidence': doc.metadata.get('similarity_score', 0) * 0.7  # Chat confidence factor
                })

        if 'public_link_documents' in results:
            for doc in results['public_link_documents']:
                merged.append({
                    'type': 'public_link',
                    'content': doc.page_content,
                    'score': doc.metadata.get('similarity_score', 0),
                    'metadata': doc.metadata,
                    'source': f"Public Link: {doc.metadata.get('public_link_title', doc.metadata.get('source', 'Unknown'))}",
                    'confidence': doc.metadata.get('similarity_score', 0) * 0.85,
                })

        if 'external_db_documents' in results:
            for doc in results['external_db_documents']:
                merged.append({
                    'type': 'external_db',
                    'content': doc.page_content,
                    'score': doc.metadata.get('similarity_score', 0),
                    'metadata': doc.metadata,
                    'source': f"DB: {doc.metadata.get('external_db_label', 'Database')} / {doc.metadata.get('source', 'Unknown')}",
                    'confidence': doc.metadata.get('similarity_score', 0) * 0.9,  # DB confidence factor
                })

        # Sort by confidence score
        merged.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply intent-based boosting
        if intent_analysis['is_aggregation']:
            # Boost database results for aggregation
            for item in merged:
                if item['type'] in ('database', 'external_db'):
                    item['confidence'] *= 1.5
        elif intent_analysis['is_explanation']:
            # Boost PDF results for explanation
            for item in merged:
                if item['type'] == 'pdf':
                    item['confidence'] *= 1.5
        
        # Re-sort after boosting
        merged.sort(key=lambda x: x['confidence'], reverse=True)
        
        return merged[:20]  # Return top 20 results

    def detect_conflicts(self, merged_results: List[Dict[str, Any]]) -> List[Dict]:
        """Enhanced conflict detection - identifies contradictions between sources"""
        conflicts = []
        
        # Extract entities and values from all sources
        entity_values: Dict[str, List[Dict[str, Any]]] = {}  # {entity_key: [{value, source, type, confidence}]}
        
        # 1. Extract numerical values (prices, quantities, counts)
        for result in merged_results:
            content = result.get('content', '')
            source = result.get('source', 'Unknown')
            source_type = result.get('type', 'unknown')
            confidence = result.get('confidence', 0)
            
            # Extract numbers with context (price, quantity, count)
            import re
            
            # Price patterns: "Rp 500000", "$100", "500 ribu"
            price_matches = re.findall(r'(?:Rp\s*|\$)?([0-9.,]+)\s*(?:ribu|juta|rb|jt|k|million)?', content, re.IGNORECASE)
            for match in price_matches:
                key = f"price_{source_type}"
                if key not in entity_values:
                    entity_values[key] = []
                entity_values[key].append({
                    'value': match,
                    'source': source,
                    'type': source_type,
                    'confidence': confidence
                })
            
            # Name conflicts: different person names for same role
            # Pattern: "IT Manager: Ahmad" vs "IT Manager: Budi"
            role_name_matches = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', content)
            for role, name in role_name_matches:
                key = f"role_{role.lower().replace(' ', '_')}"
                if key not in entity_values:
                    entity_values[key] = []
                entity_values[key].append({
                    'value': name,
                    'source': source,
                    'type': source_type,
                    'confidence': confidence
                })
        
        # 2. Detect conflicts: same entity with different values
        for entity_key, values in entity_values.items():
            if len(values) > 1:
                unique_values: Dict[str, List[Dict[str, Any]]] = {}
                for item in values:
                    val = item['value']
                    if val not in unique_values:
                        unique_values[val] = []
                    unique_values[val].append(item)
                
                # If multiple distinct values exist for same entity
                if len(unique_values) > 1:
                    conflicts.append({
                        'entity': entity_key,
                        'type': 'value_mismatch',
                        'values': [
                            {
                                'value': k,
                                'sources': [i['source'] for i in v],
                                'source_types': [i['type'] for i in v],
                                'avg_confidence': sum(i['confidence'] for i in v) / len(v)
                            }
                            for k, v in unique_values.items()
                        ],
                        'message': f"Conflicting information for {entity_key}: {list(unique_values.keys())}",
                        'resolution': f"Prioritize {max(unique_values.items(), key=lambda x: sum(i['confidence'] for i in x[1]))[0]} (highest confidence)"
                    })
        
        # 3. Semantic conflicts: contradictory statements
        for i, result1 in enumerate(merged_results[:3]):
            for result2 in merged_results[i+1:4]:
                if result1['type'] != result2['type']:  # Only check across different source types
                    # Check for negation conflicts
                    content1_lower = result1['content'].lower()
                    content2_lower = result2['content'].lower()
                    
                    # Simple contradiction detection
                    if ('tidak' in content1_lower or 'no' in content1_lower) and \
                       ('tidak' not in content2_lower and 'no' not in content2_lower):
                        # Potential contradiction
                        conflicts.append({
                            'type': 'semantic_contradiction',
                            'sources': [result1['source'], result2['source']],
                            'source_types': [result1['type'], result2['type']],
                            'message': f"Potential contradiction between {result1['type']} and {result2['type']}",
                            'resolution': f"Prioritize {result1['type']} (higher confidence: {result1['confidence']:.2%})"
                        })
        
        return conflicts
