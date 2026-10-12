"""Searching chat collections (WhatsApp, Telegram).

Mixin of services.rag.processor.PDFQAProcessor; the methods were moved
verbatim from that class by MS-657."""
import logging
from config import config
from typing import List, Optional

logger = logging.getLogger(__name__)


class ChatSearchMixin:
    def _load_collection_keywords(self, collection_id: str) -> List[str]:
        """Load saved keywords from collection metadata"""
        import json
        import os
        
        metadata_path = os.path.join(config.chat_index_folder, collection_id, "metadata.json")
        
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    keywords = metadata.get('keywords', [])
                    if keywords:
                        logger.debug(f"📚 Loaded {len(keywords)} keywords for {collection_id}")
                        return keywords
        except Exception as e:
            logger.warning(f"Failed to load keywords for {collection_id}: {e}")
        
        return []  # Return empty list if no keywords found

    def _expand_chat_query(self, query: str) -> List[str]:
        """Expand query with variations for better chat search"""
        queries = [query]
        query_lower = query.lower()
        
        # Common Indonesian synonyms/variations for chat context
        expansions = {
            'nik': ['nomor induk karyawan', 'employee id', 'id karyawan'],
            'cuti': ['leave', 'libur', 'off', 'saldo cuti'],
            'sisa': ['remaining', 'tersisa', 'balance'],
            'berapa': ['what is', 'jumlah', 'total'],
            'nama': ['name', 'siapa'],
            'email': ['alamat email', 'mail'],
            'telepon': ['phone', 'hp', 'nomor telepon'],
        }
        
        # Add expanded queries
        for term, synonyms in expansions.items():
            if term in query_lower:
                for syn in synonyms[:2]:  # Limit to avoid too many queries
                    expanded = query_lower.replace(term, syn)
                    if expanded not in queries:
                        queries.append(expanded)
        
        return queries[:3]  # Limit to 3 queries max

    def search_across_chat_collections(
        self, 
        question: str, 
        collection_ids: Optional[List[str]] = None,
        file_filter: Optional[str] = None,
        top_k: int = 5
    ) -> List:
        """Search across chat collections with optional file filtering"""
        if collection_ids is None:
            collection_ids = self.get_all_chat_collections()
        
        # Auto-detect file reference if not explicitly provided
        if file_filter is None:
            file_filter = self.extract_file_reference(question)
        
        if not collection_ids:
            logger.info("No chat collections available")
            return []
        
        all_results = []
        
        # Expand query for better search coverage
        search_queries = self._expand_chat_query(question)
        logger.info(f"🔍 Searching with queries: {search_queries}")
        
        # Embed each query variant once and reuse the vector across all
        # collections, instead of re-embedding the same text per collection.
        query_vectors = [(query, self.embeddings.embed_query(query)) for query in search_queries]

        for collection_id in collection_ids:
            vector_store = self.get_chat_vector_store(collection_id)
            if not vector_store:
                continue

            try:
                # Search with multiple query variations
                seen_content_hashes = set()
                relevance_score_fn = vector_store._select_relevance_score_fn()

                for query, query_vector in query_vectors:
                    docs_and_scores = vector_store.similarity_search_with_score_by_vector(
                        query_vector, k=top_k
                    )
                    results = [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]

                    for doc, score in results:
                        # Deduplicate by content hash
                        content_hash = hash(doc.page_content[:100])
                        if content_hash in seen_content_hashes:
                            continue
                        seen_content_hashes.add(content_hash)
                        
                        source_file = doc.metadata.get('source', '').lower()
                        content_lower = doc.page_content.lower()
                        
                        # Log all results for debugging
                        logger.info(f"🔍 Found: {source_file} with score {score:.3f}")
                        
                        # Apply file filter if specified
                        if file_filter and file_filter.lower() not in source_file:
                            logger.debug(f"⏭️ Skipping {source_file} - doesn't match filter: {file_filter}")
                            continue
                        
                        # Multi-level boosting strategy
                        boosted_score = score
                        boost_reasons = []
                        
                        # 1. Filename matching boost
                        if file_filter and file_filter.lower() in source_file:
                            boosted_score *= 1.8
                            boost_reasons.append("filename_match")
                        
                        # 2. Load saved keywords from metadata (dynamic!)
                        saved_keywords = self._load_collection_keywords(collection_id)
                        
                        # 3. Content keyword matching boost (use saved keywords dynamically)
                        question_lower = question.lower()
                        question_words = set(question_lower.split())
                        
                        # Check for keywords match
                        keyword_matches = 0
                        matched_keywords = []
                        
                        for keyword in saved_keywords:
                            if keyword in question_lower and keyword in content_lower:
                                keyword_matches += 1
                                matched_keywords.append(keyword)
                        
                        # Boost based on keyword density
                        if keyword_matches >= 2:
                            keyword_boost = 1.5
                            boosted_score *= keyword_boost
                            boost_reasons.append(f"{keyword_matches}_keywords[{','.join(matched_keywords[:3])}]")
                            logger.info(f"🎯 Keyword boost for {source_file}: {keyword_matches} matches ({matched_keywords[:5]})")
                        
                        if boost_reasons:
                            logger.info(f"⬆️ Boosted score for {source_file}: {score:.3f} → {boosted_score:.3f} ({', '.join(boost_reasons)})")
                        
                        # Use VERY low threshold since FAISS relevance scores can be negative
                        # The key is to compare relative scores, not absolute values
                        threshold = 0.0 if file_filter else 0.05  # Allow negative scores with filtering
                        
                        if boosted_score >= threshold or (file_filter and file_filter.lower() in source_file):
                            # If file filter matches, always include regardless of score
                            doc.metadata['similarity_score'] = float(boosted_score)
                            doc.metadata['original_score'] = float(score)
                            doc.metadata['collection_id'] = collection_id
                            all_results.append(doc)
                            logger.info(f"✅ Added {source_file} (score: {boosted_score:.3f})")
                        else:
                            logger.debug(f"⏭️ Skipped {source_file} with score {boosted_score:.3f} < {threshold}")
                        
            except Exception as e:
                logger.error(f"Error searching chat collection {collection_id}: {e}")
                continue
        
        # Re-rank results based on keyword content match with query
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        for doc in all_results:
            content_lower = doc.page_content.lower()
            
            # Count how many query words appear in content
            word_matches = sum(1 for word in question_words if word in content_lower and len(word) > 2)
            
            # Boost score based on keyword overlap
            current_score = doc.metadata.get('similarity_score', 0)
            content_boost = 1 + (word_matches * 0.2)  # 20% boost per matching word
            doc.metadata['similarity_score'] = current_score * content_boost
            doc.metadata['keyword_matches'] = word_matches
            
            if word_matches > 0:
                logger.info(f"📝 Content re-rank for {doc.metadata.get('source', '')}: {word_matches} word matches, score: {current_score:.3f} → {doc.metadata['similarity_score']:.3f}")
        
        # Sort by updated score
        all_results.sort(key=lambda x: x.metadata.get('similarity_score', 0), reverse=True)
        
        filter_info = f" (filtered by: {file_filter})" if file_filter else ""
        logger.info(f"📱 Found {len(all_results)} chat results{filter_info}")
        
        if file_filter and len(all_results) == 0:
            logger.warning(f"⚠️ No results found with file filter '{file_filter}'. Try broader search.")
        
        return all_results[:top_k * 2]  # Return more context for chats
