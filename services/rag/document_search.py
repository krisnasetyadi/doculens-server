"""Searching PDF collections: query expansion, cross-collection search, context cleanup, the plain answer path and whole-document summaries.

Mixin of services.rag.processor.PDFQAProcessor; the methods were moved
verbatim from that class by MS-657."""
import logging
from config import config
from langchain.schema import Document
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DocumentSearchMixin:
    def expand_query(self, query):
        """Expand query with synonyms and related terms"""
        expanded_queries = [query]

        # Add lowercase version
        expanded_queries.append(query.lower())

        # Add synonyms for common terms
        for term, synonyms in self.query_expansion_terms.items():
            if term in query.lower():
                for synonym in synonyms:
                    expanded_query = query.lower().replace(term, synonym)
                    expanded_queries.append(expanded_query)
                    # Also try with original case
                    expanded_queries.append(query.replace(term, synonym))

        return list(set(expanded_queries))

    def search_across_collections(self, query, collection_ids=None, top_k=5):
        """Enhanced search with query expansion and better scoring"""
        if collection_ids is None:
            collection_ids = self.get_all_collections()

        if not collection_ids:
            logger.warning("❌ No collections available for search")
            return []

        logger.info(f"🔍 Starting search across {len(collection_ids)} collections")
        logger.info(f"📝 Query: '{query}'")
        logger.info(f"📚 Collections: {collection_ids}")

        # Expand query for better retrieval
        expanded_queries = self.expand_query(query)
        logger.info(f"🔄 Expanded queries: {expanded_queries}")

        all_results = []
        successful_collections = 0
        
        for expanded_query in expanded_queries:
            # Embed the query text once per expansion variant instead of once
            # per (expansion x collection) pair — the embedding is identical
            # across collections, only the vector store being searched differs.
            query_vector = self.embeddings.embed_query(expanded_query)
            for collection_id in collection_ids:
                logger.info(f"🔍 Searching collection {collection_id} with query: '{expanded_query}'")
                vector_store = self.get_vector_store(collection_id)
                if vector_store:
                    successful_collections += 1
                    try:
                        # Use similarity_search_with_score_by_vector which returns (doc, distance)
                        # Lower distance = more similar
                        results_with_score = vector_store.similarity_search_with_score_by_vector(
                            query_vector, k=top_k
                        )
                        logger.info(f"📄 Found {len(results_with_score)} results")

                        faiss_ranked = []
                        for doc, distance in results_with_score:
                            # Convert L2 distance to similarity score (0-1 range)
                            similarity_score = 1.0 / (1.0 + float(distance))
                            faiss_ranked.append((doc, similarity_score))

                        # Fuse with lexical BM25 ranking (RRF) so exact-term
                        # matches are not drowned out by dense similarity.
                        bm25, corpus_docs = self.get_bm25_index(collection_id)
                        fused = self._rrf_fuse(
                            expanded_query, faiss_ranked, bm25, corpus_docs, top_k=top_k
                        )

                        for doc, similarity_score in fused:
                            # Accept all results with reasonable similarity (> 0.05)
                            if similarity_score > 0.05:
                                doc.metadata["collection_id"] = collection_id
                                doc.metadata["similarity_score"] = similarity_score
                                doc.metadata["matched_query"] = expanded_query
                                all_results.append((doc, similarity_score))
                                logger.debug(f"✅ Added result with similarity {similarity_score:.4f}")
                            else:
                                logger.debug(f"⏭️ Skipped result with low similarity {similarity_score:.4f}")
                    except Exception as e:
                        logger.error(f"❌ Search failed for {collection_id}: {str(e)}")
                        continue
                else:
                    logger.error(f"❌ Failed to load vector store for {collection_id}")

        logger.info(f"📊 Search summary: {successful_collections}/{len(collection_ids)} collections loaded, {len(all_results)} total results")

        # Remove duplicates and sort by score
        unique_results = {}
        for doc, score in all_results:
            content = doc.page_content if isinstance(doc.page_content, str) else str(doc.page_content)
            content_hash = hash(content[:100])  # Hash first 100 chars
            if content_hash not in unique_results or score > unique_results[content_hash][1]:
                unique_results[content_hash] = (doc, score)

        sorted_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)
        
        import re
        processed_results = []
        for doc, _score in sorted_results:
            original_content = doc.page_content
            # Backfill page metadata from [PAGE x] if page is missing/empty/generic
            if "page" not in doc.metadata or doc.metadata["page"] in ["?", "Unknown", None]:
                match = re.search(r'\[PAGE\s+(\d+)\]', original_content)
                if match:
                    doc.metadata["page"] = match.group(1)
            
            # Clean any [PAGE x] markers from content text
            cleaned_content = re.sub(r'\[PAGE\s+\d+\]', '', original_content).strip()
            
            # Skip if the chunk has no useful content remaining
            if not cleaned_content or len(cleaned_content) < 10:
                continue
                
            doc.page_content = cleaned_content
            processed_results.append(doc)
            
        final_results = processed_results[:config.total_k_results]
        
        logger.info(f"🎯 Final results after deduplication: {len(final_results)}")
        return final_results

    def extract_file_reference(self, query: str) -> Optional[str]:
        """Extract file name reference from query"""
        import re
        query_lower = query.lower()
        
        # Pattern: "dari file X", "file X", "di X.txt", "tentang X", etc
        patterns = [
            r'dari file\s+([\w_-]+)',
            r'file\s+([\w_-]+)',
            r'di\s+([\w_-]+\.txt)',
            r'tentang\s+([\w_-]+)',
            r'([\w_-]+\.txt)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                file_ref = match.group(1)
                logger.info(f"📎 Detected file reference in query: {file_ref}")
                return file_ref
        
        return None

    # Summary of whole collections (task="summarize"): chunks spread evenly
    # from start to end, instead of the top similarity hits for the literal
    # words "summarize X" — which matched whichever chunk happened to contain
    # the word "Summary" and produced "the information is very limited".
    SUMMARY_MAX_CHUNKS = 12

    SUMMARY_MAX_CHARS = 24000

    def build_summary_results(self, collection_ids: List[str]) -> Dict[str, Any]:
        """A hybrid_search-shaped result for summarizing `collection_ids`, so
        it flows through generate_hybrid_answer (task="summarize") and the
        router's source listing unchanged."""
        docs: List[Document] = []
        for cid in collection_ids:
            vector_store = self.get_vector_store(cid)
            if not vector_store:
                continue
            try:
                stored = list(vector_store.docstore._dict.values())
            except Exception as e:
                logger.warning(f"build_summary_results: could not read docstore for {cid}: {e}")
                continue
            for doc in stored:
                doc.metadata.setdefault("collection_id", cid)
            docs.extend(stored)

        if len(docs) > self.SUMMARY_MAX_CHUNKS:
            step = len(docs) / self.SUMMARY_MAX_CHUNKS
            docs = [docs[int(i * step)] for i in range(self.SUMMARY_MAX_CHUNKS)]
        per_chunk = max(500, self.SUMMARY_MAX_CHARS // max(1, len(docs)))

        merged = [
            {
                "type": "pdf",
                "content": doc.page_content[:per_chunk],
                "source": doc.metadata.get("source", "dokumen"),
                "confidence": 1.0,
                "score": 1.0,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]
        return {
            "pdf_documents": docs,
            "merged_results": merged,
            "search_analysis": {"intent": {}, "source_weights": {"pdf": 1}},
            "search_terms": [],
            "target_tables": [],
        }
