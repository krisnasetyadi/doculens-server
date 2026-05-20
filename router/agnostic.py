"""
router/agnostic.py
------------------
POST /api/v1/agnostic/query  --  Delegates to the existing processor
(same FAISS-indexed collections as hybrid.py) so queries actually work
against uploaded documents.

The "agnostic" label is preserved for UI compatibility; internally this
is the same hybrid-search path with a HybridResponse-compatible schema.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from config import config
from processor import processor

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agnostic"])



# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class AgnosticQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

    # Kept for schema compatibility with HF Space / old clients — ignored locally.
    source: Optional[str] = Field(None)

    include_pdf_results:  Optional[bool] = True
    include_db_results:   Optional[bool] = False
    include_chat_results: Optional[bool] = False
    llm_provider:         Optional[str]  = None
    llm_model:            Optional[str]  = None

    # Optional collection selectors (defaults to "all")
    pdf_collection_ids:  Optional[List[str]] = None
    chat_collection_ids: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Response schema  (HybridResponse-compatible)
# ---------------------------------------------------------------------------

class PdfSourceDetail(BaseModel):
    file_name:        str
    collection_id:    str
    relevance_score:  Optional[float] = None
    content_preview:  Optional[str]   = None


class AgnosticQueryResponse(BaseModel):
    answer:               str
    model_used:           str
    pdf_sources:          List[str]
    pdf_sources_detailed: List[PdfSourceDetail]
    db_results:           Dict[str, Any]
    chat_results:         List[Any]
    processing_time:      float
    search_terms:         List[str]
    target_tables:        List[str]
    source_type:          str = "Indexed Collections"
    retrieved_count:      int = 0


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/agnostic/query", response_model=AgnosticQueryResponse)
async def agnostic_query(req: AgnosticQueryRequest, request: Request):
    start_time = datetime.now()
    logger.info("agnostic_query: question=%r", req.question)

    try:
        # Resolve collections
        pdf_collection_ids = req.pdf_collection_ids
        if not pdf_collection_ids and req.include_pdf_results:
            pdf_collection_ids = processor.get_all_collections()
            logger.info("agnostic_query: %d PDF collection(s) available", len(pdf_collection_ids))

        chat_collection_ids = req.chat_collection_ids
        if not chat_collection_ids and req.include_chat_results:
            chat_collection_ids = processor.get_all_chat_collections()

        should_search_pdfs = req.include_pdf_results and bool(pdf_collection_ids)
        should_search_db   = req.include_db_results
        should_search_chat = req.include_chat_results and bool(chat_collection_ids)

        # Run hybrid search against pre-built FAISS indexes
        hybrid_results = await asyncio.to_thread(
            processor.hybrid_search,
            req.question,
            pdf_collection_ids or [],
            should_search_chat,
            should_search_pdfs,
            should_search_db,
            chat_collection_ids or [],
        )

        # Generate answer
        answer_result = await asyncio.to_thread(
            processor.generate_hybrid_answer,
            hybrid_results,
            req.question,
            req.llm_provider,
            req.llm_model,
        )

        if isinstance(answer_result, tuple) and len(answer_result) >= 2:
            answer, model_used = answer_result[0], answer_result[1]
        else:
            answer     = str(answer_result)
            model_used = req.llm_model or config.model_name

        # Map PDF docs -> response fields
        pdf_sources: List[str] = []
        pdf_sources_detailed: List[PdfSourceDetail] = []
        for doc in hybrid_results.get("pdf_documents", []):
            meta  = getattr(doc, "metadata", {})
            fname = meta.get("source", "Unknown")
            page  = meta.get("page")
            pdf_sources.append(f"{fname} (Halaman {page})" if page else fname)
            pdf_sources_detailed.append(PdfSourceDetail(
                file_name=fname,
                collection_id=meta.get("collection_id", ""),
                relevance_score=meta.get("similarity_score", 0.0),
                content_preview=(doc.page_content[:300]
                                 if hasattr(doc, "page_content") else ""),
            ))

        chat_results = []
        for doc in hybrid_results.get("chat_documents", []):
            meta = getattr(doc, "metadata", {})
            chat_results.append({
                "source":          meta.get("source", "Unknown"),
                "platform":        meta.get("platform", "chat"),
                "relevance_score": meta.get("similarity_score", 0),
                "content_preview": (doc.page_content[:200]
                                    if hasattr(doc, "page_content") else ""),
            })

        elapsed = (datetime.now() - start_time).total_seconds()

        return AgnosticQueryResponse(
            answer=answer,
            model_used=model_used,
            pdf_sources=pdf_sources,
            pdf_sources_detailed=pdf_sources_detailed,
            db_results=hybrid_results.get("database_results", {}),
            chat_results=chat_results,
            processing_time=elapsed,
            search_terms=hybrid_results.get("search_terms", [req.question]),
            target_tables=hybrid_results.get("target_tables", []),
            source_type="Indexed Collections",
            retrieved_count=len(pdf_sources) + len(chat_results),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("agnostic_query failed")
        raise HTTPException(status_code=500, detail=str(e))