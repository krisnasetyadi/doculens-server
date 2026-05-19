"""
router/agnostic.py
------------------
POST /api/v1/agnostic/query  --  Wraps AgnosticRAGPipeline behind a
HybridResponse-compatible schema so the chat-ui can talk to it without
any frontend changes.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agnostic"])

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from agnostic.config import config as ag_config
            from agnostic.generator import AnswerGenerator
            from agnostic.pipeline import AgnosticRAGPipeline

            ag_config.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
            gen = AnswerGenerator()
            gen.load_gemini()
            _pipeline = AgnosticRAGPipeline(generator=gen)
            logger.info("AgnosticRAGPipeline initialised.")
        except Exception as e:
            logger.error("Pipeline init failed: %s", e)
            raise RuntimeError(f"Pipeline init failed: {e}") from e
    return _pipeline


# ---------------------------------------------------------------------------
# Request schema  (hybrid-compatible + optional agnostic extras)
# ---------------------------------------------------------------------------

class AgnosticQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

    # Optional: explicit source path/URL.
    # When omitted the router falls back to the shared PDF uploads folder.
    source: Optional[str] = Field(
        None,
        description=(
            "Data source. "
            "Folder path, postgresql://..., or 'folder|postgres' for hybrid. "
            "Defaults to the service PDF uploads directory."
        ),
    )

    # Fields forwarded by the chat-ui (ignored by agnostic pipeline, kept for
    # schema compatibility so FastAPI does not return 422).
    include_pdf_results:  Optional[bool] = True
    include_db_results:   Optional[bool] = False
    include_chat_results: Optional[bool] = False
    llm_provider:         Optional[str]  = None
    llm_model:            Optional[str]  = None

    # Agnostic extras
    pg_tables:        Optional[List[str]]      = None
    pg_queries:       Optional[Dict[str, str]] = None
    exclude_patterns: Optional[List[str]]      = None


# ---------------------------------------------------------------------------
# Response schema  (HybridResponse-compatible)
# ---------------------------------------------------------------------------

class PdfSourceDetail(BaseModel):
    file_name:        str
    collection_id:    str
    relevance_score:  Optional[float] = None
    content_preview:  Optional[str]   = None


class AgnosticQueryResponse(BaseModel):
    # Core fields expected by chat-ui
    answer:               str
    model_used:           str
    pdf_sources:          List[str]
    pdf_sources_detailed: List[PdfSourceDetail]
    db_results:           Dict[str, Any]
    chat_results:         List[Any]
    processing_time:      float
    search_terms:         List[str]
    target_tables:        List[str]

    # Agnostic extras (ignored by ui but useful for debugging)
    source_type:     str
    retrieved_count: int


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

_DEFAULT_SOURCE = os.getenv("PDF_UPLOADS_DIR", "/app/data/uploads")


@router.post("/agnostic/query", response_model=AgnosticQueryResponse)
async def agnostic_query(req: AgnosticQueryRequest):
    source = req.source or _DEFAULT_SOURCE

    try:
        pipeline = _get_pipeline()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    t0 = time.time()
    try:
        result = pipeline.ask(
            question=req.question,
            source=source,
            pg_tables=req.pg_tables,
            pg_queries=req.pg_queries,
            exclude_patterns=req.exclude_patterns,
            verbose=False,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("agnostic_query failed")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = time.time() - t0

    # Map agnostic chunks -> pdf_sources_detailed
    pdf_details: List[PdfSourceDetail] = []
    pdf_sources: List[str] = []
    for chunk in result.retrieved_chunks:
        fname = os.path.basename(chunk.source)
        pdf_sources.append(fname)
        pdf_details.append(
            PdfSourceDetail(
                file_name=fname,
                collection_id=chunk.source,
                relevance_score=chunk.score,
                content_preview=chunk.content[:300],
            )
        )

    model_used = req.llm_model or result.metadata.get("model_used", "agnostic")

    return AgnosticQueryResponse(
        answer=result.answer,
        model_used=model_used,
        pdf_sources=pdf_sources,
        pdf_sources_detailed=pdf_details,
        db_results={},
        chat_results=[],
        processing_time=elapsed,
        search_terms=[req.question],
        target_tables=[],
        source_type=result.metadata.get("source_type", "unknown"),
        retrieved_count=len(result.retrieved_chunks),
    )