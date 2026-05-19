"""
app.py — Hugging Face Space entry point
Agnostic Multi-Source RAG · Gradio UI

Tabs:
  1. Documents  — upload PDF/TXT/MD/LOG → ask questions
  2. PostgreSQL — connection string + optional table filter → ask questions
  3. Hybrid     — upload files AND a Postgres connection → merged FAISS index
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import gradio as gr

# ── Bootstrap config from HF Space Secret ─────────────────────────────────
from agnostic.config import config

config.gemini_api_key = os.getenv("GEMINI_API_KEY", "")

# ── Lazy-init pipeline (one shared instance per Space replica) ─────────────
_pipeline = None
_generator = None


def _get_pipeline():
    global _pipeline, _generator
    if _pipeline is None:
        from agnostic.generator import AnswerGenerator
        from agnostic.pipeline import AgnosticRAGPipeline

        _generator = AnswerGenerator()
        ok = _generator.load_gemini()
        if not ok:
            raise RuntimeError(
                "Could not load Gemini model. "
                "Make sure GEMINI_API_KEY is set in Space Secrets."
            )
        _pipeline = AgnosticRAGPipeline(generator=_generator)
    return _pipeline


# ── Helpers ────────────────────────────────────────────────────────────────

def _format_chunks(chunks) -> str:
    if not chunks:
        return "_No chunks retrieved._"
    lines = []
    for i, c in enumerate(chunks, 1):
        src = Path(c.source).name if not c.source.startswith(("table:", "query:")) else c.source
        preview = c.content[:200].replace("\n", " ")
        lines.append(
            f"**[{i}]** `score={c.score:.3f}` · `{c.doc_type}` · `{src}`\n"
            f"> {preview}…"
        )
    return "\n\n".join(lines)


def _format_timing(timing: dict) -> str:
    if not timing:
        return ""
    rows = [f"| {k} | {v:.3f}s |" for k, v in timing.items()]
    total = sum(timing.values())
    rows.append(f"| **TOTAL** | **{total:.3f}s** |")
    return "| Stage | Time |\n|---|---|\n" + "\n".join(rows)


def _copy_uploads_to_tmpdir(files) -> str:
    """Copy Gradio-uploaded files into a fresh temp folder, return the path."""
    tmp = tempfile.mkdtemp(prefix="rag_docs_")
    for f in files:
        src = Path(f.name)
        shutil.copy2(src, Path(tmp) / src.name)
    return tmp


# ── Tab 1 — Documents ──────────────────────────────────────────────────────

def run_documents(files, question: str, gemini_key: str) -> tuple:
    if not question.strip():
        return "⚠️ Please enter a question.", "", ""
    if not files:
        return "⚠️ Please upload at least one document.", "", ""

    api_key = gemini_key.strip() or config.gemini_api_key
    if not api_key:
        return "⚠️ No Gemini API key. Enter it in the field above or set GEMINI_API_KEY in Space Secrets.", "", ""

    config.gemini_api_key = api_key

    try:
        pipeline = _get_pipeline()
    except RuntimeError as e:
        return f"❌ {e}", "", ""

    tmp_dir = _copy_uploads_to_tmpdir(files)
    try:
        result = pipeline.ask(question, source=tmp_dir, verbose=False)
        answer  = result.answer
        chunks  = _format_chunks(result.retrieved_chunks)
        timing  = _format_timing(result.timing)
        meta    = result.metadata
        info    = (
            f"**Source type:** {meta.get('source_type', '?')}  \n"
            f"**Docs loaded:** {meta.get('raw_docs', '?')}  \n"
            f"**Chunks:** {meta.get('total_chunks', '?')} total · "
            f"{meta.get('retrieved', '?')} retrieved  \n"
            f"**LLM:** {meta.get('llm', '?')}"
        )
        return answer, chunks, timing + "\n\n" + info
    except Exception as e:
        return f"❌ Error: {e}", "", ""
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Tab 2 — PostgreSQL ─────────────────────────────────────────────────────

def run_postgres(conn_str: str, tables_input: str, question: str, gemini_key: str) -> tuple:
    if not question.strip():
        return "⚠️ Please enter a question.", "", ""
    if not conn_str.strip():
        return "⚠️ Please enter a PostgreSQL connection string.", "", ""

    api_key = gemini_key.strip() or config.gemini_api_key
    if not api_key:
        return "⚠️ No Gemini API key.", "", ""

    config.gemini_api_key = api_key

    tables: Optional[List[str]] = None
    if tables_input.strip():
        tables = [t.strip() for t in tables_input.split(",") if t.strip()]

    try:
        pipeline = _get_pipeline()
        result   = pipeline.ask(question, source=conn_str, pg_tables=tables, verbose=False)
        answer   = result.answer
        chunks   = _format_chunks(result.retrieved_chunks)
        timing   = _format_timing(result.timing)
        meta     = result.metadata
        info     = (
            f"**Source type:** {meta.get('source_type', '?')}  \n"
            f"**Tables loaded:** {meta.get('raw_docs', '?')}  \n"
            f"**Chunks:** {meta.get('total_chunks', '?')} total · "
            f"{meta.get('retrieved', '?')} retrieved  \n"
            f"**LLM:** {meta.get('llm', '?')}"
        )
        return answer, chunks, timing + "\n\n" + info
    except Exception as e:
        return f"❌ Error: {e}", "", ""


# ── Tab 3 — Hybrid ─────────────────────────────────────────────────────────

def run_hybrid(files, conn_str: str, question: str, gemini_key: str) -> tuple:
    if not question.strip():
        return "⚠️ Please enter a question.", "", ""
    if not files and not conn_str.strip():
        return "⚠️ Provide at least one source (files or PostgreSQL).", "", ""

    api_key = gemini_key.strip() or config.gemini_api_key
    if not api_key:
        return "⚠️ No Gemini API key.", "", ""

    config.gemini_api_key = api_key

    try:
        pipeline = _get_pipeline()
    except RuntimeError as e:
        return f"❌ {e}", "", ""

    tmp_dir = None
    try:
        if files and conn_str.strip():
            tmp_dir = _copy_uploads_to_tmpdir(files)
            source  = f"{tmp_dir}|{conn_str.strip()}"
        elif files:
            tmp_dir = _copy_uploads_to_tmpdir(files)
            source  = tmp_dir
        else:
            source = conn_str.strip()

        result = pipeline.ask(question, source=source, verbose=False)
        answer = result.answer
        chunks = _format_chunks(result.retrieved_chunks)
        timing = _format_timing(result.timing)
        meta   = result.metadata
        info   = (
            f"**Source type:** {meta.get('source_type', '?')}  \n"
            f"**Docs loaded:** {meta.get('raw_docs', '?')}  \n"
            f"**Chunks:** {meta.get('total_chunks', '?')} total · "
            f"{meta.get('retrieved', '?')} retrieved  \n"
            f"**LLM:** {meta.get('llm', '?')}"
        )
        return answer, chunks, timing + "\n\n" + info
    except Exception as e:
        return f"❌ Error: {e}", "", ""
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── UI ─────────────────────────────────────────────────────────────────────

DESCRIPTION = """
# 🔍 Agnostic Multi-Source RAG

Ask questions across **PDF / TXT documents** and **PostgreSQL databases** — unified into one FAISS index.

> **Embedding:** `paraphrase-multilingual-MiniLM-L12-v2` (multilingual, 384-dim)  
> **LLM:** Gemini 2.5-flash with automatic fallback chain  
> **Retrieval:** FAISS in-memory · cosine similarity · top-8 chunks
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Agnostic RAG") as demo:
    gr.Markdown(DESCRIPTION)

    # Shared API key input (shown at top, used by all tabs)
    with gr.Row():
        api_key_input = gr.Textbox(
            label="🔑 Gemini API Key",
            placeholder="Paste your key here — or set GEMINI_API_KEY in Space Secrets",
            type="password",
            scale=2,
        )

    with gr.Tabs():

        # ── Tab 1: Documents ───────────────────────────────────────────────
        with gr.TabItem("📄 Documents"):
            gr.Markdown("Upload **PDF, TXT, MD, or LOG** files and ask a question.")
            with gr.Row():
                with gr.Column(scale=1):
                    doc_files = gr.File(
                        label="Upload Documents",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".md", ".log"],
                    )
                    doc_question = gr.Textbox(
                        label="Question",
                        placeholder="e.g. What are the key financial highlights?",
                        lines=2,
                    )
                    doc_btn = gr.Button("Ask →", variant="primary")
                with gr.Column(scale=2):
                    doc_answer  = gr.Markdown(label="Answer")
                    doc_chunks  = gr.Markdown(label="Retrieved Chunks")
                    doc_meta    = gr.Markdown(label="Timing & Metadata")

            doc_btn.click(
                fn=run_documents,
                inputs=[doc_files, doc_question, api_key_input],
                outputs=[doc_answer, doc_chunks, doc_meta],
            )

        # ── Tab 2: PostgreSQL ──────────────────────────────────────────────
        with gr.TabItem("🗄️ PostgreSQL"):
            gr.Markdown("Connect to a PostgreSQL database and query it with natural language.")
            with gr.Row():
                with gr.Column(scale=1):
                    pg_conn = gr.Textbox(
                        label="Connection String",
                        placeholder="postgresql://user:password@host:5432/dbname",
                    )
                    pg_tables = gr.Textbox(
                        label="Tables (optional — comma-separated, blank = all tables)",
                        placeholder="orders, customers, products",
                    )
                    pg_question = gr.Textbox(
                        label="Question",
                        placeholder="e.g. What are the top 5 customers by revenue?",
                        lines=2,
                    )
                    pg_btn = gr.Button("Ask →", variant="primary")
                with gr.Column(scale=2):
                    pg_answer = gr.Markdown(label="Answer")
                    pg_chunks = gr.Markdown(label="Retrieved Chunks")
                    pg_meta   = gr.Markdown(label="Timing & Metadata")

            pg_btn.click(
                fn=run_postgres,
                inputs=[pg_conn, pg_tables, pg_question, api_key_input],
                outputs=[pg_answer, pg_chunks, pg_meta],
            )

        # ── Tab 3: Hybrid ──────────────────────────────────────────────────
        with gr.TabItem("🔀 Hybrid"):
            gr.Markdown(
                "Combine **uploaded documents + PostgreSQL** into one merged FAISS index.  \n"
                "You can use either or both sources."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    hyb_files = gr.File(
                        label="Upload Documents (optional)",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".md", ".log"],
                    )
                    hyb_conn = gr.Textbox(
                        label="PostgreSQL Connection String (optional)",
                        placeholder="postgresql://user:password@host:5432/dbname",
                    )
                    hyb_question = gr.Textbox(
                        label="Question",
                        placeholder="e.g. Summarize and compare insights from documents and database.",
                        lines=2,
                    )
                    hyb_btn = gr.Button("Ask →", variant="primary")
                with gr.Column(scale=2):
                    hyb_answer = gr.Markdown(label="Answer")
                    hyb_chunks = gr.Markdown(label="Retrieved Chunks")
                    hyb_meta   = gr.Markdown(label="Timing & Metadata")

            hyb_btn.click(
                fn=run_hybrid,
                inputs=[hyb_files, hyb_conn, hyb_question, api_key_input],
                outputs=[hyb_answer, hyb_chunks, hyb_meta],
            )

    gr.Markdown(
        "---\n"
        "Built by [krisnasetyadi](https://github.com/krisnasetyadi) · "
        "[Source](https://github.com/krisnasetyadi/pdf-reader) · "
        "MIT License"
    )

if __name__ == "__main__":
    demo.launch()
