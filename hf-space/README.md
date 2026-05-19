---
title: Agnostic Multi-Source RAG
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: RAG pipeline — ask questions across PDFs, TXT files, and PostgreSQL databases
---

# Agnostic Multi-Source RAG

A source-agnostic Retrieval-Augmented Generation (RAG) system that answers questions from:

- 📄 **Documents** — PDF, TXT, MD, LOG (upload files)
- 🗄️ **PostgreSQL** — any table or custom SQL query
- 🔀 **Hybrid** — both sources merged into one FAISS index

Built with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual, 384-dim) + Gemini 2.5-flash.

## Usage

Set your `GEMINI_API_KEY` in the Space Secrets (Settings → Repository secrets).
