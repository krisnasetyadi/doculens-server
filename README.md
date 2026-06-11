---
title: DocuLens API
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🔍 DocuLens API

AI-powered document Q&A system with hybrid search across PDF, Database, and Chat logs.

## Features
- 📄 PDF Document Analysis
- 🗄️ PostgreSQL Database Query
- 💬 Chat Log Search
- 🔀 Hybrid Search (combine all sources)
- 🤖 Multiple LLM Support (HuggingFace, Ollama, Gemini)

## API Endpoints
- GET /health - Health check
- POST /api/v1/query/hybrid - Hybrid search query
- POST /api/v1/upload - Upload PDF files
- GET /api/v1/collections - List PDF collections
- GET /api/v1/models/available - List available LLM models

## Tech Stack
- FastAPI
- LangChain
- FAISS Vector Store
- PostgreSQL (Neon)
- HuggingFace Transformers
