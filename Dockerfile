# ===========================================
# Dockerfile for HuggingFace Spaces
# PDF Reader Backend - FastAPI + ML Models
# ===========================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Set working directory
WORKDIR /app

# Install system dependencies
# antiword: extracts text from legacy binary .doc uploads (services/ingest/documents.py
# extract_text_from_doc) — python-docx only understands the newer .docx format.
# tesseract-ocr (+ Indonesian, English and orientation data): reads scanned PDF
# pages with no embedded text (services/ingest/ocr.py). The osd data is what
# lets it straighten sideways or upside-down photos.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    antiword \
    tesseract-ocr \
    tesseract-ocr-ind \
    tesseract-ocr-eng \
    tesseract-ocr-osd \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
# CPU-only torch first: the Space has no GPU, and the default Linux wheel pulls
# several GB of CUDA libraries. requirements.txt's torch==2.7.0 is then already
# satisfied by 2.7.0+cpu, so pip does not fetch the CUDA build again.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Create directories for data and cache
RUN mkdir -p /app/data/uploads /app/data/indices /app/.cache/huggingface

# Copy application code
COPY . .

# Create non-root user for security (HuggingFace Spaces requirement)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application
# HuggingFace Spaces expects the app to run on port 7860
# --timeout-keep-alive: uvicorn's 5s default closes idle connections before
# the HF proxy stops reusing them, so a request can land on a just-closed
# connection and the proxy answers 502. Outlive the proxy's idle timeout.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "75"]
