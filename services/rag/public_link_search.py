"""Realtime search over public links: downloading their items (Google Drive included) and searching them per request.

Mixin of services.rag.processor.PDFQAProcessor; the methods were moved
verbatim from that class by MS-657."""
import logging
import os
import re
import tempfile
import uuid
from config import config
from services.ingest.documents import extract_text_from_pptx
from services.ingest.ocr import PdfOcr
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

logger = logging.getLogger(__name__)


class PublicLinkSearchMixin:
    def _ocr_scanned_pages(self, pdf_path: str, pages: List[Document]) -> None:
        """Fill in pages that PyPDFLoader found no text on (scans) with OCR."""
        ocr = PdfOcr(pdf_path)
        try:
            for index, page in enumerate(pages):
                if ocr.needs_ocr(page.page_content):
                    page.page_content = ocr.read_page(index, page.page_content)
        finally:
            ocr.close()

    def _extract_google_drive_file_id(self, raw_url: str) -> Optional[str]:
        parsed = urlparse(raw_url)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host not in {"drive.google.com", "docs.google.com"}:
            return None

        match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", parsed.path)
        if match:
            return match.group(1)

        query = parse_qs(parsed.query)
        return query.get("id", [None])[0]

    def _normalize_public_link_download_url(self, raw_url: str) -> str:
        file_id = self._extract_google_drive_file_id(raw_url)
        if file_id:
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        return raw_url

    def _extract_remote_filename(self, response: Any, source_url: str, fallback_name: str) -> str:
        content_disposition = response.headers.get("content-disposition", "")
        filename_star = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition, re.IGNORECASE)
        if filename_star:
            return unquote(filename_star.group(1).strip())

        filename = re.search(r'filename="?([^";]+)"?', content_disposition, re.IGNORECASE)
        if filename:
            return filename.group(1).strip()

        parsed = urlparse(str(response.url or source_url))
        candidate = os.path.basename(parsed.path.rstrip("/"))
        return candidate or fallback_name

    def _download_public_link_item(self, item: Dict[str, Any], destination_dir: str) -> Optional[Tuple[str, str]]:
        import requests
        from services.security.ssrf_guard import assert_public_url_safe

        source_url = item.get("url", "").strip()
        if not source_url:
            return None

        download_url = self._normalize_public_link_download_url(source_url)
        fallback_name = item.get("name") or f"public-link-{uuid.uuid4().hex[:8]}"

        try:
            assert_public_url_safe(download_url)
        except Exception as exc:
            logger.warning("Rejected public link item %s: %s", source_url, exc)
            return None

        try:
            response = requests.get(
                download_url,
                stream=True,
                timeout=(20, 60),
                allow_redirects=True,
                headers={"User-Agent": "DocuLens/1.0 (+public-link-runtime)"},
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Failed downloading public link item %s: %s", source_url, exc)
            return None

        content_type = (response.headers.get("content-type") or "").lower()
        resolved_name = self._extract_remote_filename(response, source_url, fallback_name)
        suffix = Path(resolved_name).suffix.lower()

        if "pdf" in content_type or suffix == ".pdf":
            final_name = resolved_name if resolved_name.lower().endswith(".pdf") else f"{resolved_name}.pdf"
        elif "presentationml.presentation" in content_type or suffix == ".pptx":
            final_name = resolved_name if suffix == ".pptx" else f"{resolved_name}.pptx"
        elif content_type.startswith("text/") or suffix in {".txt", ".md", ".log"}:
            final_name = resolved_name if suffix in {".txt", ".md", ".log"} else f"{resolved_name}.txt"
        else:
            logger.info("Skipping unsupported public link item %s (content-type=%s)", source_url, content_type)
            return None

        safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(final_name).name).strip(" ._") or fallback_name
        destination_path = os.path.join(destination_dir, safe_name)

        try:
            with open(destination_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        handle.write(chunk)
        finally:
            response.close()

        return destination_path, safe_name

    def _load_public_link_documents(self, public_link_sources: List[Dict[str, Any]]) -> List[Document]:
        documents: List[Document] = []
        seen_urls = set()

        with tempfile.TemporaryDirectory(prefix="public-link-runtime-") as temp_dir:
            for source in public_link_sources:
                link_id = source.get("link_id", "")
                link_title = source.get("title") or link_id or "Public Link"
                items = source.get("items") or []
                if not items and source.get("url"):
                    items = [{"url": source["url"], "name": link_title, "item_type": "file"}]

                for item in items:
                    item_url = (item.get("url") or "").strip()
                    if not item_url or item_url in seen_urls or item.get("item_type") == "folder":
                        continue
                    seen_urls.add(item_url)

                    downloaded = self._download_public_link_item(item, temp_dir)
                    if not downloaded:
                        continue

                    file_path, file_name = downloaded
                    extension = Path(file_path).suffix.lower()

                    try:
                        if extension == ".pdf":
                            pdf_loader = PyPDFLoader(file_path)
                            loaded_docs = pdf_loader.load()
                            self._ocr_scanned_pages(file_path, loaded_docs)
                        elif extension == ".pptx":
                            loaded_docs = [
                                Document(page_content=page["text"])
                                for page in extract_text_from_pptx(file_path) or []
                            ]
                        elif extension in {".txt", ".md", ".log"}:
                            try:
                                text_loader = TextLoader(file_path, encoding="utf-8")
                                loaded_docs = text_loader.load()
                            except UnicodeDecodeError:
                                text_loader = TextLoader(file_path, encoding="latin-1")
                                loaded_docs = text_loader.load()
                        else:
                            continue
                    except Exception as exc:
                        logger.warning("Failed loading public link item %s: %s", item_url, exc)
                        continue

                    # Merge all pages of one file into a single Document before
                    # splitting. Slide-deck PDFs carry only a few words per page,
                    # so page-level chunks end up as bare headings with no body
                    # text for the LLM to work with.
                    merged_text = "\n\n".join(
                        doc.page_content for doc in loaded_docs if doc.page_content.strip()
                    )
                    if not merged_text.strip():
                        continue

                    merged_doc = Document(
                        page_content=merged_text,
                        metadata={
                            "source": file_name,
                            "file_path": file_path,
                            "source_kind": "public_link",
                            "public_link_id": link_id,
                            "public_link_title": link_title,
                            "public_link_url": source.get("url"),
                            "item_url": item_url,
                            "collection_id": f"public-link:{link_id}" if link_id else "public-link",
                        },
                    )
                    documents.append(merged_doc)

        return documents

    def search_public_links_realtime(
        self,
        query: str,
        public_link_sources: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
    ) -> List[Document]:
        if not public_link_sources:
            return []

        documents = self._load_public_link_documents(public_link_sources)
        if not documents:
            logger.info("No readable realtime documents found for public links")
            return []

        # Larger chunks than the global config: public-link sources are often
        # slide decks whose per-page text is tiny, so chunks must span several
        # pages to give the LLM usable context.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.public_link_chunk_size,
            chunk_overlap=config.public_link_chunk_overlap,
        )
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            return []

        vector_store = FAISS.from_documents(chunks, self.embeddings)
        # Wider FAISS candidate pool than top_k so RRF fusion with BM25 has
        # dense scores for (nearly) every lexical match it promotes.
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
            doc.metadata.setdefault("source_kind", "public_link")
            realtime_results.append(doc)

        return realtime_results[: config.total_k_results]
