"""Public links: downloading their items (Google Drive included), building one
saved index per link, and searching those indexes.

Until MS-907 every question downloaded and embedded every file of every active
link and threw the index away. Now a link is indexed once, in the background
(services/public_links.py runs the job), its index is kept in Supabase Storage
under public-links/{link_id}/{version}/, and a question only loads it, from
memory or Storage, like a collection's.

Mixin of services.rag.processor.PDFQAProcessor (moved out of it by MS-657)."""
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from config import config
from services.ingest.documents import extract_text_from_pptx
from services.ingest.ocr import PdfOcr
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

logger = logging.getLogger(__name__)

SKIP_DOWNLOAD = "Couldn't be downloaded, or isn't a PDF, PowerPoint or text file"
SKIP_UNREADABLE = "Couldn't be read"
SKIP_NO_TEXT = "No readable text"


def public_link_cache_key(link_id: str) -> str:
    """Key of a link's index in the processor's index cache (beside PDF and "chat_" keys)."""
    return f"public_link_{link_id}"


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

    def _load_public_link_documents(
        self,
        public_link_sources: List[Dict[str, Any]],
        skipped: Optional[List[Dict[str, str]]] = None,
        on_item: Optional[Callable[[int, int], None]] = None,
    ) -> List[Document]:
        """One Document per readable file. A file that can't be used is left out
        and, when `skipped` is given, added to it with the reason, so one bad
        file never fails the whole link. `on_item(done, total)` follows the
        downloads."""
        documents: List[Document] = []
        seen_urls = set()
        skipped = skipped if skipped is not None else []
        total = sum(
            1 for source in public_link_sources
            for item in (source.get("items") or [])
            if item.get("item_type") != "folder"
        ) or 1
        done = 0

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
                    item_name = item.get("name") or item_url

                    downloaded = self._download_public_link_item(item, temp_dir)
                    done += 1
                    if on_item:
                        on_item(done, total)
                    if not downloaded:
                        skipped.append({"name": item_name, "url": item_url, "reason": SKIP_DOWNLOAD})
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
                        skipped.append({"name": item_name, "url": item_url, "reason": SKIP_UNREADABLE})
                        continue

                    # Merge all pages of one file into a single Document before
                    # splitting. Slide-deck PDFs carry only a few words per page,
                    # so page-level chunks end up as bare headings with no body
                    # text for the LLM to work with.
                    merged_text = "\n\n".join(
                        doc.page_content for doc in loaded_docs if doc.page_content.strip()
                    )
                    if not merged_text.strip():
                        skipped.append({"name": item_name, "url": item_url, "reason": SKIP_NO_TEXT})
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

    def build_public_link_index(
        self,
        source: Dict[str, Any],
        on_progress: Optional[Callable[[str, int], None]] = None,
    ) -> Tuple[Optional[FAISS], List[Dict[str, str]]]:
        """Download, read, split and embed one link's files into a FAISS index.
        Returns (index or None when no file was readable, skipped files). Same
        chunking as before MS-907, so answers draw on the same chunks."""
        report = on_progress or (lambda stage, progress: None)
        skipped: List[Dict[str, str]] = []
        # Downloads are most of the time; they take the bar from 5% to 80%.
        documents = self._load_public_link_documents(
            [source],
            skipped=skipped,
            on_item=lambda done, total: report("downloading", 5 + int(75 * done / total)),
        )
        if not documents:
            return None, skipped

        report("embedding", 85)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.public_link_chunk_size,
            chunk_overlap=config.public_link_chunk_overlap,
        )
        chunks = text_splitter.split_documents(documents)
        for chunk in chunks:
            # Points into the download folder, which is gone once the build ends.
            chunk.metadata.pop("file_path", None)
        if not chunks:
            return None, skipped
        return FAISS.from_documents(chunks, self.embeddings), skipped

    def _public_link_index_path(self, link_id: str, version: str) -> str:
        return os.path.join(config.public_link_index_folder, link_id, version)

    def _cache_public_link_index(self, link_id: str, version: str, vector_store: FAISS) -> None:
        key = public_link_cache_key(link_id)
        with self._cache_lock:
            self.vector_store_cache[key] = vector_store
            self.bm25_cache.pop(key, None)
            self._public_link_versions[link_id] = version

    def prune_public_link_versions(self, link_id: str, keep_version: str) -> None:
        """Deletes the link's other versions from local disk. Called by the build
        once the database points at `keep_version`; not when a version is merely
        loaded, since a question may still be loading the one a rebuild replaces."""
        link_dir = os.path.join(config.public_link_index_folder, link_id)
        try:
            for entry in os.listdir(link_dir):
                if entry != keep_version:
                    shutil.rmtree(os.path.join(link_dir, entry), ignore_errors=True)
        except OSError:
            pass

    def save_public_link_index(self, link_id: str, version: str, vector_store: FAISS) -> Optional[str]:
        """Write a freshly built index to local disk and keep it loaded; returns its folder."""
        index_path = self._public_link_index_path(link_id, version)
        try:
            vector_store.save_local(index_path)
        except Exception as exc:
            logger.error("Saving public link index %s/%s failed: %s", link_id, version, exc)
            return None
        self._cache_public_link_index(link_id, version, vector_store)
        return index_path

    def get_public_link_vector_store(self, link_id: str, version: str) -> Optional[FAISS]:
        """The link's index at `version` (the one the database says is live),
        from memory, local disk or Supabase Storage. A cached older version is
        replaced, so a rebuild made by another server instance is picked up."""
        key = public_link_cache_key(link_id)
        # Backoff entries are per version: a new build is tried at once.
        missing_key = f"{key}@{version}"
        with self._cache_lock:
            if self._public_link_versions.get(link_id) == version and key in self.vector_store_cache:
                return self.vector_store_cache[key]
            retry_at, _ = self._missing_vector_stores.get(missing_key, (0.0, 0))
            if time.monotonic() < retry_at:
                return None

        vector_store = self._load_public_link_index(link_id, version)
        with self._cache_lock:
            if vector_store is not None:
                self._missing_vector_stores.pop(missing_key, None)
            elif self.embeddings is not None:
                self._mark_missing(missing_key)
        return vector_store

    def _load_public_link_index(self, link_id: str, version: str) -> Optional[FAISS]:
        index_path = self._public_link_index_path(link_id, version)
        files = [os.path.join(index_path, name) for name in ("index.faiss", "index.pkl")]
        if not all(os.path.exists(f) for f in files):
            with self._download_lock(("public_link", link_id)):
                if not all(os.path.exists(f) for f in files):
                    from services.storage import supabase as supabase_storage
                    if not supabase_storage.download_public_link_index(link_id, version, index_path):
                        logger.warning("Public link index %s/%s could not be downloaded", link_id, version)
                        return None
        if self.embeddings is None:
            logger.error("Embeddings model not loaded")
            return None
        try:
            vector_store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        except Exception as exc:
            logger.error("Loading public link index %s/%s failed: %s", link_id, version, exc)
            return None
        self._cache_public_link_index(link_id, version, vector_store)
        return vector_store

    def drop_public_link_index(self, link_id: str) -> None:
        """Forget a deleted link's index: memory, its BM25 index and local disk."""
        key = public_link_cache_key(link_id)
        with self._cache_lock:
            self.vector_store_cache.pop(key, None)
            self.bm25_cache.pop(key, None)
            self._public_link_versions.pop(link_id, None)
        shutil.rmtree(os.path.join(config.public_link_index_folder, link_id), ignore_errors=True)

    def search_public_link_indexes(
        self,
        query: str,
        public_link_sources: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
    ) -> List[Document]:
        """Search the saved index of each link (sources carry `index_version`).
        Nothing is downloaded from the link itself. Each link is ranked as
        before MS-907, BM25 + FAISS fused with RRF, and the links' results are
        merged by score."""
        sources = [s for s in (public_link_sources or []) if s.get("index_version")]
        if not sources:
            return []

        query_vector = self.embeddings.embed_query(query)
        merged: Dict[int, Tuple[Document, float]] = {}
        for source in sources:
            link_id = source["link_id"]
            vector_store = self.get_public_link_vector_store(link_id, source["index_version"])
            if vector_store is None:
                continue
            try:
                # Wider FAISS candidate pool than top_k so RRF fusion with BM25 has
                # dense scores for (nearly) every lexical match it promotes.
                k_faiss = min(int(vector_store.index.ntotal), max(top_k * 3, 10))
                results_with_score = vector_store.similarity_search_with_score_by_vector(query_vector, k=k_faiss)
                faiss_ranked = [(doc, 1.0 / (1.0 + float(distance))) for doc, distance in results_with_score]
                bm25, corpus_docs = self.get_bm25_index(public_link_cache_key(link_id))
                fused = self._rrf_fuse(query, faiss_ranked, bm25, corpus_docs, top_k=max(top_k, 1))
            except Exception as exc:
                logger.error("Public link search failed for %s: %s", link_id, exc)
                continue
            for doc, similarity_score in fused:
                if similarity_score <= 0.05:
                    continue
                key = hash(doc.page_content[:200])
                if key not in merged or similarity_score > merged[key][1]:
                    merged[key] = (doc, similarity_score)

        results: List[Document] = []
        for doc, similarity_score in sorted(merged.values(), key=lambda pair: pair[1], reverse=True):
            doc.metadata["similarity_score"] = similarity_score
            doc.metadata.setdefault("source_kind", "public_link")
            results.append(doc)
        return results[: config.total_k_results]
