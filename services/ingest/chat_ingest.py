# services/ingest/chat_ingest.py
"""
Shared pipeline for turning a list of ChatMessage into a searchable
ChatCollection: chunk -> extract keywords -> build FAISS index -> persist
(disk + optionally Supabase Storage/DB).

Used by both the WhatsApp file-upload flow (services/chat.py) and the
Telegram live-sync flow (services/telegram.py), so the two paths can't drift
apart — a Telegram-synced chat is searchable through the exact same code
path as an uploaded WhatsApp export.
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Optional

from services.storage import supabase as supabase_storage
from config import config
from models import ChatMessage, ChatCollection, ChatPlatform
from services.ingest.chat_parser import ChatParser
from services.ingest.progress import ProgressCallback

logger = logging.getLogger(__name__)


async def ingest_chat_messages(
    collection_id: str,
    messages: List[ChatMessage],
    file_name: str,
    platform: ChatPlatform,
    raw_file_path: Optional[str] = None,
    on_progress: ProgressCallback | None = None,
    owner_id: Optional[str] = None,
    size_bytes: int = 0,
) -> ChatCollection:
    """Chunk, embed, index, and register a set of parsed chat messages as a ChatCollection.

    `raw_file_path`, when given, is the original export file on disk (WhatsApp
    upload) and gets uploaded to Supabase Storage alongside the index. Telegram
    sync passes a generated text transcript so its messages can be previewed.
    """
    if not messages:
        raise ValueError("No messages to ingest")

    parser = ChatParser()
    participants = sorted({m.sender for m in messages})
    timestamps = [m.timestamp for m in messages if m.timestamp]
    date_range = {
        "start": min(timestamps).isoformat() if timestamps else None,
        "end": max(timestamps).isoformat() if timestamps else None,
    }

    if on_progress:
        on_progress("preparing", 30)
    keywords = parser.extract_keywords(messages, top_n=30)
    chunks = parser.chunk_messages_by_conversation(
        messages,
        chunk_size=config.chat_chunk_size,
        overlap=config.chat_chunk_overlap,
    )

    index_dir = os.path.join(config.chat_index_folder, collection_id)
    os.makedirs(index_dir, exist_ok=True)
    await _build_vector_store(collection_id, chunks, file_name, platform.value, on_progress)
    if on_progress:
        on_progress("saving", 88)

    collection = ChatCollection(
        collection_id=collection_id,
        platform=platform,
        file_name=file_name,
        message_count=len(messages),
        date_range=date_range,
        participants=participants,
        created_at=datetime.now(),
    )
    collection_dict = collection.model_dump(mode="json")
    collection_dict["keywords"] = keywords
    # S3 must receive metadata from this build, not the previous sync.
    _save_collection_metadata(collection_id, collection_dict)

    storage_paths: List[str] = []
    if supabase_storage.is_enabled():
        try:
            if raw_file_path and os.path.isfile(raw_file_path):
                supabase_storage.upload_chat_file(collection_id, raw_file_path, file_name)
                storage_paths = [f"chat-uploads/{collection_id}/{file_name}"]
            supabase_storage.upload_chat_index(collection_id, index_dir)
            logger.info("Chat collection uploaded to Supabase Storage: %s", collection_id)
        except Exception as exc:
            logger.warning("Supabase Storage upload failed (disk fallback): %s", exc)

    # The chat collection list reads the database whenever there is one, so
    # the row is written even without Supabase Storage (it used to be
    # skipped then, and the collection never showed up).
    if supabase_storage.has_database():
        try:
            supabase_storage.register_chat_collection(
                collection_id=collection_id,
                file_name=file_name,
                platform=platform.value,
                message_count=len(messages),
                participants=participants,
                date_range=date_range,
                keywords=keywords,
                storage_paths=storage_paths,
                owner_id=owner_id,
                size_bytes=size_bytes,
            )
            logger.info("Chat collection registered in the database: %s", collection_id)
        except Exception as exc:
            logger.warning("Chat collection registration failed: %s", exc)
    if on_progress:
        on_progress("saving", 97)

    return collection


async def _build_vector_store(collection_id: str, chunks: List[dict], file_name: str, platform: str,
                              on_progress: ProgressCallback | None = None) -> None:
    from langchain_core.documents import Document
    from services.rag.processor import processor
    from services.ingest.documents import build_source_index

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "source": file_name,
                "collection_id": collection_id,
                "data_type": "chat",
                "platform": platform,
                "chunk_index": i,
                "message_count": chunk["message_count"],
                "participants": ", ".join(chunk["participants"]),
                "time_range_start": chunk["time_range"]["start"] or "",
                "time_range_end": chunk["time_range"]["end"] or "",
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    if not processor.embeddings:
        processor.initialize_components()

    vector_store = build_source_index(documents, processor.embeddings, on_progress)
    index_path = os.path.join(config.chat_index_folder, collection_id)
    vector_store.save_local(index_path)
    logger.info("Chat vector store saved to: %s", index_path)


def _save_collection_metadata(collection_id: str, collection: dict) -> None:
    metadata_path = os.path.join(config.chat_index_folder, collection_id, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2, default=str)
