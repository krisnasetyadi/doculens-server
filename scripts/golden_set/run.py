"""Golden question set runner: retrieval hit rate without an LLM (MS-908).

    python -m scripts.golden_set.run                  # OCR off vs on, side by side
    python -m scripts.golden_set.run --ocr on         # one mode only
    python -m scripts.golden_set.run --report out.md  # also write a Markdown report

Runs fully offline: no Gemini, no database, no Supabase Storage. Every source
in fixtures.py is indexed with the app's own ingestion (process_pdfs and the
chat ingest), into temporary folders, and every question is searched with the
app's own retrieval (search_across_collections: FAISS + BM25, and the chat
search), using the same top_k as services/rag/hybrid.py.

Per question it reports:
  rank     where the expected file (and page) first appears in the results,
           "-" when it is not returned at all, "rejected" when the file could
           not be indexed (a fully scanned PDF without OCR)
  context  whether the expected answer text is in the returned chunks, i.e.
           whether the LLM would even be given the answer
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from collections import defaultdict
from unittest.mock import patch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("JWT_SECRET", "golden-set-not-for-production")

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_PATH = os.path.join(HERE, "questions.json")


def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def category(question):
    from scripts.golden_set.fixtures import SCANNED_PAGES

    source = question["source"]
    if question["page"] in SCANNED_PAGES.get(source, ()):
        return "pdf scan"
    return {".pdf": "pdf digital", ".docx": "docx", ".xlsx": "xlsx", ".csv": "csv", ".txt": "chat"}[
        os.path.splitext(source)[1]
    ]


def first_rank(results, source, page):
    """1-based position of the first result from `source` (and `page`), or None."""
    for position, doc in enumerate(results, 1):
        if doc.metadata.get("source") != source:
            continue
        if page is None or str(doc.metadata.get("page")) == str(page):
            return position
    return None


def ocr_word_recall(truth, fixtures_dir):
    """Share of each scanned page's words that the extraction read back."""
    from scripts.golden_set.fixtures import SCANNED_PAGES
    from services.ingest.documents import extract_text_from_pdf

    recall = {}
    for source, pages in SCANNED_PAGES.items():
        read = {p["page"]: p["text"] for p in extract_text_from_pdf(os.path.join(fixtures_dir, source)) or []}
        for page in sorted(pages):
            expected = set(re.findall(r"[\w.,-]+", normalize(truth[source][page])))
            got = set(re.findall(r"[\w.,-]+", normalize(read.get(page, ""))))
            recall[(source, page)] = len(expected & got) / len(expected)
    return recall


def run_mode(ocr_on, questions, fixtures_dir, truth):
    from langchain_community.embeddings import HuggingFaceEmbeddings

    from config import config
    from models import ChatPlatform
    from services.ingest.chat_ingest import ingest_chat_messages
    from services.ingest.chat_parser import ChatParser
    from services.ingest.documents import process_pdfs
    from services.rag.processor import processor
    from services.storage import supabase as supabase_storage

    if processor.embeddings is None:
        # Same settings as processor.initialize_components (services/rag/llm.py),
        # which also loads the LLM and the database, so it is not called here.
        processor.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    mode = "ocr-on" if ocr_on else "ocr-off"
    with tempfile.TemporaryDirectory() as work, \
            patch.object(config, "index_folder", os.path.join(work, "indices")), \
            patch.object(config, "chat_index_folder", os.path.join(work, "chat_indices")), \
            patch.object(config, "ocr_enabled", ocr_on), \
            patch.object(supabase_storage, "is_enabled", return_value=False), \
            patch.object(supabase_storage, "has_database", return_value=False):
        doc_ids, chat_ids, rejected = {}, {}, set()
        for source in sorted({q["source"] for q in questions}):
            collection_id = f"golden-{mode}-{os.path.splitext(source)[0]}"
            path = os.path.join(fixtures_dir, source)
            if source.endswith(".txt"):
                messages, _ = ChatParser().parse_whatsapp(path)
                asyncio.run(ingest_chat_messages(collection_id, messages, source, ChatPlatform.WHATSAPP))
                chat_ids[source] = collection_id
            elif process_pdfs([path], collection_id) > 0:
                doc_ids[source] = collection_id
            else:
                rejected.add(source)

        rows = []
        for question in questions:
            if question["source"] in rejected:
                rows.append({"rank": "rejected", "context": False})
                continue
            if question["source"].endswith(".txt"):
                results = processor.search_across_chat_collections(
                    question["question"], collection_ids=list(chat_ids.values()),
                    file_filter=processor.extract_file_reference(question["question"]),
                    top_k=max(int(config.k_per_collection * 0.5), 6),
                )
            else:
                results = processor.search_across_collections(
                    question["question"], collection_ids=list(doc_ids.values()),
                    top_k=int(config.k_per_collection * 0.6),
                )
            context = normalize(" ".join(doc.page_content for doc in results))
            rows.append({
                "rank": first_rank(results, question["source"], question["page"]),
                "context": normalize(question["answer"]) in context,
            })
        recall = ocr_word_recall(truth, fixtures_dir) if ocr_on else None
    return rows, recall


def summarize(questions, rows):
    groups = defaultdict(list)
    for question, row in zip(questions, rows):
        groups[category(question)].append(row)
        groups["TOTAL"].append(row)
    summary = {}
    for name, members in groups.items():
        ranks = [r["rank"] if isinstance(r["rank"], int) else None for r in members]
        summary[name] = {
            "n": len(members),
            "hit@1": sum(1 for r in ranks if r == 1),
            "hit@3": sum(1 for r in ranks if r and r <= 3),
            "hit": sum(1 for r in ranks if r),
            "context": sum(1 for r in members if r["context"]),
        }
    return summary


def render(questions, results):
    """Markdown tables: per question, then per source type, for each mode."""
    modes = list(results)
    lines = ["## Per question", "",
             "| id | type | lang | " + " | ".join(f"rank {m} | context {m}" for m in modes) + " |",
             "|" + "---|" * (3 + 2 * len(modes))]
    for index, question in enumerate(questions):
        cells = []
        for mode in modes:
            row = results[mode]["rows"][index]
            cells += [str(row["rank"] or "-"), "yes" if row["context"] else "no"]
        lines.append(f"| {question['id']} | {category(question)} | {question['lang']} | " + " | ".join(cells) + " |")
    lines += ["", "## Per source type", "",
              "| type | n | " + " | ".join(f"hit@1 {m} | hit@3 {m} | context {m}" for m in modes) + " |",
              "|" + "---|" * (2 + 3 * len(modes))]
    order = ["pdf digital", "pdf scan", "docx", "xlsx", "csv", "chat", "TOTAL"]
    for name in [n for n in order if n in results[modes[0]]["summary"]]:
        cells = []
        for mode in modes:
            s = results[mode]["summary"][name]
            cells += [f"{s['hit@1']}/{s['n']}", f"{s['hit@3']}/{s['n']}", f"{s['context']}/{s['n']}"]
        lines.append(f"| {name} | {results[modes[0]]['summary'][name]['n']} | " + " | ".join(cells) + " |")
    recall = next((r["recall"] for r in results.values() if r["recall"]), None)
    if recall:
        lines += ["", "## OCR word recall per scanned page (ocr-on)", "", "| file | page | words read |", "|---|---|---|"]
        lines += [f"| {source} | {page} | {value:.0%} |" for (source, page), value in recall.items()]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ocr", choices=["on", "off", "both"], default="both")
    parser.add_argument("--fixtures", help="folder to write the generated sources to (default: a temp folder)")
    parser.add_argument("--report", help="also write the Markdown report to this file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

    from scripts.golden_set.fixtures import build_fixtures

    with open(QUESTIONS_PATH, encoding="utf-8") as handle:
        questions = json.load(handle)
    with tempfile.TemporaryDirectory() as temp:
        fixtures_dir = args.fixtures or temp
        truth = build_fixtures(fixtures_dir)
        modes = {"off": [False], "on": [True], "both": [False, True]}[args.ocr]
        results = {}
        for ocr_on in modes:
            rows, recall = run_mode(ocr_on, questions, fixtures_dir, truth)
            results["ocr-on" if ocr_on else "ocr-off"] = {
                "rows": rows, "summary": summarize(questions, rows), "recall": recall,
            }
    report = render(questions, results)
    print(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write("# Golden set results\n\n" + report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
