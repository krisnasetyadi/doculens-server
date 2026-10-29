# Golden question set

A fixed set of 34 questions, each with its expected answer and the file (and
page) it is on, used to measure retrieval before and after a change. Added
for MS-908 (OCR for scanned PDFs), with PowerPoint questions added for MS-910;
meant to be reused for later retrieval
work such as a reranker.

## Run it

From the `doculens-server` folder, with the project venv:

```bash
python -m scripts.golden_set.run                    # OCR off vs OCR on, side by side
python -m scripts.golden_set.run --ocr on           # one mode only
python -m scripts.golden_set.run --report out.md    # also write the tables to a file
```

It runs fully offline: no Gemini, no database, no Supabase Storage. OCR needs
the `tesseract` binary with `ind`, `eng` and `osd` data (`brew install
tesseract tesseract-lang` on a Mac; the Dockerfile installs it for the Space).
A run takes about a minute, most of it loading the embedding model.

## What it does

1. `fixtures.py` generates the sources, so the text of every page is known:
   digital PDFs, scanned PDFs (image-only pages with shadow, a 90 degree
   rotation, tilt plus stamp, noise, low contrast, a small photo on an empty
   page), a mixed PDF, DOCX, XLSX (two sheets), CSV, a WhatsApp export and a
   PowerPoint deck (a title slide, slide text, speaker notes, and a slide whose
   answer is only inside a picture), in Bahasa Indonesia and English.
2. Each source is indexed with the app's own ingestion (`process_pdfs`, the
   chat ingest) into temporary folders.
3. Each question in `questions.json` is searched with the app's own retrieval
   (`search_across_collections`: FAISS + BM25, and the chat search) with the
   same `top_k` as `services/rag/hybrid.py` and the same embedding settings as
   `processor.initialize_components`.

## What it reports

| Column | Meaning |
|---|---|
| rank | Position of the expected file (and page) in the results. `-` means not returned, `rejected` means the file could not be indexed at all |
| context | The expected answer text is inside the returned chunks, i.e. the LLM would be given the answer |
| hit@1, hit@3 | Questions whose expected source is ranked first, or in the top 3 |
| OCR word recall | Share of the words on each scanned page that extraction read back |

"OCR off" (`OCR_ENABLED=false`) behaves like the code before MS-908: digital
pages extract the same text, scanned pages are dropped.

## Results for MS-908

| type | n | hit@1 OCR off | hit@1 OCR on | context OCR off | context OCR on |
|---|---|---|---|---|---|
| pdf digital | 8 | 8/8 | 8/8 | 8/8 | 8/8 |
| pdf scan | 9 | 0/9 | 9/9 | 0/9 | 9/9 |
| docx | 5 | 5/5 | 5/5 | 5/5 | 5/5 |
| xlsx | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| csv | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| chat | 4 | 4/4 | 4/4 | 4/4 | 4/4 |
| total | 31 | 22/31 | 31/31 | 22/31 | 31/31 |

Scanned-PDF questions went from unanswerable (the file is rejected, or the
scanned page is missing from a mixed PDF) to ranked first; no other question
changed. OCR read 100% of the words on all 7 scanned pages.

## Results for MS-910

Added 3 questions on `pelatihan-k3-gudang.pptx`. The other 31 rows did not change.

| type | n | hit@1 OCR off | hit@1 OCR on | context OCR off | context OCR on |
|---|---|---|---|---|---|
| pptx (slide text, speaker notes) | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| pptx picture | 1 | 0/1 | 1/1 | 0/1 | 1/1 |
| total | 34 | 24/34 | 34/34 | 24/34 | 34/34 |

The picture question is only answerable once slide pictures are OCR'd; OCR
read 100% of the words on that slide.

## Limits

- The set is small and every source is synthetic, so retrieval is easy:
  questions reuse words from their page and there are only 11 sources. It
  shows whether text reaches the index, not how good ranking is on a large
  workspace.
- It checks retrieval only. Whether the LLM then answers correctly is not
  measured here.
- Each source type is searched on its own; the cross-source weighting and
  merge in `hybrid_search` is not exercised.

## Adding a question

Put the fact on a page in `fixtures.py`, then add an entry to
`questions.json` with `source`, `page` (`null` for chat) and an `answer`
string that appears on that page. `tests/test_golden_set.py` fails if the
answer is not on the page you named.
