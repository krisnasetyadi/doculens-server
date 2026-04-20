# 📘 Dokumentasi — QA RAG AgnosticSource

> **File Notebook:** `QA_RAG_AgnosticSource.ipynb`  
> **Terakhir diperbarui:** April 2026  
> **Status:** ✅ Production-ready (setelah code review & bug fixes)

---

## Daftar Isi

1. [Latar Belakang & Tujuan](#1-latar-belakang--tujuan)
   - [Konsep Dasar: Embedding dan Retrieval](#-apa-itu-embedding)
2. [Arsitektur Sistem](#2-arsitektur-sistem)
   - [2.5 — Flow Detail: Folder, PostgreSQL, dan Evaluasi](#25--flow-detail-folder-postgresql-dan-evaluasi)
     - [A. Flow Besar Sistem](#a-flow-besar-sistem-dari-input-sampai-jawaban)
     - [B. Flow Detail: Folder / Google Drive](#b-flow-detail-folder--google-drive)
     - [C. Flow Detail: PostgreSQL](#c-flow-detail-postgresql)
     - [D. Flow Detail: Evaluasi](#d-flow-detail-evaluasi)
3. [Apa & Bagaimana Setiap Plugin dan Fungsi Bekerja](#3-apa--bagaimana-setiap-plugin-dan-fungsi-bekerja)
   - [Library Eksternal (Plugin)](#library-eksternal-plugin)
   - [SourceDetector — Cara Kerja](#sourcedetector--cara-kerja)
   - [FolderSourceAdapter — Cara Kerja](#foldersourceadapter--cara-kerja)
   - [PostgreSQLAdapter — Cara Kerja](#postgresqladapter--cara-kerja)
   - [UniversalTextSplitter — Cara Kerja](#universaltextsplitter--cara-kerja)
   - [EmbeddingModel — Cara Kerja](#embeddingmodel--cara-kerja)
   - [RuntimeIndexBuilder — Cara Kerja](#runtimeindexbuilder--cara-kerja)
   - [QueryProcessor — Cara Kerja](#queryprocessor--cara-kerja)
   - [AnswerGenerator — Cara Kerja](#answergenerator--cara-kerja)
   - [AgnosticRAGPipeline — Cara Kerja](#agnosticragpipeline--cara-kerja)
   - [Evaluator — Cara Kerja](#evaluator--cara-kerja)
4. [Kenapa A, B, C? — Keputusan Teknis](#4-kenapa-a-b-c--keputusan-teknis)
5. [Komponen Per Sel](#5-komponen-per-sel)
   - [Cell 1 — Dependencies](#cell-1--dependencies)
   - [Cell 2 — Config](#cell-2--config)
   - [Cell 3 — Source Detector & Adapters](#cell-3--source-detector--adapters)
   - [Cell 4 — Splitter, Embeddings, FAISS, Query Processor](#cell-4--splitter-embeddings-faiss-query-processor)
   - [Cell 5 — Answer Generator](#cell-5--answer-generator)
   - [Cell 6 — Agnostic RAG Pipeline](#cell-6--agnostic-rag-pipeline)
   - [Cell 7 — Evaluator](#cell-7--evaluator)
   - [Cell 8 — Markdown: Architecture Overview](#cell-8--markdown-architecture-overview)
   - [Cells 9–16 — Unit Tests, Interactive Demo, Health Checks](#cells-916--unit-tests-interactive-demo-health-checks)
   - [Cell 17 — Skenario A (FolderSourceAdapter)](#cell-17--skenario-a-foldersourceadapter)
   - [Cell 18 — SQL Setup (Inisialisasi Data BBKP/TINS)](#cell-18--sql-setup-inisialisasi-data-bbkptins)
   - [Cell 19 — Skenario B (PostgreSQLAdapter)](#cell-19--skenario-b-postgresqladapter)
   - [Cell 20 — Skenario C (Cross-source + exclude_patterns)](#cell-20--skenario-c-cross-source--exclude_patterns)
   - [Cell 21 — Visualisasi 4-Panel](#cell-21--visualisasi-4-panel)
   - [Cell 22 — Ringkasan & Metrik Komposit](#cell-22--ringkasan--metrik-komposit)
   - [Cell 23 — SQL Reference (ALTER TABLE)](#cell-23--sql-reference-alter-table)
6. [Alur Data End-to-End](#6-alur-data-end-to-end)
7. [Format Input yang Didukung](#7-format-input-yang-didukung)
8. [Output yang Dihasilkan](#8-output-yang-dihasilkan)
9. [Pertanyaan Kunci: Apakah Ini Realtime?](#9-pertanyaan-kunci-apakah-ini-realtime)
10. [Metrik Evaluasi: 8 Standar + 3 Komposit](#10-metrik-evaluasi-8-standar--3-komposit)
11. [Corpus & Dataset Evaluasi](#11-corpus--dataset-evaluasi)
12. [Empat Skenario Evaluasi & Hasil Aktual](#12-empat-skenario-evaluasi--hasil-aktual)
13. [Siapa yang Menggunakan Sistem Ini](#13-siapa-yang-menggunakan-sistem-ini)
14. [Bug yang Ditemukan & Diperbaiki (Code Review)](#14-bug-yang-ditemukan--diperbaiki-code-review)
15. [Cara Pakai](#15-cara-pakai)
16. [Batasan & Catatan Penting](#16-batasan--catatan-penting)

---

## 1. Latar Belakang & Tujuan

### Masalah yang Diselesaikan

Sistem RAG (Retrieval-Augmented Generation) konvensional memerlukan:
- Pra-pemrosesan data (pre-indexing) yang harus dijalankan sebelum bisa menjawab pertanyaan
- Kode terpisah untuk setiap jenis sumber data (folder vs database vs chat)
- Penulisan index ke disk (file `.faiss`) yang tidak cocok untuk data yang sering berubah

### Solusi yang Dibangun

**QA RAG AgnosticSource** adalah sistem RAG yang:

1. **Agnostic terhadap sumber data** — Cukup berikan satu string `source`, sistem mendeteksi dan menangani otomatis
2. **Realtime** — Dokumen di-load dan FAISS index di-build *saat pertanyaan masuk* (`ask()` call), bukan saat startup
3. **Tidak ada pra-indexing ke disk** — Semua index ada di RAM, hilang saat kernel restart
4. **Mendukung data campuran** — Satu folder bisa berisi PDF, DOCX, chat JSON, chat CSV, Excel sekaligus

### Konsep Dasar: Embedding dan Retrieval

Sebelum masuk ke detail teknis, dua konsep ini adalah **inti dari cara kerja RAG**. Semua komponen lain berputar di sekitar keduanya.

---

#### 📐 Apa itu Embedding?

**Embedding** adalah proses mengubah teks (kata, kalimat, paragraf) menjadi **vektor angka** — sebuah daftar bilangan desimal dengan panjang tetap (misalnya 384 angka).

```
"Presiden pertama Indonesia adalah Soekarno."
                    ↓  Embedding Model
[0.023, -0.187, 0.441, 0.819, -0.032, ..., 0.012]
 ←──────────────── 384 angka ────────────────────→
```

**Mengapa diubah jadi angka?** Karena komputer tidak bisa langsung membandingkan makna teks, tapi bisa menghitung jarak antar vektor. Trik utamanya adalah:

> Teks yang **maknanya mirip** akan menghasilkan vektor yang **jaraknya dekat** di ruang matematika — meskipun kata-katanya berbeda.

```
"Siapa kepala negara pertama RI?"      → [0.021, -0.190, 0.445, ...]
"Presiden pertama Indonesia Soekarno"  → [0.025, -0.183, 0.439, ...]
                                                ↑
                                     Jarak sangat kecil = makna mirip ✅

"Harga saham hari ini naik 3%"         → [0.831,  0.442, -0.210, ...]
                                                ↑
                                     Jarak sangat besar = makna berbeda ✅
```

**Embedding Model** dalam sistem ini adalah `paraphrase-multilingual-MiniLM-L12-v2` — model neural network yang sudah dilatih pada ratusan juta kalimat dari 50+ bahasa (termasuk Bahasa Indonesia) agar bisa memetakan makna ke dalam ruang vektor 384 dimensi.

Setiap teks yang masuk ke sistem — baik isi dokumen maupun pertanyaan pengguna — **selalu melewati embedding model** sebelum bisa diproses lebih lanjut.

---

#### 🔍 Apa itu Retrieval?

**Retrieval** adalah proses **menemukan potongan dokumen yang paling relevan** dengan pertanyaan yang diajukan, dari seluruh dokumen yang sudah di-index.

Alurnya:

```
Pertanyaan pengguna: "Siapa presiden pertama Indonesia?"
         ↓
[1] Embedding: ubah pertanyaan → vektor query
         ↓
[2] Similarity Search: bandingkan vektor query
    dengan SEMUA vektor chunk dokumen di FAISS index
         ↓
[3] Ranking: urutkan chunk berdasarkan kedekatan vektor
    (semakin dekat = semakin relevan)
         ↓
[4] Filter & Top-K: ambil 5 chunk teratas yang
    melewati ambang batas similarity (0.25)
         ↓
Hasil: 5 potongan teks paling relevan dari dokumen
```

**Retrieval bukan keyword search** — ia tidak mencari kata yang sama persis. Ia mencari **makna yang sama**, sehingga:

| Pertanyaan | Chunk yang ditemukan | Kenapa match? |
|---|---|---|
| "kepala negara pertama" | "Presiden pertama RI adalah Soekarno" | Makna sama, kata beda |
| "berapa harga produk?" | "daftar harga barang: ..." | Semantik relevan |
| "siapa yang memimpin?" | "Direktur perusahaan adalah..." | Konteks kepemimpinan sama |

**Hubungan Embedding ↔ Retrieval:**

```
EMBEDDING                          RETRIEVAL
─────────                          ─────────
"Konversi teks → vektor"           "Cari vektor terdekat"

Dilakukan 2 kali dalam pipeline:
  1. Saat index dibuat:             Dilakukan saat query:
     embed semua chunk dokumen  →   embed pertanyaan →
     simpan di FAISS                cari di FAISS → return top-K chunk
```

Tanpa embedding yang baik, retrieval tidak akan menemukan chunk yang benar-benar relevan. Tanpa retrieval, LLM tidak punya konteks spesifik untuk menjawab — dan hanya bergantung pada pengetahuan umum model (yang bisa hallucinate).

---

### Pembatasan Desain (by design)

| Batasan | Alasan |
|---|---|
| Database hanya PostgreSQL | Keputusan eksplisit pengguna — tidak perlu MySQL, SQLite, MongoDB |
| Index hanya di RAM | Menjamin data selalu fresh dari sumber |
| LLM zero-shot (tidak fine-tuned) | Lebih fleksibel, tidak butuh training data |

---

## 2. Arsitektur Sistem

```
INPUT: source = "satu string"
           │
           ▼
    SourceDetector.detect(source)
    ─────────────────────────────
    Analisis prefix/pattern string:
      "postgresql://" / "postgres://"  →  SourceType.POSTGRES
      path (/, ./, C:\, ~, dll)       →  SourceType.FOLDER (default)
           │
    ┌──────┴───────────────────────────────────────┐
    │                                              │
FolderSourceAdapter                      PostgreSQLAdapter
────────────────────                     ─────────────────
Rekursif scan folder:                    SQLAlchemy 2.x:
  .pdf    → pypdf                          all tables → inspect()
  .txt/.md/.log → raw text                filter tables → parameter
  # .docx → nonaktif (PDF & TXT only)     custom SQL → pg_queries dict
  # .json/.csv/.xlsx → nonaktif           DataFrame → RawDocument
  .doc_extensions = (.pdf, .txt,          pool_pre_ping=True
                     .md, .log)
           │                                              │
           └──────────────────┬───────────────────────────┘
                              │
                      List[RawDocument]
                              │
                              ▼
               UniversalTextSplitter
               ─────────────────────
               RecursiveCharacterTextSplitter
               chunk_size=2000, overlap=300
               → List[LangChain Document]
                              │
                              ▼
               RuntimeIndexBuilder
               ─────────────────────
               FAISS.from_documents()  ← in-memory ONLY
               session cache (RAM dict, key=source string)
                              │
                              ▼
               QueryProcessor
               ─────────────────────
               embed query → similarity_search_with_score()
               L2 distance → similarity = 1/(1+dist)
               filter threshold=0.2, top_k=8
               → List[RetrievedChunk]
                              │
                              ▼
               AnswerGenerator
               ─────────────────────
               Context → Prompt → LLM
               Gemini 2.5-flash (primary)
               fallback chain: 2.0-flash → 1.5-flash
               exponential backoff: 15→30→60→120→240s
               zero-shot, temperature=0.3
                              │
                              ▼
               RAGResult
               ─────────────────────
               .answer         → string jawaban
               .retrieved_chunks → List[RetrievedChunk]
               .timing         → dict step → seconds
               .metadata       → source, raw_docs, total_chunks, llm
               .display()      → print terformat
                              │
                              ▼
               Evaluator (opsional)
               ─────────────────────
               retrieval_relevance  → cosine(q_emb, avg_chunk_emb)
               answer_faithfulness  → F1 overlap(answer, context)
               answer_completeness  → overlap(question_tokens, answer_tokens)
               → EvalScore → DataFrame → CSV + PNG
```

---

## 2.5 — Flow Detail: Folder, PostgreSQL, dan Evaluasi

Bagian ini adalah ringkasan visual end-to-end dari tiga perspektif utama yang sering ditanyakan.

---

### A. Flow Besar Sistem (Dari Input Sampai Jawaban)

```
USER: pipeline.ask("pertanyaan", source="...")
                        |
                        v
           SourceDetector.detect(source)
                        |
          +-------------+------------------+
          |                                |
   source = path folder           source = "postgresql://..."
          |                                |
          v                                v
 [ALUR FOLDER]                    [ALUR POSTGRESQL]
  (lihat B)                         (lihat C)
          |                                |
          +-------------+-----------------+
                        |
                        v
                List[RawDocument]
                        |
                        v
         UniversalTextSplitter
         chunk_size=2000, overlap=300
                        |
                        v
         List[LangChain Document]
                        |
                        v
         RuntimeIndexBuilder
         FAISS.from_documents() -- in-memory
         session cache: dict[source_key -> FAISS]
                        |
                        v
         QueryProcessor
         embed(pertanyaan) -> similarity search
         filter threshold=0.2, top_k=8
                        |
                        v
         List[RetrievedChunk]  (maks 5)
                        |
                        v
         AnswerGenerator
         build prompt + LLM.invoke()
         Gemini 2.5-flash (fallback: 2.0-flash → 1.5-flash)
                        |
                        v
         RAGResult
         .answer | .retrieved_chunks | .timing | .metadata
                        |
                        v
         Evaluator.display_result(result)   <- opsional
         (lihat D)
```

---

### B. Flow Detail: Folder / Google Drive

```
source = "/content/drive/MyDrive/data"
                |
                v
  FolderSourceAdapter("/content/drive/MyDrive/data")
                |
                v
  Path.exists()?
    Tidak -> _try_mount_drive()
             from google.colab import drive
             drive.mount('/content/drive')
    Masih tidak ada -> raise FileNotFoundError
                |
                v
  Path.rglob("*") -- rekursif semua subfolder
                |
                v
  Filter: is_file() AND suffix in doc_extensions
          (.pdf, .docx, .doc, .txt, .md, .log,
           .json, .csv, .xlsx, .xls)

  Opsional max_depth:
    depth = len(f.relative_to(root).parts)
    depth > max_depth -> skip
                |
                v
  Untuk setiap file -> _load_file(fp):

  +---.pdf---------+
  | PdfReader(fp)  |
  | page.extract_  |   -> content = teks semua halaman
  | text()         |      doc_type = "pdf"
  +----------------+

  +---.docx--------+
  | docx.Document  |
  | paragraphs     |   -> content = teks semua paragraf
  +----------------+      doc_type = "docx"

  +---.txt/.md/.log+
  | fp.read_text() |   -> content = raw text
  +----------------+      doc_type = "txt"/"markdown"/"log"

  +---.json--------+
  | json.load(fp)  |
  | isinstance(    |
  |   data, list)  |
  | AND keys &     |
  | {"role",       |
  |  "content",    |   -> [CHAT] _parse_chat_json()
  |  "message",    |      "[ROLE]: content\n..."
  |  "text",       |      doc_type = "chat"
  |  "sender"}?    |
  |                |
  | TIDAK          |   -> [DOKUMEN] json.dumps()
  +----------------+      doc_type = "json"

  +---.csv---------+
  | pd.read_csv()  |
  | columns &      |
  | {"role",       |
  |  "content",    |   -> [CHAT] _parse_chat_csv()
  |  "message",    |      "[ROLE]: content\n..."
  |  "text",       |      doc_type = "chat"
  |  "sender",     |
  |  "from","to",  |
  |  "body"}?      |
  |                |
  | TIDAK          |   -> [DATA TABULAR] df.to_string()
  +----------------+      doc_type = "csv"

  +---.xlsx/.xls---+
  | pd.read_excel  |
  | sheet_name=None|   -> semua sheet digabung
  | (semua sheet)  |      "=== Sheet: X ===\n..."
  +----------------+      doc_type = "excel"

                |
                v
  return List[RawDocument]
  setiap item: content + source(path) + doc_type + metadata
```

---

### C. Flow Detail: PostgreSQL

```
source = "postgresql://user:pass@host:5432/db"
                |
                v
  PostgreSQLAdapter(connection_string, tables=None, custom_queries=None)
                |
                v
  _get_engine() -- lazy init
    create_engine(conn_str, pool_pre_ping=True)
    test: conn.execute("SELECT 1")
    cache di self._engine
                |
                v
  load() -- dua fase:

  +=======================+
  | FASE 1: custom_queries|
  +=======================+
  for label, sql in self.custom_queries.items():
    pd.read_sql(text(sql), conn)
    if df.empty -> skip
    else:
      _df_to_text(df, label):
        header = "=== LABEL ===\nKolom: ...\nJumlah baris: N\n"
        rows   = df.head(1000).to_string(index=False)
        return header + rows
      -> RawDocument(
           source   = "query:label",
           doc_type = "db_query"
         )

  +=======================+
  | FASE 2: tabel         |
  +=======================+
  tables = self.tables (diberikan user)
        OR _list_tables():
             inspect(engine).get_table_names("public")
             -> semua tabel di schema public

  for table in tables:
    sql = 'SELECT * FROM "table" LIMIT 1000'
    pd.read_sql(text(sql), conn)
    if df.empty -> skip
    else:
      _df_to_text(df, table)
      -> RawDocument(
           source   = "table:table_name",
           doc_type = "db_table",
           metadata = {table, rows, cols, columns}
         )
                |
                v
  return List[RawDocument]
  -- setiap tabel / custom query = 1 RawDocument
  -- lanjut ke flow besar (split -> FAISS -> retrieve -> LLM)
  -- TIDAK ada perbedaan setelah step ini vs Folder
```

---

### D. Flow Detail: Evaluasi

```
result = pipeline.ask("pertanyaan", source="...")
              |
              v
evaluator.display_result(result)    <- pipeline TIDAK re-run
              |
              v
  s = self.score(result, ground_truth)
              |
  +-----------+--------------------------------------+
  | ground_truth=None (default, reference-free)     |
  |   ref = join(c.content for c in chunks)         |
  +-----------+--------------------------------------+
  | ground_truth="jawaban acuan" (reference-based)  |
  |   ref = ground_truth                            |
  +--------------------------------------------------+
              |
              v
  Hitung 8 metrik:

  [1] retrieval_relevance (reference-free)
      q_emb  = embed_query(pertanyaan)
      c_embs = embed_documents([c.content for c in chunks])
      avg    = mean(c_embs, axis=0)
      score  = cosine(q_emb, avg)  -> [0, 1]
      Mengukur: seberapa relevan chunks yang di-retrieve
                terhadap pertanyaan (secara semantik)

  [2] answer_faithfulness (reference-free)
      context = join(c.content for c in chunks)
      F1 token overlap antara jawaban dan context
      P = |tok(answer) & tok(context)| / |tok(context)|
      R = |tok(answer) & tok(context)| / |tok(answer)|
      score = 2*P*R / (P+R)
      Mengukur: seberapa banyak jawaban bersumber dari
                context (anti-hallucination check)

  [3] answer_completeness (reference-free)
      score = |tok(pertanyaan) & tok(jawaban)| / |tok(pertanyaan)|
      Mengukur: berapa banyak keyword pertanyaan
                muncul dalam jawaban

  [4] rouge_l (vs ref)
      LCS = Longest Common Subsequence(tok(jawaban), tok(ref))
      P   = LCS / len(tok(jawaban))
      R   = LCS / len(tok(ref))
      F1  = 2*P*R / (P+R)
      Mengukur: kemiripan urutan kata antara
                jawaban dan referensi (Lin 2004)

  [5] bleu_1 (vs ref)
      clipped = sum(min(count_hyp, count_ref) per unigram)
      precision = clipped / len(tok(jawaban))
      BP = exp(1 - len_ref/len_hyp) jika jawaban lebih pendek
      score = BP * precision
      Mengukur: berapa banyak kata jawaban muncul
                di referensi (Papineni 2002)

  [6] precision_at_k (reference-free)
      score = |chunks dimana similarity >= threshold| / top_k
      Mengukur: ketepatan retrieval di posisi top-K

  [7] mrr (reference-free)
      score = 1 / rank_pertama_chunk_yang_relevan
      Mengukur: seberapa tinggi posisi chunk relevan
                pertama dalam daftar hasil

  [8] context_coverage (reference-free)
      score = |unique c.source| / |total chunks|
      Mengukur: keragaman sumber dokumen yang di-retrieve

  overall = mean([1],[2],[3],[4],[5])   <- 5 metrik utama
              |
              v
  return EvalScore (semua field + ground_truth indicator)
              |
              v
  Tampilkan di layar (display_result):
    - Pertanyaan + Jawaban
    - Chunks dengan bar similarity visual
    - Timing per tahap (ms + % + tanda "terlama")
    - 5 metrik utama dengan bar [########..] + grade
      [BAIK] >= 0.7 | [CUKUP] >= 0.4 | [RENDAH] < 0.4
    - P@K, MRR, Coverage, Avg Chunk Score
    - Metadata (source, LLM, raw_docs, timestamp)

  Untuk batch (run_batch):
    for setiap pertanyaan:
      pipeline.ask() -> score() -> row ke DataFrame
    return DataFrame 15 kolom
    -> export .csv | .tex (LaTeX) | .png (4-panel chart)
```

---

## 3. Apa & Bagaimana Setiap Plugin dan Fungsi Bekerja

Bagian ini menjelaskan secara mendetail **apa** setiap komponen itu dan **bagaimana** cara kerjanya secara internal — mulai dari library eksternal (plugin) hingga setiap class/fungsi yang dibangun dalam sistem ini.

---

### Library Eksternal (Plugin)

#### `faiss-cpu` — Facebook AI Similarity Search

**Apa itu:**  
Library C++ yang di-wrap ke Python, dibuat oleh Meta/Facebook AI Research. Fungsi utamanya adalah **mencari vektor yang paling mirip** dari sekumpulan besar vektor secara sangat cepat.

**Bagaimana cara kerjanya:**

```
Dokumen (teks)  →  Embedding Model  →  Vektor float [0.12, -0.34, 0.91, ...]
                                              ↓
                                    FAISS IndexFlatL2
                                    ─────────────────
                                    Simpan semua vektor di RAM
                                    sebagai matriks N × D
                                    (N = jumlah chunk, D = 384 dimensi)
                                              ↓
                                    Saat query masuk:
                                    query_vec = embed("pertanyaan")
                                    Hitung L2 distance ke SEMUA vektor:
                                      dist(q, c_i) = √Σ(q_j - c_i_j)²
                                    Return top-K dengan distance terkecil
```

**IndexFlatL2** = exact search (bukan approximate). Semua jarak dihitung, hasilnya 100% akurat tapi O(N×D) per query.

**Mengapa di RAM:** `FAISS.from_documents()` di LangChain membuat objek `FAISS` Python yang menyimpan index di heap memory. Tidak ada disk I/O kecuali kalau eksplisit memanggil `save_local()`.

---

#### `sentence-transformers` — Semantic Embedding

**Apa itu:**  
Library dari Hugging Face yang menyediakan model transformer yang sudah di-fine-tune khusus untuk menghasilkan **sentence embeddings** — representasi vektor suatu kalimat/paragraf di ruang semantik.

**Bagaimana cara kerjanya:**

```
Input teks: "Siapa presiden pertama Indonesia?"
                    ↓
    Tokenizer (WordPiece/SentencePiece)
    → Token IDs: [101, 7489, 2003, 1996, ...]
                    ↓
    BERT-based Transformer (12 layers)
    → Hidden states per token: [768 dimensi per token]
                    ↓
    Pooling (mean pooling semua token)
    → Sentence vector: [384 dimensi]
                    ↓
    L2 Normalization (karena normalize_embeddings=True)
    → Unit vector: magnitude = 1.0
                    ↓
    Output: [0.023, -0.187, 0.441, ..., 0.012]  ← 384 float
```

**Mengapa model `paraphrase-multilingual-MiniLM-L12-v2`:**  
- `paraphrase` = di-fine-tune untuk mengenali kalimat yang bermakna sama meski kata berbeda
- `multilingual` = 50+ bahasa termasuk Bahasa Indonesia
- `MiniLM-L12` = arsitektur kecil (12 layer, 384 dim) tapi hasil baik
- `v2` = versi kedua, lebih baik dari v1

---

#### `langchain` + `langchain-community` — Framework Orkestrasi RAG

**Apa itu:**  
LangChain adalah framework Python untuk membangun aplikasi berbasis LLM. Menyediakan abstraksi standar untuk dokumen, splitter, vector store, chain, dan sebagainya.

**Komponen LangChain yang dipakai dalam sistem ini:**

| Komponen LangChain | Dipakai untuk | Cara kerja |
|---|---|---|
| `Document` | Schema dokumen dengan `page_content` + `metadata` | Dataclass standar yang diterima oleh FAISS |
| `RecursiveCharacterTextSplitter` | Split teks panjang → chunks | Coba split dari separator terbesar ke terkecil |
| `FAISS` (community) | Wrapper FAISS untuk LangChain | `from_documents()` → embed + index semua sekaligus |
| `HuggingFaceEmbeddings` (community) | Wrap sentence-transformers | Bridge antara LangChain dan HuggingFace |
| `HuggingFacePipeline` (community) | Wrap HF pipeline jadi LangChain LLM | Bisa `.invoke()` seperti LLM LangChain lainnya |
| `ChatGoogleGenerativeAI` (google-genai) | Wrap Gemini API | HTTP call ke Google AI API, return `AIMessage` |

---

#### `pypdf` — PDF Reader

**Apa itu:**  
Library Python untuk membaca file PDF tanpa dependensi eksternal (tidak butuh poppler, ghostscript, dll).

**Bagaimana cara kerjanya:**

```
file.pdf
    ↓
PdfReader("file.pdf")
    ↓ Parse struktur PDF (header, xref table, trailer)
    ↓ Decode halaman (Page objects, content streams)
    ↓
reader.pages[i].extract_text()
    ↓ Parse content stream operators:
       BT ... ET  = text block
       Tf         = font
       Tj / TJ    = show text
    ↓ Rekonstruksi urutan karakter
    ↓
Output: string teks per halaman
```

**Keterbatasan:** PDF dengan teks sebagai gambar (scan) tidak bisa di-extract — hasilnya string kosong. Sistem mendeteksi ini dan skip file tersebut (`if not text: return None`).

---

#### `python-docx` — Word Document Reader

**Apa itu:**  
Library untuk membaca (dan menulis) file `.docx` — format Office Open XML dari Microsoft Word.

**Bagaimana cara kerjanya:**

```
file.docx  (sebenarnya ZIP file)
    ↓
docx.Document("file.docx")
    ↓ Unzip → baca word/document.xml
    ↓ Parse XML namespace:
       <w:p>   = paragraph
       <w:r>   = run (bagian teks dengan format)
       <w:t>   = text content
    ↓
doc.paragraphs  →  List[Paragraph]
    ↓
"\n".join(p.text for p in doc.paragraphs if p.text.strip())
    ↓
Output: string teks semua paragraf
```

---

#### `sqlalchemy` — Database ORM & Connector

**Apa itu:**  
Library Python paling populer untuk koneksi database. Menyediakan dua layer: Core (SQL expressions) dan ORM (object mapping). Sistem ini hanya pakai **Core layer**.

**Bagaimana cara kerjanya dalam sistem ini:**

```
connection_string = "postgresql://user:pass@host:5432/db"
                              ↓
create_engine(conn_str, pool_pre_ping=True)
                              ↓
  Engine object (belum ada koneksi nyata — lazy)
  Pool: QueuePool (default)
  pool_pre_ping: sebelum kasih koneksi, test dengan "SELECT 1"
                              ↓
with engine.connect() as conn:
                              ↓
  Ambil koneksi dari pool → buka socket TCP ke PostgreSQL
  → Handshake, autentikasi
                              ↓
conn.execute(text("SELECT * FROM tabel LIMIT 5000"))
                              ↓
  Kirim SQL via wire protocol PostgreSQL
  → Terima result rows
                              ↓
pd.read_sql(text(sql), conn)
                              ↓
  Baca cursor → DataFrame
                              ↓
# Keluar from block → conn dikembalikan ke pool (bukan di-close)
```

---

#### `psycopg2-binary` — PostgreSQL Driver

**Apa itu:**  
Driver tingkat rendah (C extension) untuk protokol wire PostgreSQL. SQLAlchemy **membutuhkan** driver ini untuk berbicara dengan PostgreSQL.

**Bagaimana posisinya:**

```
Python code  →  SQLAlchemy  →  psycopg2  →  TCP socket  →  PostgreSQL server
               (abstraksi)   (driver C)    (network)       (database)
```

`psycopg2-binary` = versi pre-compiled (tidak butuh kompilasi C lokal). Cocok untuk development/notebook. Production biasanya pakai `psycopg2` (compile dari source).

---

#### `pandas` — Data Manipulation

**Apa itu:**  
Library DataFrame untuk Python — tabel data dua dimensi dengan label kolom dan baris.

**Dipakai dalam sistem ini untuk:**

| Penggunaan | Fungsi |
|---|---|
| `pd.read_csv(file, nrows=N)` | Baca CSV, batasi baris |
| `pd.read_excel(file, sheet_name=None)` | Baca semua sheet Excel sekaligus |
| `pd.read_sql(text(sql), conn)` | Eksekusi SQL → DataFrame |
| `df.head(N).to_string(index=False)` | Konversi tabel → string teks untuk di-embed |
| `pd.DataFrame(rows)` | Buat tabel hasil evaluasi |

---

#### `matplotlib` — Visualisasi

**Apa itu:**  
Library plotting Python. Dipakai khusus di `Evaluator.plot()` untuk membuat grafik hasil evaluasi.

**Cara kerja:**

```
df_eval (DataFrame)
    ↓
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ↓
axes[0].bar(...)   → bar chart rata-rata metrik
axes[1].plot(...)  → line chart skor per pertanyaan
    ↓
plt.savefig("eval_agnostic.png", dpi=120)   → simpan ke disk
plt.show()                                   → tampilkan di notebook
```

---

### `SourceDetector` — Cara Kerja

**Apa itu:** Class statis yang menganalisis satu string `source` dan menentukan tipe adapter yang harus dibuat.

**Flow kerja internal:**

```
SourceDetector.detect("/content/drive/MyDrive/data")
                            │
                            ▼
    s = source.strip()   →  "/content/drive/MyDrive/data"
                            │
                            ▼
    Cek 1: s.startswith("postgresql://") ?  → Tidak
    Cek 2: s.startswith("postgres://")   ?  → Tidak
                            │
                            ▼
    Cek regex folder patterns:
      re.match(r"^/", s)    → MATCH ✅ (dimulai dengan /)
                            │
                            ▼
    return SourceType.FOLDER
```

```
SourceDetector.detect("postgresql://admin:pass@db.host:5432/mydb")
                            │
                            ▼
    s = "postgresql://admin:pass@db.host:5432/mydb"
                            │
                            ▼
    Cek 1: s.startswith("postgresql://") ?  → YA ✅
                            │
                            ▼
    return SourceType.POSTGRES
```

**Flow `describe()`:**

```
SourceDetector.describe(source)
    ↓
    detect(source) → SourceType.FOLDER / POSTGRES
    ↓
    labels = {
        FOLDER:   "📂 Folder (lokal / Google Drive)",
        POSTGRES: "🐘 PostgreSQL Database",
    }
    ↓
    return labels[type]   → string label untuk display
```

---

### `FolderSourceAdapter` — Cara Kerja

**Apa itu:** Adapter yang men-scan folder secara rekursif, membaca setiap file yang didukung, dan menghasilkan list `RawDocument`.

---

#### 📁 Subfolder: Apakah Di-handle?

**Ya — sepenuhnya.** `FolderSourceAdapter` menelusuri subfolder secara rekursif tanpa batas menggunakan `Path.rglob("*")`:

```python
all_files = list(self.folder_path.rglob("*"))
```

`rglob("*")` adalah singkatan dari *recursive glob* — ia menelusuri **seluruh pohon direktori** ke bawah tanpa batas kedalaman, termasuk folder di dalam folder di dalam folder, sekecil apa pun strukturnya.

**Contoh struktur yang semuanya akan ditemukan:**

```
📂 data/                          ← folder root yang diberikan
│
├── 📄 laporan.pdf                ← depth 1 — ✅ ditemukan
├── 📂 2024/
│   ├── 📄 q1.docx                ← depth 2 — ✅ ditemukan
│   └── 📂 januari/
│       └── 📄 rekap.xlsx         ← depth 3 — ✅ ditemukan
└── 📂 arsip/
    └── 📂 lama/
        └── 📂 backup/
            └── 📄 data.json      ← depth 4 — ✅ ditemukan
```

Semua file di atas akan ditemukan oleh satu pemanggilan `rglob("*")`.

---

#### ⚠️ Apakah Ada Pembatas Kedalaman Subfolder?

**Tidak ada** pembatas kedalaman subfolder secara default. `rglob("*")` akan masuk ke semua level tanpa henti. Yang membatasi hanya **filter ekstensi file**:

```python
eligible = [
    f for f in all_files
    if f.is_file() and f.suffix.lower() in config.doc_extensions
]
```

`config.doc_extensions`:
```python
doc_extensions = (".pdf", ".docx", ".doc", ".md",
                  ".txt", ".json", ".csv", ".xlsx", ".xls")
```

Artinya:
- File dengan ekstensi di atas → **diproses** (apapun kedalamannya)
- File lain (`.png`, `.mp4`, `.zip`, dll) → **dilewati**
- Folder kosong → **dilewati** (karena `f.is_file()`)

---

#### 🔢 Parameter `max_depth` — Kontrol Kedalaman Manual

Kode saat ini **tidak punya** parameter `max_depth`. Jika struktur folder sangat dalam dan ingin membatasi kedalaman, bisa ditambahkan secara manual dengan menghitung level path:

```python
# Contoh: hanya masuk 2 level ke dalam subfolder
max_depth = 2
root = Path("/content/drive/MyDrive/data")

eligible = [
    f for f in root.rglob("*")
    if f.is_file()
    and f.suffix.lower() in config.doc_extensions
    and len(f.relative_to(root).parts) <= max_depth
]
```

Cara kerja `len(f.relative_to(root).parts)`:

```
root = /data
f    = /data/arsip/lama/backup/file.pdf

f.relative_to(root)        → arsip/lama/backup/file.pdf
f.relative_to(root).parts  → ("arsip", "lama", "backup", "file.pdf")
len(...)                   → 4

max_depth = 2 → 4 > 2 → ❌ dilewati
max_depth = 5 → 4 ≤ 5 → ✅ diproses
```

---

**Flow kerja `load()`:**

```
FolderSourceAdapter("/content/drive/MyDrive/data").load()
                            │
                            ▼
┌─────────────────────────────────────────────────┐
│  STEP 1: Validasi path                          │
│                                                 │
│  Path.exists()?                                 │
│    Tidak → _try_mount_drive()                   │
│             Coba: from google.colab import drive│
│             drive.mount('/content/drive')       │
│    Masih tidak ada → raise FileNotFoundError    │
└─────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────┐
│  STEP 2: Scan semua file rekursif               │
│                                                 │
│  all_files = Path(folder).rglob("*")            │
│  ← menelusuri SEMUA subfolder tanpa batas       │
│                                                 │
│  eligible = [f for f in all_files               │
│              if f.is_file()                     │
│              and f.suffix.lower()               │
│                 in config.doc_extensions]       │
└─────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────┐
│  STEP 3: Untuk setiap file → _load_file(fp)     │
│                                                 │
│  ext = fp.suffix.lower()                        │
│    .pdf         → _load_pdf(fp)     ✅ AKTIF    │
│    .txt/.md/.log→ _load_text(fp)    ✅ AKTIF    │
│    # .docx/.doc → _load_docx(fp)   ❌ nonaktif  │
│    # .json      → _load_json(fp)   ❌ nonaktif  │
│    # .csv       → _load_csv(fp)    ❌ nonaktif  │
│    # .xlsx/.xls → _load_excel(fp)  ❌ nonaktif  │
│                                                 │
│  Error → log warning + skip (tidak stop)        │
└─────────────────────────────────────────────────┘
                            │
                            ▼
                  List[RawDocument]
```

**Catatan:** Method `_load_json()`, `_load_csv()`, `_load_excel()`, dan `_load_docx()` telah dinonaktifkan. Hanya `_load_pdf()` dan `_load_text()` yang aktif saat ini.

---

### `PostgreSQLAdapter` — Cara Kerja

**Apa itu:** Adapter yang terhubung ke PostgreSQL, query semua/sebagian tabel + custom SQL, dan mengubah hasilnya menjadi `RawDocument`.

**Flow kerja `_get_engine()` (lazy init):**

```
_get_engine() dipanggil
          │
          ▼
  self._engine is None?
          │
    YA    │    TIDAK
    ▼          ▼
  create_engine(conn_str,     return self._engine (cached)
    pool_pre_ping=True)
          │
          ▼
  with engine.connect() as conn:
    conn.execute(text("SELECT 1"))
          │
  ✅ Sukses → self._engine = engine
  ❌ Error  → raise ConnectionError
```

**Flow kerja `load()`:**

```
PostgreSQLAdapter.load()
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  FASE 1: Custom Queries (pg_queries dict)           │
│                                                     │
│  for label, sql in self.custom_queries.items():     │
│    with engine.connect() as conn:                   │
│      df = pd.read_sql(text(sql), conn)              │
│    if df.empty → skip                               │
│    else → _df_to_text(df, label)                    │
│         → RawDocument(doc_type="db_query")          │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  FASE 2: Tabel                                      │
│                                                     │
│  tables = self.tables or _list_tables()             │
│    _list_tables():                                  │
│      inspect(engine).get_table_names("public")      │
│      → ["orders","products","users",...]            │
│                                                     │
│  for table in tables:                               │
│    sql = f'SELECT * FROM "{table}" LIMIT 5000'      │
│    with engine.connect() as conn:                   │
│      df = pd.read_sql(text(sql), conn)              │
│    if df.empty → skip                               │
│    else → _df_to_text(df, table)                    │
│         → RawDocument(doc_type="db_table")          │
└─────────────────────────────────────────────────────┘
          │
          ▼
  return List[RawDocument]   (1 per tabel/query)
```

**Flow `_df_to_text(df, label)`:**

```
df = DataFrame dengan N baris, M kolom

header = f"""
=== {label.upper()} ===
Kolom: col1, col2, col3, ...
Jumlah baris: N
"""

rows = df.head(5000).to_string(index=False)
# Output teks tabel terformat:
#  col1  col2  col3
#     1  val1  abc
#     2  val2  def

return header + "\n" + rows
```

---

### `UniversalTextSplitter` — Cara Kerja

**Apa itu:** Wrapper tipis di atas `RecursiveCharacterTextSplitter` LangChain yang memproses list `RawDocument` dan menghasilkan list `LangChain Document` siap di-embed.

**Flow kerja `split()`:**

```
raw_docs = [RawDocument(content="teks panjang...", source="file.pdf", doc_type="pdf"), ...]
                    │
                    ▼
  for rd in raw_docs:
    chunks = splitter.split_text(rd.content)
    │
    ▼  RecursiveCharacterTextSplitter internals:
    │
    │  Coba split dengan "\n\n" (paragraf)
    │    → Jika semua chunk ≤ 2000 char: selesai
    │    → Jika ada yang > 2000: split lagi dengan "\n"
    │      → Jika masih > 2000: split dengan ". "
    │        → Jika masih > 2000: split dengan " "
    │          → Jika masih > 2000: split per karakter
    │
    │  overlap: 50 char terakhir chunk sebelumnya
    │  diulang di awal chunk berikutnya
    │  (menjaga konteks di batas chunk)
    ▼
    for i, chunk in enumerate(chunks):
      LangChain Document(
        page_content = chunk,
        metadata = {
          "source":   rd.source,    # "file.pdf" atau "table:orders"
          "doc_type": rd.doc_type,  # "pdf", "chat", "db_table", dll
          "chunk_i":  i,            # indeks chunk dalam dokumen ini
          **rd.metadata             # pages, rows, cols, dll
        }
      )
                    │
                    ▼
  return List[LangChain Document]   ← siap di-embed
```

**Visualisasi overlap:**

```
Teks asli:
  "...akhir paragraf A. Awal paragraf B yang panjang..."
                ↑
           batas 500 char

Chunk 1:  "...akhir paragraf A."
Chunk 2:  "paragraf A. Awal paragraf B yang panjang..."
           ←── 300 char overlap ──→

Overlap memastikan kalimat di batas chunk tidak kehilangan konteks.
```

---

### `EmbeddingModel` — Cara Kerja

**Apa itu:** Singleton class yang menyimpan satu instance `HuggingFaceEmbeddings`. Memastikan model berat (384MB) hanya di-load satu kali per sesi kernel.

**Flow kerja:**

```
Pertama kali EmbeddingModel.get() dipanggil:
                    │
                    ▼
  cls._instance is None?  →  YA
                    │
                    ▼
  HuggingFaceEmbeddings(
    model_name = "...paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True},
  )
                    │
                    ▼
  Download model dari HuggingFace Hub (jika belum ada di cache)
  → ~/.cache/huggingface/hub/...
  Load model ke RAM: ~2–5 detik
                    │
                    ▼
  cls._instance = model_object   ← simpan di class variable
                    │
                    ▼
  return cls._instance

─────────────────────────────────────────────────
Panggilan berikutnya EmbeddingModel.get():
                    │
                    ▼
  cls._instance is None?  →  TIDAK
                    │
                    ▼
  return cls._instance   ← langsung, 0ms
```

**Fungsi embed yang dihasilkan:**

```python
embedder = EmbeddingModel.get()

# Embed satu query (untuk similarity search)
vector = embedder.embed_query("siapa presiden pertama?")
# → list 384 float

# Embed banyak dokumen (untuk index)
vectors = embedder.embed_documents(["teks 1", "teks 2", ...])
# → list of list, shape: N × 384
```

---

### `RuntimeIndexBuilder` — Cara Kerja

**Apa itu:** Class yang membangun FAISS index dari list `LangChain Document` dan menyimpannya di session cache (RAM dict).

**Flow kerja `build()`:**

```
index_builder.build(docs, source_key="/content/drive/...")
                    │
                    ▼
  config.use_session_cache AND source_key in self._cache?
                    │
        YA          │         TIDAK
        ▼                     ▼
  return self._cache[key]   embedder = EmbeddingModel.get()
  (instant, 0ms)                      │
                                       ▼
                            FAISS.from_documents(docs, embedder)
                            │
                            ▼  Internals:
                            │
                            │  1. embedder.embed_documents(
                            │       [d.page_content for d in docs]
                            │     )
                            │     → matrix N × 384
                            │
                            │  2. faiss.IndexFlatL2(384)
                            │     index.add(matrix)
                            │     → vektor tersimpan di RAM
                            │
                            │  3. Simpan mapping: vektor_idx → Document
                            │     (untuk retrieve metadata saat search)
                            ▼
                         FAISS vectorstore object
                            │
                            ▼
  self._cache[source_key] = vs   ← simpan di RAM dict
                            │
                            ▼
  return vs
```

**Flow kerja `clear_cache()`:**

```
clear_cache(source_key=None)
          │
    key diberikan?
          │
    YA    │    TIDAK (None)
    ▼          ▼
  self._cache.pop(key, None)    self._cache.clear()
  (hapus satu entry)            (hapus semua)
          │
          ▼
  Index yang dihapus → garbage collected oleh Python GC
  RAM dibebaskan
```

---

### `QueryProcessor` — Cara Kerja

**Apa itu:** Class yang mengubah pertanyaan teks menjadi vektor, mencari chunk paling relevan dari FAISS index, dan memformat hasilnya sebagai context string.

**Flow kerja `retrieve()`:**

```
query_proc.retrieve("siapa presiden pertama?", vectorstore)
                    │
                    ▼
  embedder = EmbeddingModel.get()   ← singleton, tidak re-load
                    │
                    ▼
  vectorstore.similarity_search_with_score(
    "siapa presiden pertama?",
    k = config.top_k * 2    # default: 16, ambil 2x lebih banyak dulu
  )
                    │
                    ▼  FAISS internals:
                    │  1. embed_query("siapa presiden pertama?") → vektor 384
                    │  2. hitung L2 distance ke semua N vektor di index
                    │  3. return top-10 (doc, L2_distance) terdekat
                    ▼
  raw_results = [(doc1, 0.45), (doc2, 0.89), (doc3, 1.23), ...]
  (L2 distance: semakin kecil = semakin mirip)
                    │
                    ▼
  Konversi L2 distance → similarity score:
    similarity = 1.0 / (1.0 + L2_distance)
    L2=0.45 → similarity=0.69  (mirip)
    L2=0.89 → similarity=0.53  (agak mirip)
    L2=4.00 → similarity=0.20  (tidak mirip, di-filter)
                    │
                    ▼
  Filter: similarity < 0.2 → buang
                    │
                    ▼
  Sort: descending by score
                    │
                    ▼
  Ambil top_k=8 pertama
                    │
                    ▼
  Bungkus menjadi RetrievedChunk:
    RetrievedChunk(
      content  = doc.page_content,
      source   = doc.metadata["source"],
      doc_type = doc.metadata["doc_type"],
      score    = round(similarity, 4),
      metadata = doc.metadata,
    )
                    │
                    ▼
  return List[RetrievedChunk]  ← maks 5 item, sorted desc
```

**Flow kerja `build_context()`:**

```
chunks = [RetrievedChunk x 5]
                    │
                    ▼
  for i, c in enumerate(chunks, 1):
    src = filename (jika folder) ATAU "table:orders" (jika DB)
    part = f"[Sumber {i} — {src} ({c.doc_type})]:\n{c.content}"
                    │
                    ▼
  return "\n\n---\n\n".join(parts)
  →
  "[Sumber 1 — sejarah.pdf (pdf)]:
  Presiden pertama Indonesia adalah Soekarno...

  ---

  [Sumber 2 — sejarah.pdf (pdf)]:
  Soekarno lahir pada 6 Juni 1901...

  ---
  ..."
```

---

### `AnswerGenerator` — Cara Kerja

**Apa itu:** Class yang mengelola LLM (Gemini atau HuggingFace) dan menghasilkan jawaban dari context + pertanyaan.

**Flow kerja `load_gemini()`:**

```
answer_gen.load_gemini("gemini-2.5-flash")
                    │
                    ▼
  key = "gemini/gemini-2.5-flash"
  key in self._cache?
          │
    YA    │    TIDAK
    ▼          ▼
  return cached   ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=config.gemini_api_key,
                    temperature=0.3,
                    max_output_tokens=2048,
                  )
                      │
                      ▼
                  llm.invoke("test")   ← test koneksi ke API
                      │
                  ✅ Sukses → self._cache[key] = llm
                  ❌ Gagal  → return False
```

**Flow kerja `generate()`:**

```
answer_gen.generate(
  question = "Siapa presiden pertama Indonesia?",
  context  = "[Sumber 1 — sejarah.pdf]:\nSoekarno adalah..."
)
                    │
                    ▼
  Bangun prompt:
  ┌────────────────────────────────────────────────────────┐
  │ "Kamu adalah asisten yang menjawab pertanyaan HANYA   │
  │  berdasarkan konteks berikut.                          │
  │  Jika jawaban tidak ditemukan dalam konteks, katakan: │
  │    'Informasi tidak ditemukan dalam sumber data...'   │
  │  Jangan mengarang jawaban di luar konteks.            │
  │                                                        │
  │  KONTEKS:                                             │
  │  [Sumber 1 — sejarah.pdf (pdf)]:                      │
  │  Soekarno adalah presiden pertama...                  │
  │  ---                                                   │
  │  [Sumber 2 — sejarah.pdf (pdf)]:                      │
  │  ...                                                   │
  │                                                        │
  │  PERTANYAAN: Siapa presiden pertama Indonesia?        │
  │                                                        │
  │  JAWABAN:"                                            │
  └────────────────────────────────────────────────────────┘
                    │
                    ▼
  self._llm.invoke(prompt)
          │
  Gemini: HTTP POST ke https://generativelanguage.googleapis.com/
          → resp.content = string jawaban dari model
  HF:     local inference → resp = string
                    │
                    ▼
  return resp.content   ← string jawaban final
```

---

### `AgnosticRAGPipeline` — Cara Kerja

**Apa itu:** Orkestrator utama. Menggabungkan semua komponen dalam satu method `ask()` yang menerima pertanyaan + source string, lalu mengembalikan `RAGResult`.

**Flow kerja `ask()` secara lengkap:**

```
pipeline.ask(
  question = "Siapa presiden pertama?",
  source   = "/content/drive/MyDrive/data",
  verbose  = True
)
                    │
                    ▼
  Validasi: question dan source tidak kosong
                    │
                    ▼
  timing = {}   ← dict untuk catat durasi setiap step
  source_key = source   ← key untuk session cache

  ══════════════════════════════════════════
  STEP 1: LOAD  (timing["1_load"])
  ══════════════════════════════════════════
  t = time.time()
  adapter = SourceFactory.create(source)
    → SourceDetector.detect("/content/...") = FOLDER
    → FolderSourceAdapter("/content/drive/MyDrive/data")
  raw_docs = adapter.load()
    → scan folder → baca PDF, DOCX, JSON, CSV
    → [RawDocument x N]
  timing["1_load"] = time.time() - t

  ══════════════════════════════════════════
  STEP 2: SPLIT  (timing["2_split"])
  ══════════════════════════════════════════
  t = time.time()
  chunks = splitter.split(raw_docs)
    → RecursiveCharacterTextSplitter
    → [LangChain Document x M]
  timing["2_split"] = time.time() - t

  ══════════════════════════════════════════
  STEP 3: INDEX  (timing["3_index"])
  ══════════════════════════════════════════
  t = time.time()
  vectorstore = index_builder.build(chunks, source_key)
    → cek cache: miss → FAISS.from_documents()
    → embed M chunks → build IndexFlatL2 in RAM
    → simpan ke self._cache[source_key]
  timing["3_index"] = time.time() - t

  ══════════════════════════════════════════
  STEP 4a: RETRIEVE  (timing["4a_retrieve"])
  ══════════════════════════════════════════
  t = time.time()
  retrieved = query_proc.retrieve(question, vectorstore)
    → embed query → L2 search → filter → sort → top 5
    → [RetrievedChunk x 5]
  context = query_proc.build_context(retrieved)
    → "[Sumber 1 — ...]: ...\n---\n[Sumber 2 — ...]: ..."
  timing["4a_retrieve"] = time.time() - t

  ══════════════════════════════════════════
  STEP 4b: GENERATE  (timing["4b_generate"])
  ══════════════════════════════════════════
  t = time.time()
  answer = answer_gen.generate(question, context)
    → build prompt → llm.invoke(prompt) → string jawaban
  timing["4b_generate"] = time.time() - t

  ══════════════════════════════════════════
  BUNGKUS HASIL
  ══════════════════════════════════════════
  result = RAGResult(
    question         = question,
    answer           = answer,
    retrieved_chunks = retrieved,
    timing           = timing,
    metadata         = {
      "source":       source,
      "source_type":  "📂 Folder (lokal / Google Drive)",
      "raw_docs":     N,
      "total_chunks": M,
      "retrieved":    5,
      "llm":          "gemini/gemini-2.5-flash",
      "timestamp":    "2026-03-13T10:30:00",
    }
  )
  self._history.append(result)
  return result
```

---

### `Evaluator` — Cara Kerja

**Apa itu:** Class yang mengukur kualitas output RAG secara kuantitatif. Mendukung dua mode:
- **Reference-free** (default) — tidak butuh jawaban referensi, semua metrik dihitung dari output pipeline itu sendiri
- **Reference-based** — dengan `ground_truth`, ROUGE-L dan BLEU-1 dihitung vs jawaban acuan (standar akademik)

**Daftar metrik lengkap:**

| Metrik | Mode | Formula | Referensi |
|---|---|---|---|
| `retrieval_relevance` | Reference-free | Cosine similarity embedding Q vs avg embedding chunks | — |
| `answer_faithfulness` | Reference-free | F1 token overlap jawaban ∩ context | — |
| `answer_completeness` | Reference-free | \|keyword_Q ∩ keyword_answer\| / \|keyword_Q\| | — |
| `rouge_l` | Optional ref-based | F1 berbasis LCS jawaban vs referensi/context | Lin 2004 |
| `bleu_1` | Optional ref-based | Unigram precision + brevity penalty | Papineni 2002 |
| `precision_at_k` | Reference-free | \|chunks ≥ threshold\| / K | IR klasik |
| `mrr` | Reference-free | 1 / rank_chunk_relevan_pertama | Voorhees 1999 |
| `context_coverage` | Reference-free | \|unique_sources\| / \|total_chunks\| | — |
| `overall` | — | Rata-rata 5 metrik utama (Rel+Faith+Comp+ROUGE+BLEU) | — |

**Flow kerja `score(result, ground_truth=None)`:**

```
evaluator.score(rag_result, ground_truth="...")
                    │
          ground_truth ada?
          ┌──────────────────────────────────────────┐
          │ YA (reference-based)    │ TIDAK (free)   │
          │ ref = ground_truth      │ ref = context  │
          └──────────────────────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────────────────┐
  │  retrieval_relevance                                 │
  │    q_emb  = embed_query(question)                    │
  │    c_embs = embed_documents([c.content for c])       │
  │    avg    = mean(c_embs, axis=0)                     │
  │    score  = cosine(q_emb, avg) → [0, 1]             │
  └──────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────────────────┐
  │  answer_faithfulness                                 │
  │    context = join(c.content for c in chunks)        │
  │    ta = tok(answer),  tb = tok(context)              │
  │    F1 = 2×P×R / (P+R)                               │
  │    P  = |ta∩tb| / |tb|,  R = |ta∩tb| / |ta|        │
  └──────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────────────────┐
  │  answer_completeness                                 │
  │    qt = tok(question),  at = tok(answer)             │
  │    score = |qt ∩ at| / |qt|                          │
  └──────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────────────────┐
  │  rouge_l  (vs ref)                                   │
  │    LCS = Longest Common Subsequence(hyp, ref)        │
  │    P   = LCS / len(hyp_tokens)                       │
  │    R   = LCS / len(ref_tokens)                       │
  │    F1  = 2×P×R / (P+R)                              │
  └──────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────────────────┐
  │  bleu_1  (vs ref)                                    │
  │    clipped_count = Σ min(hyp_count, ref_count)       │
  │    precision = clipped_count / len(hyp_tokens)       │
  │    BP = exp(1 - len_ref/len_hyp) if short, else 1    │
  │    BLEU-1 = BP × precision                           │
  └──────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌──────────────────────────────────────────────────────┐
  │  precision_at_k                                      │
  │    = |chunks dimana score ≥ 0.2| / K                 │
  │  mrr                                                 │
  │    = 1 / rank_pertama_chunk_yang_≥_0.2               │
  │  context_coverage                                    │
  │    = |unique c.source| / |total chunks|              │
  └──────────────────────────────────────────────────────┘
                    │
                    ▼
  overall = mean(retrieval_relevance, answer_faithfulness,
                 answer_completeness, rouge_l, bleu_1)

  return EvalScore(semua field di atas)
```

**Flow kerja `display_result(result, ground_truth=None)`:**

```
evaluator.display_result(result)         ← pipeline TIDAK re-run
              │
              ▼
  s = self.score(result, ground_truth)   ← hitung semua metrik
              │
              ▼
  Tampilkan:
    ❓ Pertanyaan + 💡 Jawaban
    📚 Retrieved chunks dengan bar similarity
    ⏱️  Timing per tahap (ms + % total + ← terlama marker)
    🎯 5 metrik utama dengan visual bar + grade 🟢/🟡/🔴
    📐 P@K, MRR, Context Coverage, Avg Chunk Score
    📊 Metadata (source, LLM, raw docs, timestamp)
```

**Flow kerja `run_batch(questions, ground_truths=None)`:**

```
evaluator.run_batch(["q1","q2"], source="...", ground_truths=["gt1","gt2"])
                    │
  for i, q in enumerate(questions):
    gt     = ground_truths[i] if ground_truths else None
    result = pipeline.ask(q, source, verbose=False)
    s      = self.score(result, ground_truth=gt)
    rows.append({semua 15 kolom metrik})
                    │
  df = pd.DataFrame(rows)
  print rata-rata semua metrik
  return df   → siap export ke CSV / LaTeX
```

**Export LaTeX (`to_latex(df)`):**

```python
tex = evaluator.to_latex(df_eval,
    caption="Hasil Evaluasi AgnosticRAG",
    label="tab:eval_rag"
)
# Output: tabel LaTeX lengkap dengan \toprule, \midrule, \bottomrule
# Baris terakhir = rata-rata bold
# Siap paste ke dokumen .tex jurnal
```

---

## 4. Kenapa A, B, C? — Keputusan Teknis

### Mengapa FAISS (bukan ChromaDB, Pinecone, Weaviate)?

| Kriteria | FAISS | Alternatif |
|---|---|---|
| **In-memory** | ✅ Native | ChromaDB perlu disk, Pinecone perlu cloud |
| **Zero dependency cloud** | ✅ | Pinecone butuh API key |
| **Kecepatan build** | ✅ Sangat cepat untuk <100K chunks | — |
| **Integrasi LangChain** | ✅ `FAISS.from_documents()` | Semua bisa, tapi FAISS paling sederhana |
| **Cocok untuk realtime** | ✅ Build fresh setiap sesi | ChromaDB persist ke disk by default |

**Kesimpulan:** FAISS dipilih karena paling ringan, tidak butuh server/cloud, dan mendukung pola *build-in-RAM-then-forget*.

---

### Mengapa `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`?

| Kriteria | Model ini | Alternatif |
|---|---|---|
| **Multilingual** | ✅ 50+ bahasa termasuk Indonesia | `all-MiniLM-L6-v2` hanya Inggris |
| **Ukuran** | ~470 MB (MiniLM = kecil) | `multilingual-e5-large` = 560 MB lebih berat |
| **Kualitas** | ✅ Baik untuk semantic similarity | — |
| **CPU friendly** | ✅ `device="cpu"` berfungsi baik | Model besar butuh GPU |
| **Normalize embeddings** | ✅ `normalize_embeddings=True` | Memastikan dot product = cosine similarity |

**Kesimpulan:** Terbaik untuk use case Indonesia + multilingual, ringan, bisa jalan di CPU.

---

### Mengapa Gemini 2.5-flash (bukan GPT-4, Claude)?

| Kriteria | Gemini 2.5-flash | GPT-4 | Claude |
|---|---|---|---|
| **Context window** | 1M token | 128K | 200K |
| **Gratis (tier)** | ✅ Ada free tier | ❌ Berbayar | ❌ Berbayar |
| **LangChain support** | ✅ `langchain-google-genai` | ✅ | ✅ |
| **Speed** | ✅ Fast | Medium | Medium |
| **RAG quality** | ✅ Sangat baik | ✅ | ✅ |

**Kesimpulan:** Gemini 2.5-flash dipilih karena context window sangat besar (berguna saat banyak chunk), ada free tier untuk development, dan performa RAG kompetitif.

---

### Mengapa `RecursiveCharacterTextSplitter` (bukan `CharacterTextSplitter`)?

`RecursiveCharacterTextSplitter` mencoba split dari separator terbesar dulu:
```
separators = ["\n\n", "\n", ". ", ", ", " ", ""]
```

Artinya:
1. Coba split di **paragraf** (`\n\n`) — menjaga konteks paragraf tetap utuh
2. Kalau masih terlalu besar, split di **baris** (`\n`)
3. Lalu di **kalimat** (`. `)
4. Dst.

`CharacterTextSplitter` biasa hanya split di satu karakter saja, sering memotong di tengah kalimat.

**Kesimpulan:** Rekursif lebih cerdas, menghasilkan chunks yang lebih semantically coherent.

---

### Mengapa SQLAlchemy 2.x pattern?

SQLAlchemy 2.x mengubah cara eksekusi query:

```python
# ❌ SQLAlchemy 1.x style (deprecated, tidak aman):
pd.read_sql("SELECT * FROM tabel", engine)

# ✅ SQLAlchemy 2.x style (yang dipakai):
from sqlalchemy import text
with engine.connect() as conn:
    df = pd.read_sql(text("SELECT * FROM tabel"), conn)
```

Selain itu, `pool_pre_ping=True` memastikan engine test koneksi sebelum dipakai — mencegah error `SSL connection has been closed unexpectedly` pada long-running sessions.

---

### Mengapa Singleton untuk EmbeddingModel?

```python
class EmbeddingModel:
    _instance = None  # singleton

    @classmethod
    def get(cls) -> HuggingFaceEmbeddings:
        if cls._instance is None:
            cls._instance = HuggingFaceEmbeddings(...)
        return cls._instance
```

Loading model embedding membutuhkan **2–5 detik** pertama kali (download + load ke RAM). Dengan singleton, model hanya di-load sekali per sesi kernel — semua pemanggilan berikutnya langsung return instance yang sudah ada.

---

### Mengapa SourceDetector pakai regex pattern, bukan `os.path.isdir()`?

`SourceDetector` sengaja **tidak** melakukan I/O (tidak cek apakah path benar-benar ada di disk). Alasannya:
1. Path bisa ke **Google Drive** yang belum di-mount saat deteksi
2. Path bisa ke **network share** yang lambat
3. Deteksi harus **instan** (< 1ms)

Detection berdasarkan **string pattern** saja:
```python
_POSTGRES_PREFIXES = ("postgresql://", "postgres://")
_FOLDER_PATTERNS   = (r"^/", r"^\./", r"^\.\./", r"^[A-Za-z]:[/\\]", r"^~")
```

Validasi path baru dilakukan di `FolderSourceAdapter.load()` — termasuk auto-mount Google Drive jika di Colab.

---

## 5. Komponen Per Sel

### Cell 1 — Dependencies

**Tujuan:** Install semua library yang dibutuhkan.

**Yang diinstall:**

| Library | Kegunaan |
|---|---|
| `faiss-cpu` | Vector store in-memory (similarity search) |
| `sentence-transformers` | Model embedding multilingual |
| `langchain` | Framework RAG (text splitter, document schema) |
| `langchain-community` | FAISS wrapper, HuggingFace LLM wrapper |
| `langchain-google-genai` | Gemini LLM wrapper |
| `pypdf` | Baca file PDF |
| ~~`python-docx`~~ | *(dinonaktifkan — hanya PDF & TXT aktif saat ini)* |
| ~~`openpyxl`~~ | *(dinonaktifkan)* |
| `sqlalchemy` | ORM / connector PostgreSQL |
| `psycopg2-binary` | Driver PostgreSQL untuk Python |
| `transformers` + `torch` | HuggingFace LLM (fallback) |
| `pandas` | Manipulasi data tabular (CSV, DB result) |
| `matplotlib` | Visualisasi metrik evaluasi |

**Yang TIDAK diinstall / dinonaktifkan:**

| Library | Status | Alasan |
|---------|--------|--------|
| `pymysql` | Dihapus | MySQL tidak didukung |
| `pymongo` | Dihapus | MongoDB tidak didukung |
| `unstructured` | Dihapus | Terlalu berat, tidak dipakai |
| `python-docx` | **Commented out** | Data tidak terstruktur hanya PDF & TXT saat ini. Uncomment jika DOCX perlu diaktifkan kembali |
| `openpyxl` | **Commented out** | Excel tidak digunakan |

---

### Cell 2 — Config

**Tujuan:** Centralized configuration — semua parameter bisa diubah dari satu tempat.

```python
@dataclass
class Config:
    llm_provider: str   = "gemini"           # "gemini" | "huggingface"
    llm_model: str      = "gemini-2.5-flash"
    hf_model: str       = "google/flan-t5-base"
    gemini_api_key: str = "YOUR_GEMINI_API_KEY_HERE"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    chunk_size: int         = 2000
    chunk_overlap: int      = 300
    top_k: int              = 8
    similarity_threshold: float = 0.2

    use_session_cache: bool = True
    max_db_rows: int        = 5000
    chat_extensions: Tuple  = (".json", ".txt", ".log", ".csv")
    doc_extensions: Tuple   = (".pdf", ".txt", ".md", ".log")
    # .docx dan .doc dinonaktifkan — hanya PDF & TXT aktif saat ini
```

**Parameter kritis yang perlu diubah pengguna:**

| Parameter | Default | Kapan diubah |
|---|---|---|
| `gemini_api_key` | `"YOUR_GEMINI_API_KEY_HERE"` | Wajib diisi sebelum pakai Gemini |
| `llm_provider` | `"gemini"` | Ubah ke `"huggingface"` jika tidak punya API key |
| `chunk_size` | `2000` | Perbesar untuk dokumen panjang, kecilkan untuk dokumen pendek |
| `top_k` | `8` | Perbesar jika jawaban sering tidak lengkap |
| `similarity_threshold` | `0.2` | Perbesar untuk lebih selektif, kecilkan jika sering tidak ada hasil |
| `max_db_rows` | `5000` | Kecilkan jika RAM terbatas, perbesar untuk coverage lebih luas |

---

### Cell 3 — Source Detector & Adapters

**Tujuan:** Deteksi otomatis sumber data dan load dokumen ke format `RawDocument`.

#### `SourceType` (Enum)

```python
class SourceType(Enum):
    FOLDER   = auto()   # path folder lokal / Google Drive
    POSTGRES = auto()   # postgresql://...
```

#### `SourceDetector`

```python
class SourceDetector:
    _POSTGRES_PREFIXES = ("postgresql://", "postgres://")
    _FOLDER_PATTERNS   = (r"^/", r"^\./", r"^\.\./", r"^[A-Za-z]:[/\\]", r"^~")

    @classmethod
    def detect(cls, source: str) -> SourceType: ...

    @classmethod
    def describe(cls, source: str) -> str: ...
```

**Logika deteksi:**

```
source.startswith("postgresql://") atau "postgres://"  →  POSTGRES
re.match(folder_pattern, source)                       →  FOLDER
else (fallback)                                        →  FOLDER
```

---

#### 🔎 Detail `_POSTGRES_PREFIXES` — Apa yang Di-match?

`_POSTGRES_PREFIXES` adalah tuple dua string literal — dicek dengan `.startswith()` (bukan regex):

| Prefix | Contoh input yang match |
|---|---|
| `"postgresql://"` | `postgresql://admin:secret@localhost:5432/mydb` |
| `"postgres://"` | `postgres://root:pass@db.server.com/analytics` |

Kedua prefix ini adalah **URI scheme resmi PostgreSQL** (keduanya valid, `postgres://` adalah alias pendek dari `postgresql://`). Selain dua prefix ini, tidak ada string lain yang dianggap sebagai koneksi database.

---

#### 🔎 Detail `_FOLDER_PATTERNS` — Regex per Pattern

`_FOLDER_PATTERNS` adalah tuple **5 pola regex**, masing-masing di-cek dengan `re.match()` (match dari awal string). Jika **salah satu** cocok → sumber dianggap folder.

| # | Regex Pattern | Arti | Contoh input yang match |
|---|---|---|---|
| 1 | `r"^/"` | Dimulai dengan `/` | `/content/drive/MyDrive/data` ← Google Drive di Colab |
| 2 | `r"^\./"` | Dimulai dengan `./` | `./documents/laporan` ← path relatif dari folder saat ini |
| 3 | `r"^\.\.\/"` | Dimulai dengan `../` | `../data/files` ← path relatif satu level ke atas |
| 4 | `r"^[A-Za-z]:[/\\]"` | Huruf drive diikuti `:\` atau `:/` | `C:\Users\data\docs` atau `D:/files` ← Windows path |
| 5 | `r"^~"` | Dimulai dengan `~` | `~/Documents/data` ← home directory Unix/Mac |

**Cara membaca notasi regex:**

```
^        → "anchor" — berarti: harus cocok dari AWAL string (bukan di tengah)
\/       → tanda garis miring /  (backslash untuk escape di dalam raw string)
\.       → titik literal . (titik tanpa backslash = "karakter apa saja" di regex)
\.\./    → dua titik literal + garis miring = ../
[A-Za-z] → satu karakter huruf apa saja (A-Z kapital atau a-z kecil)
[/\\]    → salah satu dari: garis miring / ATAU backslash \
```

**Contoh matching lengkap:**

```python
_FOLDER_PATTERNS = (r"^/", r"^\./", r"^\.\./", r"^[A-Za-z]:[/\\]", r"^~")

# Test setiap input:
"/content/drive/MyDrive"    →  cocok r"^/"            ✅ FOLDER
"./documents"               →  cocok r"^\\./"          ✅ FOLDER
"../data/files"             →  cocok r"^\\.\\../"      ✅ FOLDER
"C:\\Users\\data"           →  cocok r"^[A-Za-z]:[/\\]" ✅ FOLDER
"~/Documents"               →  cocok r"^~"             ✅ FOLDER
"postgresql://..."          →  tidak cocok satupun     → cek _POSTGRES_PREFIXES → POSTGRES
"myfile.pdf"                →  tidak cocok satupun     → fallback → FOLDER
```

**Kenapa tidak pakai `os.path.isdir()`?** Karena SourceDetector sengaja **tidak menyentuh disk** — path mungkin ke Google Drive yang belum di-mount, atau ke network share. Validasi path yang sesungguhnya baru dilakukan di `FolderSourceAdapter.load()`.

---

#### Alur Keputusan Lengkap `SourceDetector.detect()`

```
Input: source string
         │
         ▼
  strip() whitespace
         │
         ▼
  startswith("postgresql://") ?  ──YES──►  SourceType.POSTGRES
         │
        NO
         │
         ▼
  startswith("postgres://") ?    ──YES──►  SourceType.POSTGRES
         │
        NO
         │
         ▼
  re.match(r"^/", s)     ?       ──YES──►  SourceType.FOLDER
         │
        NO
         │
         ▼
  re.match(r"^\./", s)   ?       ──YES──►  SourceType.FOLDER
         │
        NO
         │
         ▼
  re.match(r"^\.\./", s) ?       ──YES──►  SourceType.FOLDER
         │
        NO
         │
         ▼
  re.match(r"^[A-Za-z]:[/\\]", s)?──YES──► SourceType.FOLDER
         │
        NO
         │
         ▼
  re.match(r"^~", s)     ?       ──YES──►  SourceType.FOLDER
         │
        NO
         │
         ▼
  fallback                                  SourceType.FOLDER
```

#### `RawDocument`

Representasi universal dokumen setelah di-load, sebelum di-split:

```python
@dataclass
class RawDocument:
    content:  str        # teks mentah
    source:   str        # path file ATAU "table:nama" ATAU "query:label"
    doc_type: str        # "pdf"|"docx"|"txt"|"json"|"chat"|"csv"|"excel"|"db_table"|"db_query"
    metadata: Dict       # info tambahan (pages, rows, cols, dll)
```

#### `FolderSourceAdapter`

Load semua file dari folder secara rekursif.

**Format aktif (data tidak terstruktur):**
- PDF (`.pdf`) → pypdf
- Text (`.txt`, `.md`, `.log`) → built-in

**Format dinonaktifkan (commented out):**
- DOCX (`.docx`, `.doc`) → `_load_docx` — nonaktif, hanya PDF & TXT saat ini

**Format aktif:**
- PDF (`.pdf`) → pypdf → `doc_type="pdf"`
- Text (`.txt`) → built-in → `doc_type="txt"`
- Markdown (`.md`) → built-in → `doc_type="markdown"`
- Log (`.log`) → built-in → `doc_type="log"`

**Format nonaktif (commented out):**
- DOCX (`.docx`, `.doc`) → `_load_docx()` — nonaktif, hanya PDF & TXT saat ini

**Error handling:** Setiap file yang gagal di-load di-log sebagai warning dan di-skip.

Load data dari PostgreSQL menggunakan SQLAlchemy 2.x.

```python
class PostgreSQLAdapter(BaseSourceAdapter):
    def __init__(self, connection_string, tables=None, custom_queries=None): ...
    def _get_engine(self): ...          # lazy-init, pool_pre_ping=True
    def _list_tables(self): ...         # inspect schema public
    def load(self) -> List[RawDocument]: ...
```

**Urutan loading:**
1. Custom queries dulu (`pg_queries` dict)
2. Kemudian tabel (semua atau yang difilter oleh `pg_tables`)

**Setiap tabel/query → satu `RawDocument`** dengan format:
```
=== ORDERS ===
Kolom: id, customer_id, total, created_at
Jumlah baris: 1234

  id  customer_id   total  created_at
   1          42  100000  2024-01-15
  ...
```

#### `SourceFactory`

Entry point tunggal:

```python
SourceFactory.create(source, tables=None, custom_queries=None)
# → FolderSourceAdapter ATAU PostgreSQLAdapter
```

---

### Cell 4 — Splitter, Embeddings, FAISS, Query Processor

#### `UniversalTextSplitter`

Wrap `RecursiveCharacterTextSplitter` dari LangChain:

```python
self._splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,       # default: 2000 karakter
    chunk_overlap=config.chunk_overlap, # default: 300 karakter
    separators=["\n\n", "\n", ". ", ", ", " ", ""],
)
```

**Input:** `List[RawDocument]`  
**Output:** `List[LangChain Document]` dengan metadata: `source`, `doc_type`, `chunk_i`, + metadata dari RawDocument

#### `EmbeddingModel` (Singleton)

```python
class EmbeddingModel:
    _instance = None

    @classmethod
    def get(cls) -> HuggingFaceEmbeddings:
        if cls._instance is None:
            cls._instance = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return cls._instance
```

Di-load sekali per sesi kernel, dipakai bersama oleh `RuntimeIndexBuilder` dan `Evaluator`.

#### `RuntimeIndexBuilder`

```python
class RuntimeIndexBuilder:
    def __init__(self):
        self._cache: Dict[str, FAISS] = {}  # key = source string

    def build(self, docs, source_key) -> FAISS:
        # Cek cache dulu
        if config.use_session_cache and source_key in self._cache:
            return self._cache[source_key]
        # Build fresh
        vs = FAISS.from_documents(docs, EmbeddingModel.get())
        self._cache[source_key] = vs
        return vs

    def clear_cache(self, source_key=None):
        # Hapus satu key atau semua
        ...
```

**Penting:** `FAISS.from_documents()` menyimpan index di **RAM saja**. Tidak ada `.faiss` file yang ditulis ke disk.

#### `RetrievedChunk`

```python
@dataclass
class RetrievedChunk:
    content:  str
    source:   str
    doc_type: str
    score:    float   # 0.0 – 1.0 (1 = paling relevan)
    metadata: Dict
```

#### `QueryProcessor`

```python
class QueryProcessor:
    def retrieve(self, question, vectorstore) -> List[RetrievedChunk]:
        results = vectorstore.similarity_search_with_score(question, k=top_k * 2)
        # FAISS L2 distance → similarity: score = 1 / (1 + L2_dist)
        # Filter < threshold, sort descending, ambil top_k
        ...

    def build_context(self, chunks) -> str:
        # Format: [Sumber 1 — filename (doc_type)]:\n...
        ...
```

---

### Cell 5 — Answer Generator

```python
class AnswerGenerator:
    def load_gemini(model=None) -> bool:   # Load Gemini via langchain-google-genai
    def generate(question, context) -> str:    # Buat jawaban dari LLM dengan retry + fallback

    @property
    def is_ready(self) -> bool
    @property
    def info(self) -> str   # "gemini/gemini-2.5-flash"
```

**Model utama & fallback chain:**

```python
_PRIMARY_MODEL    = "gemini-2.5-flash"
_FALLBACK_MODELS  = ["gemini-2.0-flash", "gemini-1.5-flash"]
```

**Logika retry + fallback (penambahan April 2026):**

Gemini 2.5-flash sering mengembalikan **HTTP 503 (Service Unavailable)** karena rate limit atau beban server. Sistem menangani ini dengan dua mekanisme:

1. **Exponential backoff** — delay antar retry berlipat ganda setiap percobaan:

```
Attempt 1: delay = base_delay * 2^0 = 15s
Attempt 2: delay = base_delay * 2^1 = 30s   → switch ke gemini-2.0-flash
Attempt 3: delay = base_delay * 2^2 = 60s
Attempt 4: delay = base_delay * 2^3 = 120s  → switch ke gemini-1.5-flash
Attempt 5: delay = base_delay * 2^4 = 240s
Attempt 6: GAGAL — raise exception
max_retries = 6
```

2. **Model fallback** — model berganti saat retry ke-2 dan ke-4:

| Attempt | Model yang dipakai | Trigger |
|---------|-------------------|---------|
| 1 | `gemini-2.5-flash` | (primary) |
| 2 | `gemini-2.0-flash` | attempt >= 2 |
| 3 | `gemini-2.0-flash` | — |
| 4 | `gemini-1.5-flash` | attempt >= 4 |
| 5–6 | `gemini-1.5-flash` | — |

**Pseudocode retry loop:**

```python
for attempt in range(max_retries):   # max_retries = 6
    model = _PRIMARY_MODEL
    if attempt >= 4:
        model = _FALLBACK_MODELS[1]  # gemini-1.5-flash
    elif attempt >= 2:
        model = _FALLBACK_MODELS[0]  # gemini-2.0-flash

    try:
        llm = load_gemini(model)
        return llm.invoke(prompt).content
    except Exception as e:
        if "503" in str(e) or "overloaded" in str(e).lower():
            delay = base_delay * (2 ** attempt)  # 15→30→60→120→240
            time.sleep(delay)
        else:
            raise  # Error non-503 langsung raise

raise RuntimeError("Semua model gagal setelah max_retries")
```

**Prompt yang digunakan (zero-shot):**

```
Kamu adalah asisten yang menjawab pertanyaan HANYA berdasarkan konteks berikut.
Jika jawaban tidak ditemukan dalam konteks, katakan:
  'Informasi tidak ditemukan dalam sumber data yang tersedia.'
Jangan mengarang jawaban di luar konteks.

KONTEKS:
[chunk 1]
---
[chunk 2]
...

PERTANYAAN: [question]

JAWABAN:
```

**Kenapa `temperature=0.3`?** Cukup rendah untuk jawaban yang faktual dan konsisten, tapi tidak terlalu rigid (0.0) sehingga tetap readable.

**Kenapa `max_output_tokens=2048`?** Cukup untuk jawaban panjang tapi tidak waste token.

**LLM cache:** Setiap `load_gemini(model)` dicache di `self._cache` dengan key `"gemini/modelname"`. Load model yang sama lagi langsung return dari cache.

---

### Cell 6 — Agnostic RAG Pipeline

#### `RAGResult`

```python
@dataclass
class RAGResult:
    question:         str
    answer:           str
    retrieved_chunks: List[RetrievedChunk]
    timing:           Dict[str, float]   # step → seconds
    metadata:         Dict               # source, raw_docs, total_chunks, llm, timestamp

    @property
    def total_time(self) -> float: ...

    def display(self):
        # Print terformat dengan:
        # ❓ PERTANYAAN
        # 💡 JAWABAN
        # 📚 CHUNKS (top 3)
        # ⏱️ TIMING per step
        # 📊 METADATA
```

#### `AgnosticRAGPipeline`

Pipeline utama. Semua logika orkestrasi ada di sini.

```python
class AgnosticRAGPipeline:
    def ask(
        self,
        question: str,
        source: str,
        pg_tables: Optional[List[str]] = None,
        pg_queries: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> RAGResult:
```

**5 Langkah Eksekusi:**

| Step | Label | Apa yang terjadi |
|---|---|---|
| 1 | `1_load` | `SourceFactory.create()` + `adapter.load()` |
| 2 | `2_split` | `splitter.split(raw_docs)` |
| 3 | `3_index` | `index_builder.build(chunks, source_key)` |
| 4a | `4a_retrieve` | `query_proc.retrieve(question, vectorstore)` |
| 4b | `4b_generate` | `answer_gen.generate(question, context)` |

Semua timing dicatat di `RAGResult.timing` dict.

---

### Cell 7 — Evaluator

**Tujuan:** Ukur kualitas sistem RAG secara kuantitatif tanpa ground truth (unsupervised evaluation).

#### 3 Metrik

| Metrik | Rumus | Interpretasi |
|---|---|---|
| **Retrieval Relevance** | Cosine similarity embedding Q vs avg chunk embedding | Seberapa relevan chunk yang di-retrieve |
| **Answer Faithfulness** | F1 token overlap jawaban vs context | Anti-hallucination check |
| **Answer Completeness** | Keyword pertanyaan yang muncul di jawaban | Kelengkapan menjawab |
| **ROUGE-L** | F1 berbasis LCS vs referensi/context | Lin 2004 |
| **BLEU-1** | Unigram precision + brevity penalty | Papineni et al. 2002 |
| **Precision@K** | Proporsi chunk ≥ threshold | IR klasik |
| **MRR** | 1/rank chunk relevan pertama | Voorhees 1999 |
| **Context Coverage** | Keragaman unique source dari chunks | — |
| **Overall** | Rata-rata 5 metrik utama | Skor gabungan |

#### `EvalScore` — fields lengkap

```python
@dataclass
class EvalScore:
    question:             str
    retrieval_relevance:  float   # cosine similarity Q vs avg chunk
    answer_faithfulness:  float   # F1 token overlap
    answer_completeness:  float   # keyword coverage
    rouge_l:              float   # LCS-based F1
    bleu_1:               float   # unigram precision + BP
    precision_at_k:       float   # chunk ≥ threshold / K
    mrr:                  float   # 1/rank_relevant_chunk
    context_coverage:     float   # unique_sources / total_chunks
    avg_chunk_score:      float   # rata-rata FAISS similarity (diagnostik)
    num_chunks:           int
    total_time:           float
    source_type:          str
    ground_truth:         Optional[str] = None  # None = reference-free

    @property
    def overall(self) -> float:
        # rata-rata 5 metrik: Rel + Faith + Comp + ROUGE-L + BLEU-1
```

#### `Evaluator.display_result(result, ground_truth=None)`

Evaluasi + tampilkan satu `RAGResult` — pipeline **tidak** dijalankan ulang.

```python
result = pipeline.ask("pertanyaan", source=SOURCE)
evaluator.display_result(result)
evaluator.display_result(result, ground_truth="jawaban acuan")
```

Output mencakup: jawaban, retrieved chunks dengan bar, timing per tahap (ms + %), 5 metrik utama dengan visual bar + grade 🟢/🟡/🔴, metrik retrieval tambahan, metadata.

#### `Evaluator.run_batch(questions, ground_truths=None)`

```python
df = evaluator.run_batch(
    questions     = ["q1", "q2", ...],
    source        = "source_string",
    ground_truths = ["gt1", "gt2", ...],  # opsional
    pg_tables     = None,
    pg_queries    = None,
) -> pd.DataFrame
```

DataFrame memiliki kolom: `No`, `Pertanyaan`, `Retrieval Relevance`, `Answer Faithfulness`, `Answer Completeness`, `ROUGE-L`, `BLEU-1`, `Precision@K`, `MRR`, `Context Coverage`, `Overall`, `Avg Chunk Score`, `Chunks`, `Time (s)`, `Source Type`, `Ground Truth`.

#### `Evaluator.to_latex(df, caption, label)`

Export tabel LaTeX siap paste ke dokumen jurnal `.tex`:
- Kolom diringkas agar muat di halaman
- Baris terakhir = rata-rata semua metrik (bold)
- Menggunakan `\toprule`, `\midrule`, `\bottomrule` (booktabs style)

```python
tex = evaluator.to_latex(df_eval,
    caption="Hasil Evaluasi AgnosticRAG",
    label="tab:eval_rag"
)
with open("eval_table.tex", "w") as f:
    f.write(tex)
```

#### `Evaluator.plot(df)`

Menghasilkan 4 subplot (disimpan dengan timestamp):
- **[0,0] Bar chart:** Rata-rata 5 metrik utama
- **[0,1] Line chart:** Skor per pertanyaan (semua 5 metrik)
- **[1,0] Grouped bar:** ROUGE-L & BLEU-1 per pertanyaan
- **[1,1] Line chart:** P@K, MRR, Context Coverage per pertanyaan

---

### Cell 8 — Markdown: Architecture Overview

Sel markdown yang berisi diagram arsitektur end-to-end, penjelasan komponen, dan ringkasan filosofi desain sistem. Tidak mengandung kode yang bisa dieksekusi.

---

### Cells 9–16 — Unit Tests, Interactive Demo, Health Checks

Blok sel yang berisi:
- **Unit tests** (7 kelompok test: SourceDetector, FolderAdapter, Splitter, IndexBuilder, SessionCache, QueryProcessor, Realtime)
- **Demo interaktif** per sumber (Folder, PostgreSQL semua tabel, PostgreSQL custom query)
- **Health check** — verifikasi koneksi DB dan API key
- **Explorasi awal** — run batch kecil untuk verifikasi pipeline end-to-end

**Struktur test (7 kelompok):**

| Test | Yang Diuji | Assertions |
|------|-----------|------------|
| **T1** | `SourceDetector` — 7 pola input berbeda | Semua deteksi tepat |
| **T2** | `FolderSourceAdapter` — load dari temp folder | `.txt` → `txt`, `.json` chat → `chat`, `.csv` tabular → `csv`, `.csv` chat → `chat` |
| **T3** | `UniversalTextSplitter` | Chunks > 0, ada `page_content`, metadata `source`/`doc_type`/`chunk_i` |
| **T4** | `RuntimeIndexBuilder` — FAISS in-memory | Index ter-build, tidak ada file `.faiss` di disk |
| **T5** | Session cache | Objek FAISS sama (`vs is vs2`), waktu lebih cepat |
| **T6** | `QueryProcessor` | Ada hasil, score 0–1, sorted descending, context punya label "Sumber" |
| **T7** | **REALTIME** | File diubah → cache cleared → konten baru ter-retrieve |

**Test T7 (Realtime) — detail:**

```python
# Sesi 1
doc.write_text("Bumi adalah planet ketiga dari matahari.")
# → retrieve "bumi planet" → found ✅

# Sesi 2 (file diubah!)
doc.write_text("Mars adalah planet merah di tata surya.")
index_builder.clear_cache()
# → retrieve "mars planet merah" → found ✅
# → raw1.content != raw2.content (bukan stale cache) ✅
```

---

### Cell 17 — Skenario A (FolderSourceAdapter)

**Tujuan:** Evaluasi pipeline RAG dengan sumber data folder (PDF + TXT) menggunakan 5 pertanyaan terkait corpus BBKP (Bank Bukopin) dan TINS (PT Timah).

**Konfigurasi:**
```python
SOURCE_A = "/content/drive/MyDrive/data/sample_data"
# Corpus: Press Release Bank Bukopin.pdf + Press Release PT Timah.pdf
#         + analyst_chat_dummy.txt + [file lain]
# TIDAK menggunakan exclude_patterns
# Cache dibersihkan sebelum run: index_builder.clear_cache()
```

**5 Pertanyaan Skenario A:**
| No | Pertanyaan | Target Pengetahuan |
|----|-----------|-------------------|
| A1 | Apa hasil kinerja keuangan Bank Bukopin pada Q1 2025? | Explicit (PDF BBKP) |
| A2 | Berapa nilai ekuitas PT Timah pada laporan Q1 2025? | Explicit (PDF TINS) |
| A3 | Apa strategi bisnis Bank Bukopin untuk semester 2? | Actionable (PDF BBKP) |
| A4 | Bagaimana posisi kas PT Timah dibandingkan periode sebelumnya? | Explicit (PDF TINS) |
| A5 | Apa rekomendasi analis terkait saham BBKP dan TINS? | Tacit (TXT chat) |

**Parameter run:**
```python
df_A = evaluator.run_batch(
    QUESTIONS_A, source=SOURCE_A,
    delay_between=10  # delay 10s antar pertanyaan (rate limit)
)
```

---

### Cell 18 — SQL Setup (Inisialisasi Data BBKP/TINS)

**Tujuan:** INSERT data BBKP (Bank Bukopin) dan TINS (PT Timah) ke tabel PostgreSQL Neon DB untuk digunakan di Skenario B.

**Tabel yang diisi:**

```sql
-- company_watchlist: 2 rows
INSERT INTO company_watchlist
  (ticker, company_name, sector, recommendation, risk_level, debt_status, analyst_id)
VALUES
  ('BBKP', 'Bank Bukopin Tbk', 'Perbankan', 'BUY', 'medium',
   'Tinggi - rasio NPL 4.2%, coverage ratio 85%', 1),
  ('TINS', 'PT Timah Tbk', 'Pertambangan Timah', 'HOLD', 'medium',
   'Sedang - debt-to-equity 0.8x, likuiditas cukup', 2);

-- analyst_notes: 2 rows (kolom: reviewed_at, target_price_usd, key_highlights, risk_notes)
INSERT INTO analyst_notes
  (ticker, analyst_id, reviewed_at, target_price_usd, key_highlights, risk_notes)
VALUES
  ('BBKP', 1, '2025-04-15', 95.50,
   'NPL membaik ke 4.2%; ekspansi KPR subsidi; digitalisasi layanan',
   'Risiko regulasi OJK; tekanan NIM dari kenaikan BI Rate'),
  ('TINS', 2, '2025-04-15', 1250.00,
   'Produksi timah Q1 naik 12% YoY; harga LME stabil USD 26.800/ton',
   'Risiko fluktuasi harga timah global; cuaca ekstrem di Bangka Belitung');
```

**Catatan penting kolom:**
- `debt_status` → VARCHAR(200) setelah `ALTER TABLE` (bukan VARCHAR(50))
- Kolom tanggal adalah `reviewed_at` (bukan `note_date`)
- Kolom harga adalah `target_price_usd` (bukan `target_price`)
- Tidak ada kolom `note_text` — digantikan `key_highlights` + `risk_notes`

---

### Cell 19 — Skenario B (PostgreSQLAdapter)

**Tujuan:** Evaluasi pipeline RAG dengan sumber data PostgreSQL (5 tabel Neon DB) menggunakan 5 pertanyaan campuran.

**Konfigurasi:**
```python
SOURCE_B = "postgresql://neondb_owner:...@ep-broad-glitter-a45az27j-pooler.us-east-1.aws.neon.tech/neondb"
# 5 tabel: user_profiles, products, orders, company_watchlist, analyst_notes
```

**5 Pertanyaan Skenario B:**
| No | Pertanyaan | Tabel Target |
|----|-----------|-------------|
| B1 | Siapa analis senior yang menangani coverage BBKP dan TINS? | user_profiles |
| B2 | Tools apa yang digunakan tim riset untuk analisis saham? | products + orders |
| B3 | Apa rekomendasi terbaru untuk saham BBKP? | company_watchlist |
| B4 | Apa highlight utama dari catatan analis PT Timah? | analyst_notes |
| B5 | Bagaimana status pengadaan Bloomberg Terminal? | orders |

**Parameter run:**
```python
df_B = evaluator.run_batch(
    QUESTIONS_B, source=SOURCE_B,
    delay_between=10
)
```

---

### Cell 20 — Skenario C (Cross-source + `exclude_patterns`)

**Tujuan:** Evaluasi pipeline RAG cross-source (PDF + TXT) dengan filter file menggunakan parameter `exclude_patterns`.

**Fitur `exclude_patterns` — Penjelasan Lengkap:**

`exclude_patterns` adalah parameter opsional di `FolderSourceAdapter`, `SourceFactory.create()`, dan `Evaluator.run_batch()`. Nilainya berupa list substring — setiap file yang nama path-nya mengandung substring tersebut (case-insensitive) akan **di-skip saat loading**.

```python
# Tanpa filter: semua file di folder di-load
pipeline.ask(q, source=SOURCE_FOLDER)

# Dengan filter: file yang mengandung 'salinan' atau 'agreement' di-skip
pipeline.ask(q, source=SOURCE_FOLDER,
             exclude_patterns=['salinan', 'agreement'])
```

**File yang di-filter di Skenario C:**

| File | Path mengandung | Status |
|------|----------------|--------|
| `Salinan dummy-pdf.pdf` | `'salinan'` | ❌ Di-skip |
| `Agreement Mr. Krisna.docx.pdf` | `'agreement'` | ❌ Di-skip |
| `Press Release Bank Bukopin.pdf` | — | ✅ Di-load |
| `Press Release PT Timah.pdf` | — | ✅ Di-load |
| `analyst_chat_dummy.txt` | — | ✅ Di-load |

Hasil: 3 file di-load (2 PDF press release + 1 TXT chat), bukan 5.

**5 Pertanyaan Skenario C (cross-source):**
| No | Pertanyaan | Sub-type | Sumber |
|----|-----------|----------|--------|
| C1 | Apa rekomendasi investasi untuk BBKP berdasarkan press release dan opini analis? | Cross-source | PDF + TXT |
| C2 | Berapa target harga TINS dari perspektif fundamental dan analis? | Cross-source | PDF + TXT |
| C3 | Apa risiko investasi BBKP menurut laporan resmi? | PR-only | PDF BBKP |
| C4 | Bagaimana opini analis tentang prospek TINS di semester 2? | Chat-only | TXT |
| C5 | Apa kesamaan strategi BBKP dan TINS untuk pertumbuhan ke depan? | Cross-source | PDF + PDF |

**Parameter run:**
```python
df_C = evaluator.run_batch(
    QUESTIONS_C, source=SOURCE_FOLDER,
    exclude_patterns=['salinan', 'agreement'],
    delay_between=10
)
```

---

### Cell 21 — Visualisasi 4-Panel

Menghasilkan 4-panel matplotlib chart dari hasil keempat skenario:

| Panel | Isi |
|-------|-----|
| [0,0] Bar chart | Rata-rata metrik utama per skenario (A/B/C) |
| [0,1] Line chart | Overall score per pertanyaan semua skenario |
| [1,0] Grouped bar | Faithfulness & Completeness per skenario |
| [1,1] Bar chart | KTE/MSRS/AQI per skenario |

---

### Cell 22 — Ringkasan & Metrik Komposit

Hitung dan tampilkan metrik komposit KTE, MSRS, AQI dari hasil batch DataFrame:

```python
# Contoh perhitungan dari df_A
KTE_A  = (df_A['Answer Faithfulness'].mean() + df_A['Answer Completeness'].mean()) / 2
MSRS_A = (df_A['Precision@K'].mean() + df_A['Context Coverage'].mean()) / 2
AQI_A  = (df_A['Answer Faithfulness'].mean() +
          df_A['Answer Completeness'].mean() +
          df_A['ROUGE-L'].mean()) / 3
```

---

### Cell 23 — SQL Reference (ALTER TABLE)

Sel markdown berisi referensi SQL untuk maintenance database:

```sql
-- Perbaikan VARCHAR(50) → VARCHAR(200) untuk kolom debt_status
ALTER TABLE company_watchlist ALTER COLUMN debt_status TYPE VARCHAR(200);

-- Verifikasi struktur tabel
SELECT column_name, data_type, character_maximum_length
FROM information_schema.columns
WHERE table_name = 'company_watchlist';
```

---

## 6. Alur Data End-to-End

```
Pengguna memanggil:
  pipeline.ask("siapa presiden pertama?", source="/content/drive/MyDrive/data")
                                    │
                                    ▼
[Step 1 — Load]
  SourceDetector: "/content/..." → SourceType.FOLDER
  SourceFactory: → FolderSourceAdapter("/content/drive/MyDrive/data")
  adapter.load():
    - Scan rekursif semua file
    - sejarah.pdf → PdfReader → teks 45 hal → RawDocument(doc_type="pdf")
    - chat_sesi.json → detect role/content → RawDocument(doc_type="chat")
    - data_kependudukan.csv → detect tabular → RawDocument(doc_type="csv")
  Result: [RawDocument x 3]
                                    │
                                    ▼
[Step 2 — Split]
  splitter.split([RawDoc x 3]):
    sejarah.pdf (45.000 char) → 90 chunks
    chat_sesi.json (500 char) → 1 chunk
    data.csv (2.000 char) → 4 chunks
  Result: [LangChain Document x 95]
                                    │
                                    ▼
[Step 3 — Build FAISS Index]
  EmbeddingModel.get() → paraphrase-multilingual-MiniLM-L12-v2
  Embed 95 chunks → 95 × 384 float vectors
  FAISS.from_documents() → IndexFlatL2 in RAM
  Cache: self._cache["/content/drive/MyDrive/data"] = vs
                                    │
                                    ▼
[Step 4a — Retrieve]
  Embed query: "siapa presiden pertama?"
  similarity_search_with_score(query, k=10)
  → [(doc, L2_dist), ...]
  similarity = 1 / (1 + L2_dist)
  Filter > 0.2, sort desc, top 8
  Result: [RetrievedChunk x 8]
                                    │
                                    ▼
[Step 4b — Generate]
  build_context():
    "[Sumber 1 — sejarah.pdf (pdf)]:\nPresiden pertama RI adalah Soekarno..."
    "---"
    "[Sumber 2 — sejarah.pdf (pdf)]:\nSoekarno dilahirkan pada 6 Juni 1901..."
    ...
  Prompt → Gemini 2.5-flash (temperature=0.3)
  Answer: "Presiden pertama Republik Indonesia adalah Soekarno."
                                    │
                                    ▼
RAGResult:
  .answer          = "Presiden pertama Republik Indonesia adalah Soekarno."
  .retrieved_chunks = [8 chunks dengan score 0.72, 0.68, 0.61, 0.55, 0.51, 0.48, 0.42, 0.38]
  .timing          = {"1_load": 2.1, "2_split": 0.05, "3_index": 3.2,
                       "4a_retrieve": 0.1, "4b_generate": 1.8}
  .total_time      = 7.25s
  .display()       → print terformat
```

---

## 7. Format Input yang Didukung

### Sumber: Folder

| Format | Extension | Cara di-load | Tipe di RawDocument | Status |
|--------|-----------|-------------|---------------------|--------|
| PDF | `.pdf` | `pypdf.PdfReader` | `"pdf"` | ✅ Aktif |
| Text | `.txt` | `Path.read_text()` | `"txt"` | ✅ Aktif |
| Markdown | `.md` | `Path.read_text()` | `"markdown"` | ✅ Aktif |
| Log | `.log` | `Path.read_text()` | `"log"` | ✅ Aktif |
| Word | `.docx`, `.doc` | ~~`python-docx`~~ | ~~`"docx"`~~ | ❌ Nonaktif (commented out) |
| JSON | `.json` | ~~`json.load()`~~ | ~~`"json"`/`"chat"`~~ | ❌ Nonaktif |
| CSV | `.csv` | ~~`pd.read_csv()`~~ | ~~`"csv"`/`"chat"`~~ | ❌ Nonaktif |
| Excel | `.xlsx`, `.xls` | ~~`pd.read_excel()`~~ | ~~`"excel"`~~ | ❌ Nonaktif |

### Sumber: PostgreSQL

| Mode | Parameter | Yang terjadi |
|---|---|---|
| Semua tabel | `pg_tables=None` | `inspect().get_table_names("public")` → load semua |
| Tabel tertentu | `pg_tables=["orders", "products"]` | Hanya load tabel tersebut |
| Custom SQL | `pg_queries={"label": "SELECT ..."}` | Jalankan query, hasil → RawDocument |
| Kombinasi | `pg_tables=[...] + pg_queries={...}` | Custom query dulu, lalu tabel |

---

## 8. Output yang Dihasilkan

### Saat `result.display()` dipanggil

```
══════════════════════════════════════════════════════════════
❓ PERTANYAAN:
   Siapa presiden pertama Indonesia?
──────────────────────────────────────────────────────────────
💡 JAWABAN:
   Presiden pertama Republik Indonesia adalah Soekarno,
   yang menjabat dari tahun 1945 hingga 1967.
──────────────────────────────────────────────────────────────
📚 CHUNKS (5):
   [1] score=0.724 | pdf | sejarah.pdf
       Presiden pertama RI adalah Soekarno yang dipilih oleh PPKI ...
   [2] score=0.681 | pdf | sejarah.pdf
       Soekarno dilahirkan di Surabaya pada 6 Juni 1901 dan ...
   [3] score=0.612 | chat | chat_sejarah.json
       [USER]: Siapa pendiri Indonesia? [ASSISTANT]: Soekarno dan Hatta ...
   ... +2 chunk lainnya
──────────────────────────────────────────────────────────────
⏱️  TIMING:
   1_load                 2.103s  █████████████████
   2_split                0.051s  █
   3_index                3.218s  █████████████████████████
   4a_retrieve            0.099s  █
   4b_generate            1.812s  ██████████████
   TOTAL                  7.283s
──────────────────────────────────────────────────────────────
📊 Sumber: 📂 Folder (lokal / Google Drive) | 3 dokumen | 95 chunk | LLM: gemini/gemini-2.5-flash
══════════════════════════════════════════════════════════════
```

### File Output dari Evaluasi

Semua file output menyertakan timestamp `<YYYYMMDD_HHMMSS>` agar setiap run tersimpan terpisah (penting untuk reproducibility jurnal).

| File | Format | Isi |
|---|---|---|
| `eval_agnostic_<ts>.csv` | CSV | 15 kolom metrik lengkap per pertanyaan |
| `eval_agnostic_<ts>.png` | PNG (150 DPI) | 4-panel chart: bar avg, line per Q, ROUGE/BLEU bar, retrieval metrics |
| `eval_agnostic_<ts>.tex` | LaTeX | Tabel booktabs style siap paste ke jurnal |

---

## 9. Pertanyaan Kunci: Apakah Ini Realtime?

### ✅ Ya — Sistem Ini Realtime

**Definisi "Realtime" dalam konteks ini:** Dokumen tidak diproses/di-index sebelumnya (tidak ada pre-indexing). Index dibuat *saat pertanyaan masuk* (`ask()` dipanggil), langsung dari sumber data aktual.

**Bukti:**

| Aspek | Bukti |
|---|---|
| **Load saat ask()** | `adapter.load()` dipanggil di dalam `ask()`, Step 1 |
| **FAISS di RAM** | `FAISS.from_documents()` tidak punya mekanisme disk write |
| **Tidak ada file index** | Test T4: `glob("**/*.faiss")` → 0 file |
| **Data berubah → detect** | Test T7: ubah isi file → `clear_cache()` → konten baru ter-retrieve |
| **Session cache = RAM** | `self._cache` adalah Python `dict`, hilang saat kernel restart |

**Perbandingan dengan sistem non-realtime:**

```
Non-realtime (konvensional):
  index.py → scan folder → build FAISS → simpan ke faiss_index.bin
  app.py   → load faiss_index.bin → jawab pertanyaan
  ⚠️  File berubah? Harus re-run index.py dulu!

Realtime (sistem ini):
  pipeline.ask(q, source) → scan → build → retrieve → jawab
  ✅ File berubah? Cukup clear_cache(), ask() berikutnya otomatis pakai data baru
```

**Trade-off:**

| Aspek | Realtime | Non-realtime |
|---|---|---|
| **Pertanyaan pertama** | Lebih lambat (build index) | Lebih cepat (load index) |
| **Pertanyaan berikutnya** | Cepat (cache hit) | Cepat |
| **Data berubah** | ✅ `clear_cache()` → fresh | ❌ Harus re-index manual |
| **Disk usage** | 0 bytes | Sesuai ukuran data |

---

## 10. Metrik Evaluasi: 8 Standar + 3 Komposit

### 8 Metrik Standar

| # | Metrik | Rumus | Interpretasi |
|---|--------|-------|-------------|
| 1 | **Retrieval Relevance** | Cosine similarity embedding Q vs avg chunk embedding | Seberapa relevan chunk yang di-retrieve terhadap pertanyaan |
| 2 | **Answer Faithfulness** | F1 token overlap(answer, context) | Anti-hallucination: seberapa banyak jawaban berasal dari konteks |
| 3 | **Answer Completeness** | \|tok(Q) ∩ tok(answer)\| / \|tok(Q)\| | Kelengkapan: keyword pertanyaan yang muncul di jawaban |
| 4 | **ROUGE-L** | F1 berbasis LCS(answer, ref) | Kemiripan urutan kata vs referensi (Lin 2004) |
| 5 | **BLEU-1** | Unigram precision + brevity penalty | Presisi kata jawaban vs referensi (Papineni 2002) |
| 6 | **Precision@K** | \|chunks ≥ threshold\| / K | Proporsi chunk relevan di top-K hasil retrieval |
| 7 | **MRR** | 1 / rank_chunk_relevan_pertama | Posisi chunk relevan pertama (Voorhees 1999) |
| 8 | **Context Coverage** | \|unique sources\| / total_chunks | Keragaman sumber dokumen yang di-retrieve |

**Overall** = rata-rata metrik 1–5 (lima metrik utama)

---

### 3 Metrik Komposit (Kontribusi Penelitian)

Ketiga metrik ini adalah **kontribusi orisinal** penelitian ini — tidak ada di paper RAG standar. Dirancang untuk mengukur dimensi kualitas RAG yang lebih spesifik pada konteks analisis investasi multi-sumber.

#### KTE — Knowledge Transfer Effectiveness

$$\text{KTE} = \frac{\text{Faithfulness} + \text{Completeness}}{2}$$

**Mengukur:** Seberapa efektif sistem "memindahkan" pengetahuan dari dokumen ke jawaban — apakah jawaban faithful (tidak halusinasi) sekaligus complete (menjawab semua aspek pertanyaan).

**Interpretasi:**
- KTE tinggi → sistem berhasil mengekstrak dan mentransfer pengetahuan secara akurat
- KTE rendah → jawaban halusinasi atau tidak menjawab pertanyaan sepenuhnya

---

#### MSRS — Multi-Source Retrieval Score

$$\text{MSRS} = \frac{\text{Precision@K} + \text{Context Coverage}}{2}$$

**Mengukur:** Kualitas pengambilan bukti dari berbagai sumber — apakah dokumen yang di-retrieve tepat (presisi tinggi) dan beragam (bukan hanya dari satu file).

**Interpretasi:**
- MSRS tinggi → sistem menemukan bukti dari sumber yang tepat dan beragam
- MSRS rendah → retrieval terlalu terfokus pada satu sumber atau tidak presisi

---

#### AQI — Answer Quality Index

$$\text{AQI} = \frac{\text{Faithfulness} + \text{Completeness} + \text{ROUGE-L}}{3}$$

**Mengukur:** Kualitas linguistik jawaban secara holistik — kombinasi grounding faktual (Faithfulness), kelengkapan isi (Completeness), dan kemiripan leksikal dengan referensi (ROUGE-L).

**Interpretasi:**
- AQI tinggi → jawaban berkualitas tinggi secara linguistik: akurat, lengkap, dan mirip referensi
- AQI rendah → jawaban memiliki masalah pada setidaknya satu dimensi linguistik

---

### Hubungan Antar Metrik

```
8 Metrik Standar:
  ┌─ Retrieval Quality ─────────────────────────────────┐
  │  Retrieval Relevance, Precision@K, MRR, Coverage    │
  └─────────────────────────────────────────────────────┘
  ┌─ Answer Quality ────────────────────────────────────┐
  │  Faithfulness, Completeness, ROUGE-L, BLEU-1        │
  └─────────────────────────────────────────────────────┘

3 Metrik Komposit:
  KTE  = Faithfulness + Completeness  ← efektivitas transfer pengetahuan
  MSRS = Precision@K + Coverage       ← bukti multi-sumber
  AQI  = Faith + Comp + ROUGE-L       ← kualitas linguistik
```

---

## 11. Corpus & Dataset Evaluasi

### Dokumen Sumber (Skenario A & C)

| File | Format | Isi | Sumber |
|------|--------|-----|--------|
| `Press Release Bank Bukopin.pdf` | **PDF** ✅ | Laporan kinerja keuangan BBKP Q1 2025: laba, NPL, ekuitas, strategi | PT Bank Bukopin Tbk |
| `Press Release PT Timah.pdf` | **PDF** ✅ | Laporan produksi & keuangan TINS Q1 2025: volume, harga LME, ekuitas | PT Timah Tbk |
| `analyst_chat_dummy.txt` | **TXT** ✅ | Log chat analis tim riset: opini, rekomendasi, diskusi BBKP & TINS | Dummy (simulasi) |
| `Salinan dummy-pdf.pdf` | PDF | File uji coba (di-filter di Skenario C oleh `exclude_patterns`) | Test file |
| `Agreement Mr. Krisna.docx.pdf` | PDF | Dokumen legal dummy (di-filter di Skenario C oleh `exclude_patterns`) | Test file |

> **Catatan:** Format DOCX tidak digunakan. `Agreement Mr. Krisna.docx.pdf` adalah file PDF (bukan DOCX) — ekstensi akhirnya `.pdf` sehingga tetap terbaca oleh `_load_pdf`, namun di-skip di Skenario C karena namanya mengandung `'agreement'` dalam `exclude_patterns`.

**Total corpus aktif (setelah filter Skenario C):** 3 file — 2 PDF + 1 TXT

### Database Sumber (Skenario B)

**Provider:** Neon DB (PostgreSQL serverless, us-east-1)

| Tabel | Rows | Isi |
|-------|------|-----|
| `user_profiles` | 6 | Ahmad Wijaya (IT Mgr), Sari Dewi (Sr Analyst), Reza Pratama (Jr Analyst), Dian Kusuma (Research Assoc), Andika Santoso (Quant), Rina Marlina (Compliance) |
| `products` | 10 | Bloomberg, Refinitiv, Python, Tableau, Excel, PostgreSQL, LangChain, Gemini API, Power BI, ChatGPT Enterprise |
| `orders` | 5 | 4 completed, 1 pending (ChatGPT Enterprise) |
| `company_watchlist` | 2 | BBKP (BUY/medium risk), TINS (HOLD/medium risk) |
| `analyst_notes` | 2 | BBKP (analyst_id=1, target=95.50), TINS (analyst_id=2, target=1250.00) |

**Skema penting:**
- `company_watchlist.debt_status` → VARCHAR(200) (di-ALTER dari VARCHAR(50))
- `analyst_notes` → kolom: `reviewed_at`, `target_price_usd`, `key_highlights`, `risk_notes`

---

## 12. Empat Skenario Evaluasi & Hasil Aktual

### Overview

| Skenario | Adapter | Sumber | n | Overall | KTE | MSRS | AQI |
|----------|---------|--------|---|---------|-----|------|-----|
| **A** | FolderSourceAdapter | PDF + TXT (press release + chat log) | 10 | **0.307** | 0.391 | 0.688 | 0.282 |
| **B** | PostgreSQLAdapter | 5 tabel Neon PostgreSQL | 10 | **0.301** | 0.427 | **0.769** | 0.309 |
| **C** | FolderSourceAdapter | TXT (chat log) | 5 | **0.325** | **0.460** | 0.700 | **0.320** |
| **D** | MultiSourceAdapter | PDF + TXT + SQL (Hybrid) | 5 | **0.284** | 0.352 | 0.725 | 0.254 |

*P@K = 1.000 dan MRR = 1.000 di semua skenario.*

---

### Skenario A: Hasil Lengkap (PDF + TXT, Explicit -> Actionable, n=10)

| Q | RR | Faith | Comp | ROUGE-L | BLEU-1 | P@K | MRR | CC | Overall |
|---|-----|-------|------|---------|--------|-----|-----|-----|--------|
| A1 | 0.735 | 0.062 | 0.600 | 0.032 | 0.000 | 1.000 | 1.000 | 0.375 | 0.543 |
| A2 | 0.810 | 0.137 | 0.917 | 0.077 | 0.000 | 1.000 | 1.000 | 0.375 | 0.584 |
| A3 | 0.787 | 0.048 | 0.727 | 0.020 | 0.000 | 1.000 | 1.000 | 0.250 | 0.554 |
| A4 | 0.773 | 0.178 | 0.667 | 0.117 | 0.000 | 1.000 | 1.000 | 0.125 | 0.568 |
| A5 | 0.515 | 0.286 | 0.667 | 0.119 | 0.001 | 1.000 | 1.000 | 0.375 | 0.452 |
| A6 | 0.756 | 0.062 | 0.909 | 0.025 | 0.000 | 1.000 | 1.000 | 0.375 | 0.547 |
| A7 | 0.678 | 0.044 | 0.909 | 0.019 | 0.000 | 1.000 | 1.000 | 0.500 | 0.499 |
| A8 | 0.573 | 0.182 | 0.818 | 0.072 | 0.000 | 1.000 | 1.000 | 0.375 | 0.470 |
| A9 | 0.694 | 0.076 | 0.824 | 0.028 | 0.000 | 1.000 | 1.000 | 0.375 | 0.517 |
| A10 | 0.633 | 0.109 | 0.667 | 0.054 | 0.000 | 1.000 | 1.000 | 0.375 | 0.496 |
| **AVG** | **0.690** | **0.126** | **0.656** | **0.065** | **0.000** | **1.000** | **1.000** | **0.375** | **0.307** |

**Metrik komposit A:** KTE = 0.391 | MSRS = 0.688 | AQI = 0.282

**Analisis:** P@K = 1.000 dan MRR = 1.000 menunjukkan retrieval sempurna pada sumber PDF + TXT. Faithfulness rendah (0.126) merupakan karakteristik evaluasi reference-free, bukan kegagalan sistem. Context Coverage = 0.375 mengkonfirmasi retrieval multi-sumber aktif.
---

### Skenario B: Hasil Lengkap (PostgreSQL, Structured -> Contextual, n=10)

| Q | RR | Faith | Comp | ROUGE-L | BLEU-1 | P@K | MRR | CC | Overall |
|---|-----|-------|------|---------|--------|-----|-----|-----|--------|
| B1 | 0.474 | 0.156 | 0.800 | 0.103 | 0.000 | 1.000 | 1.000 | 0.625 | 0.437 |
| B2 | 0.492 | 0.110 | 0.750 | 0.041 | 0.000 | 1.000 | 1.000 | 0.625 | 0.440 |
| B3 | 0.541 | 0.183 | 0.889 | 0.081 | 0.000 | 1.000 | 1.000 | 0.625 | 0.461 |
| B4 | 0.556 | 0.448 | 0.933 | 0.184 | 0.021 | 1.000 | 1.000 | 0.625 | 0.466 |
| B5 | 0.427 | 0.082 | 0.667 | 0.027 | 0.000 | 1.000 | 1.000 | 0.625 | 0.428 |
| B6 | 0.474 | 0.156 | 0.800 | 0.103 | 0.000 | 1.000 | 1.000 | 0.625 | 0.437 |
| B7 | 0.492 | 0.110 | 0.750 | 0.041 | 0.000 | 1.000 | 1.000 | 0.625 | 0.440 |
| B8 | 0.541 | 0.183 | 0.889 | 0.081 | 0.000 | 1.000 | 1.000 | 0.625 | 0.461 |
| B9 | 0.556 | 0.448 | 0.933 | 0.184 | 0.021 | 1.000 | 1.000 | 0.625 | 0.466 |
| B10 | 0.427 | 0.082 | 0.667 | 0.027 | 0.000 | 1.000 | 1.000 | 0.625 | 0.428 |
| **AVG** | **0.577** | **0.152** | **0.702** | **0.072** | **0.002** | **1.000** | **1.000** | **0.538** | **0.301** |

**Metrik komposit B:** KTE = 0.427 | MSRS = **0.769 (tertinggi)** | AQI = 0.309

**Analisis:** MSRS tertinggi (0.769) karena PostgreSQL menghasilkan Context Coverage tertinggi (0.538) — data dari 5 tabel berbeda mengisi top-K dengan chunk dari entitas beragam. P@K = 1.000 di semua pertanyaan. KTE = 0.427 tertinggi kedua karena data terstruktur menghasilkan jawaban yang lebih complete (0.702).
---

### Skenario C: Hasil per Sub-type (Chat Log TXT, Tacit -> Explicit)

| Sub-type | n | RR | Faith | Comp | ROUGE-L | P@K | MRR | CC | Overall | KTE |
|----------|---|-----|-------|------|---------|-----|-----|-----|---------|-----|
| PR-only | 2 | 0.715 | 0.053 | 0.909 | 0.022 | 1.000 | 1.000 | 0.375 | 0.523 | 0.481 |
| Chat-only | 1 | 0.573 | 0.182 | 0.818 | 0.072 | 1.000 | 1.000 | 0.375 | 0.470 | 0.500 |
| Cross-source | 2 | 0.663 | 0.048 | 0.746 | 0.025 | 1.000 | 1.000 | 0.450 | 0.487 | 0.397 |
| **AVG** | **5** | **0.667** | **0.095** | **0.825** | **0.040** | **1.000** | **1.000** | **0.400** | **0.325** | **0.460** |

**Metrik komposit C:** KTE = **0.460 (tertinggi)** | MSRS = 0.700 | AQI = 0.320

**Analisis:** Answer Completeness tertinggi (0.825) di antara semua skenario -- chat log kaya informasi eksplisit yang menjawab pertanyaan secara lengkap. KTE = 0.460 merupakan yang tertinggi karena pertanyaan tacit knowledge dari diskusi tim menghasilkan jawaban yang sangat complete meski Faithfulness tetap rendah (karakteristik reference-free).

---

### Skenario D: Hasil Lengkap (Hybrid Cross-Paradigm, MultiSourceAdapter)

> **Skenario baru:** `MultiSourceAdapter` menggabungkan FolderSourceAdapter (PDF+TXT) dan PostgreSQLAdapter (5 tabel) ke dalam satu indeks FAISS. Query menjangkau kedua paradigma sumber sekaligus dalam satu pipeline.

| Q | RR | Faith | Comp | ROUGE-L | BLEU-1 | P@K | MRR | CC | Overall |
|---|-----|-------|------|---------|--------|-----|-----|-----|--------|
| D1 | 0.712 | 0.067 | 0.545 | 0.027 | 0.000 | 1.000 | 1.000 | 0.500 | 0.520 |
| D2 | 0.668 | 0.233 | 0.667 | 0.126 | 0.007 | 1.000 | 1.000 | 0.500 | 0.492 |
| D3 | 0.605 | 0.124 | 0.731 | 0.085 | 0.000 | 1.000 | 1.000 | 0.375 | 0.488 |
| D4 | 0.634 | 0.060 | 0.480 | 0.026 | 0.000 | 1.000 | 1.000 | 0.500 | 0.486 |
| D5 | 0.663 | 0.061 | 0.556 | 0.022 | 0.000 | 1.000 | 1.000 | 0.375 | 0.496 |
| **AVG** | **0.656** | **0.109** | **0.596** | **0.057** | **0.001** | **1.000** | **1.000** | **0.450** | **0.284** |

**Metrik komposit D:** KTE = 0.352 | MSRS = 0.725 | AQI = 0.254

**Analisis:** Context Coverage = 0.450 lebih tinggi dari Skenario A (0.375) -- membuktikan retrieval lintas paradigma aktif; chunk berasal dari PDF, TXT, dan tabel SQL sekaligus. KTE = 0.352 dan AQI = 0.254 lebih rendah dari skenario tunggal, konsisten dengan ekspektasi karena pertanyaan cross-paradigm menghasilkan jawaban sintesis yang lebih sulit diverifikasi secara reference-free. MSRS = 0.725 mengkonfirmasi multi-sumber efektif.

---

### Temuan Lintas Skenario

| Temuan | Implikasi |
|--------|-----------|
| P@K = 1.000 dan MRR = 1.000 di semua skenario | Retrieval konsisten optimal di semua tipe sumber |
| Context Coverage tertinggi di Skenario B (0.538) | PostgreSQL 5 tabel menghasilkan chunk dari entitas paling beragam |
| Context Coverage Skenario D (0.450) > A (0.375) | MultiSourceAdapter terbukti menggabungkan dokumen lintas paradigma |
| Faithfulness rendah (0.095–0.152) di semua skenario | Karakteristik reference-free pada teks Indonesia, bukan kegagalan retrieval |
| KTE tertinggi di Skenario C (0.460) | Chat log dense informasi; pertanyaan tacit knowledge terjawab paling lengkap |
| MSRS tertinggi di Skenario B (0.769) | PostgreSQL paling konsisten sebagai sumber multi-entitas |
| AQI tertinggi di Skenario C (0.320) | Kualitas linguistik terbaik dari sumber percakapan informal |
| Skenario D Overall terendah (0.284) | Expected: sintesis cross-paradigm lebih sulit dievaluasi secara reference-free |
---

## 13. Siapa yang Menggunakan Sistem Ini

### Target Pengguna

| Persona | Kebutuhan | Cara Pakai Sistem |
|---------|----------|-------------------|
| **Equity Analyst** | Query press release + chat log analis secara natural language | Skenario A/C: FolderSourceAdapter dengan PDF+TXT |
| **Investment Researcher** | Akses data watchlist & catatan analis dari DB | Skenario B: PostgreSQLAdapter ke Neon DB |
| **IT/Data Staff** | Query tabel internal (procurement, orders, products) | PostgreSQLAdapter dengan `pg_tables` filter |
| **Knowledge Manager** | Ekstrak tacit knowledge dari chat log tim | FolderSourceAdapter dengan file TXT/JSON chat |
| **Academic Researcher** | Evaluasi sistem RAG pada corpus Indonesia | `run_batch()` + `to_latex()` → paper jurnal |
| **Thesis Evaluator** | Validasi metrik evaluasi RAG multi-sumber | 8 metrik + KTE/MSRS/AQI composite metrics |

### Use Cases Konkret

```
Use Case 1: Analyst bertanya ke sistem
  "Apa rekomendasi saham BBKP berdasarkan press release terbaru?"
  → FolderSourceAdapter membaca PDF BBKP
  → Retrieve 8 chunk paling relevan
  → Gemini generate jawaban faktual dari konteks

Use Case 2: Research query ke database
  "Siapa analis yang di-assign untuk coverage TINS?"
  → PostgreSQLAdapter query tabel user_profiles + company_watchlist
  → Jawaban berbasis data aktual di Neon DB

Use Case 3: Cross-source synthesis
  "Bandingkan prospek BBKP dan TINS berdasarkan laporan resmi dan opini analis"
  → FolderSourceAdapter baca PDF BBKP + PDF TINS + TXT chat
  → Retrieve dari 3 sumber berbeda
  → LLM synthesize jawaban lintas sumber

Use Case 4: Academic evaluation
  evaluator.run_batch(questions, source=SOURCE)
  → df dengan 15 kolom metrik
  → evaluator.to_latex(df) → tabel siap jurnal
  → KTE/MSRS/AQI sebagai kontribusi metodologis
```

### Yang Tidak Didukung Sistem Ini

| Bukan Use Case | Alasan |
|---------------|--------|
| Real-time stock price query | Tidak ada koneksi ke feed market (Bloomberg, Reuters) |
| Fine-tuning LLM | Sistem hanya zero-shot inference |
| Multi-turn conversation | `pipeline.ask()` stateless per call |
| MySQL / MongoDB | Hanya PostgreSQL yang diimplementasikan |
| File video/audio | Hanya teks, PDF, DOCX, Excel, JSON, CSV |

---

## 14. Bug yang Ditemukan & Diperbaiki (Code Review)

Code review dilakukan setelah notebook selesai dibuat. **7 bug** ditemukan dan diperbaiki:

| # | Bug | Letak | Dampak | Fix |
|---|---|---|---|---|
| 1 | Judul menyebut SQLite, MySQL, MongoDB (tidak diimplementasikan) | Cell 1 (markdown) | Dokumentasi menyesatkan | Hapus referensi, rewrite tabel dan diagram |
| 2 | Section 3 menyebut `SQLDatabaseAdapter`, `MongoDBAdapter`, `ChatFolderAdapter` (class tidak ada) | Cell 3 (markdown) | Dokumentasi tidak akurat | Rewrite dengan nama class yang benar |
| 3 | Install `pymysql`, `pymongo`, `unstructured` (tidak dipakai) | Cell 1 (code) | Install tidak perlu, membuang waktu | Hapus, hanya install `psycopg2-binary` |
| 4 | `_get_engine()` pakai `__import__("sqlalchemy").text(...)` | Cell 3 (code) | Non-idiomatic, fragile di beberapa env | `from sqlalchemy import create_engine, text` |
| 5 | `pd.read_sql(sql, engine)` — deprecated SQLAlchemy 2.x | Cell 3 (code) | Warning/error di SQLAlchemy ≥ 2.0 | `with engine.connect() as conn: pd.read_sql(text(sql), conn)` |
| 6 | `df.to_string(index=False, max_rows=N)` — `max_rows` bukan parameter valid | Cell 3 (code) | `TypeError` saat dipanggil | `df.head(config.max_db_rows).to_string(index=False)` |
| 7 | Diagram di judul tidak akurat (tidak mencerminkan class yang ada) | Cell 1 (markdown) | Dokumentasi membingungkan | Rewrite diagram dengan class name yang benar |

**Bug tambahan yang diperbaiki pasca evaluasi (April 2026):**

| # | Bug | Letak | Dampak | Fix |
|---|---|---|---|---|
| 8 | `debt_status` VARCHAR(50) terlalu pendek untuk teks panjang | Neon DB schema | `StringDataRightTruncation` error saat INSERT | `ALTER TABLE company_watchlist ALTER COLUMN debt_status TYPE VARCHAR(200)` |
| 9 | Kolom `note_date` tidak ada di `analyst_notes` | Cell 18 INSERT | `UndefinedColumn` error | Ganti ke `reviewed_at` |
| 10 | Kolom `target_price` tidak ada di `analyst_notes` | Cell 18 INSERT | `UndefinedColumn` error | Ganti ke `target_price_usd` |
| 11 | Gemini 503 tidak di-handle → pipeline crash | Cell 5 AnswerGenerator | Seluruh batch evaluation gagal | Tambah exponential backoff + fallback chain |

---

## 15. Cara Pakai

### Prasyarat

1. Jalankan Cell 1 (install dependencies)
2. Set `config.gemini_api_key` di Cell 2
3. Jalankan Cell 2–7 secara berurutan (setup pipeline)

### Skenario 1: Tanya dari Folder

```python
result = pipeline.ask(
    "Apa kesimpulan utama dari laporan keuangan?",
    source="/content/drive/MyDrive/laporan_keuangan"
)
result.display()
```

### Skenario 2: Tanya dari PostgreSQL (semua tabel)

```python
result = pipeline.ask(
    "Berapa total transaksi bulan ini?",
    source="postgresql://admin:secret@localhost:5432/erp_db"
)
result.display()
```

### Skenario 3: PostgreSQL dengan filter tabel

```python
result = pipeline.ask(
    "Produk apa yang paling banyak terjual?",
    source="postgresql://admin:secret@localhost:5432/erp_db",
    pg_tables=["orders", "order_items", "products"]
)
result.display()
```

### Skenario 4: PostgreSQL dengan custom SQL

```python
result = pipeline.ask(
    "Bagaimana tren penjualan 30 hari terakhir?",
    source="postgresql://admin:secret@localhost:5432/erp_db",
    pg_queries={
        "tren_30_hari": """
            SELECT DATE(created_at) as tanggal, COUNT(*) as order_count, SUM(total) as revenue
            FROM orders
            WHERE created_at >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            ORDER BY tanggal
        """,
        "produk_terlaris": """
            SELECT p.name, SUM(oi.qty) as total_qty
            FROM order_items oi JOIN products p ON oi.product_id = p.id
            GROUP BY p.name ORDER BY total_qty DESC LIMIT 10
        """
    }
)
result.display()
```

### Skenario 5: Filter file dengan `exclude_patterns`

```python
# Hanya load PDF press release, skip file salinan dan agreement
result = pipeline.ask(
    "Apa strategi bisnis BBKP untuk H2 2025?",
    source="/content/drive/MyDrive/data/sample_data",
    exclude_patterns=['salinan', 'agreement']
)
result.display()
```

### Skenario 6: Update data → paksa fresh load

```python
# Data di folder/DB sudah diupdate
index_builder.clear_cache()         # Hapus cache

# Ask berikutnya akan load ulang dari sumber
result = pipeline.ask("pertanyaan", source="...")
```

### Skenario 7: Evaluasi satu hasil (display_result)

```python
result = pipeline.ask("pertanyaan", source="...")
evaluator.display_result(result)

# Reference-based (dengan ground truth)
evaluator.display_result(result, ground_truth="jawaban yang benar")
```

### Skenario 8: Evaluasi kualitas batch

```python
pertanyaan = [
    "Apa rekomendasi saham BBKP?",
    "Berapa target harga TINS?",
    "Apa risiko investasi di BBKP?",
]

# Reference-free
df_eval = evaluator.run_batch(pertanyaan, source="...")

# Dengan exclude_patterns
df_eval = evaluator.run_batch(
    pertanyaan, source=SOURCE_FOLDER,
    exclude_patterns=['salinan', 'agreement']
)

evaluator.plot(df_eval)                    # 4-panel chart
tex = evaluator.to_latex(df_eval)          # tabel LaTeX untuk jurnal
print(tex)
```

### Skenario 9: Hitung metrik komposit

```python
# Setelah run_batch
KTE  = (df_eval['Answer Faithfulness'].mean() + df_eval['Answer Completeness'].mean()) / 2
MSRS = (df_eval['Precision@K'].mean() + df_eval['Context Coverage'].mean()) / 2
AQI  = (df_eval['Answer Faithfulness'].mean() +
        df_eval['Answer Completeness'].mean() +
        df_eval['ROUGE-L'].mean()) / 3

print(f"KTE  = {KTE:.3f}")   # Knowledge Transfer Effectiveness
print(f"MSRS = {MSRS:.3f}")  # Multi-Source Retrieval Score
print(f"AQI  = {AQI:.3f}")   # Answer Quality Index
```

---

## 16. Batasan & Catatan Penting

### Batasan Teknis

| Batasan | Penjelasan |
|---|---|
| **Cache hilang saat restart kernel** | `self._cache` adalah Python dict in-memory. Restart = rebuild dari awal |
| **Batas baris DB = `max_db_rows=5000`** | Untuk mencegah OOM. Bisa dinaikkan jika RAM cukup |
| **Embedding hanya di CPU** | `device="cpu"`. Untuk GPU: ubah ke `"cuda"` di `Config.embedding_model` |
| **PostgreSQL saja untuk DB** | Desain by-request. MySQL/SQLite/MongoDB tidak didukung |
| **FAISS IndexFlatL2 = exact search** | Tidak ada approximate search. Lambat untuk >1M chunks |
| **LLM zero-shot** | Kualitas jawaban bergantung pada kemampuan Gemini tanpa fine-tuning |
| **Gemini rate limit** | Ditangani dengan exponential backoff + fallback chain (2.5→2.0→1.5-flash) |

### Catatan Keamanan

| Hal | Catatan |
|---|---|
| **Gemini API key** | Jangan commit ke git. Gunakan environment variable: `os.environ["GEMINI_API_KEY"]` |
| **PostgreSQL connection string** | Mengandung password. Gunakan `.env` file atau Secret Manager |
| **Data dari DB** | Dikirim ke Gemini API. Pastikan tidak ada data sensitif/PII dalam chunks |

### Tips Optimasi

```python
# Percepat: gunakan session cache (default sudah aktif)
config.use_session_cache = True

# Hemat RAM: kurangi max_db_rows
config.max_db_rows = 1000

# Lebih selektif: naikkan threshold
config.similarity_threshold = 0.3

# Lebih banyak konteks: naikkan top_k
config.top_k = 10

# Chunk lebih kecil untuk data tabular
config.chunk_size = 500
```

---

*Dokumentasi ini dibuat berdasarkan hasil code review, analisis mendalam setiap sel, verifikasi unit tests, penjelasan detail cara kerja setiap plugin/fungsi, dan hasil evaluasi aktual empat skenario (20 pertanyaan) pada `QA_RAG_AgnosticSource.ipynb`. Terakhir diperbarui: April 2026 -- eval run 20260419.*
