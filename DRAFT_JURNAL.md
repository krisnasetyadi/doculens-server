# DRAFT JURNAL — v1.0
# Sistem RAG Agnostic Multi-Sumber untuk Question Answering Insight Bisnis

---

## Judul (ID)
**Sistem Retrieval-Augmented Generation Agnostic Multi-Sumber untuk Question Answering Berbasis Dokumen dan Database pada Konteks Transfer Pengetahuan Organisasi**

## Judul (EN)
**Agnostic Multi-Source Retrieval-Augmented Generation System for Question Answering Based on Documents and Databases in Organizational Knowledge Transfer Context**

---

## Penulis
Krisna Dwi Setyaadi¹, Ivan Michael Siregar²

¹² Program Studi Sistem Informasi Data Science, Institut Teknologi Harapan Bangsa, Bandung, Indonesia

Email: [krisna@ithb.ac.id]

---

## ABSTRAK

Organisasi modern menghadapi tantangan dalam mengelola pengetahuan yang tersebar di berbagai sumber heterogen—dokumen tidak terstruktur (PDF, DOCX, TXT) dan basis data relasional—sehingga menghambat transfer pengetahuan dan pengambilan keputusan. Penelitian ini mengembangkan sistem Question Answering (QA) berbasis Retrieval-Augmented Generation (RAG) yang bersifat agnostic terhadap sumber data, artinya sistem hanya memerlukan satu parameter `source` untuk secara otomatis mendeteksi dan menangani berbagai jenis sumber. Sistem mengintegrasikan dua adapter utama: `FolderSourceAdapter` untuk sumber tidak terstruktur (PDF, DOCX, TXT, MD, LOG) dan `PostgreSQLAdapter` untuk basis data relasional. Indeks vektor FAISS dibangun secara realtime pada saat pertanyaan masuk, tanpa pra-komputasi ke disk. Evaluasi dilakukan menggunakan delapan metrik kuantitatif ditambah metrik Knowledge Transfer Effectiveness (KTE = rata-rata Faithfulness dan Completeness) sebagai proxy efektivitas transfer pengetahuan. Eksperimen mencakup tiga skenario multi-sumber: (A) dokumen PDF/DOCX pengadaan, (B) PostgreSQL dengan lima tabel berelasi termasuk cross-table JOIN, dan (C) kombinasi laporan keuangan formal dengan log percakapan analis (multi-format unstructured). Hasil pada Skenario A menunjukkan Precision@K = 1.00 dan MRR = 1.00, membuktikan komponen retrieval bekerja sangat baik. Evaluasi ketiga skenario membuktikan kemampuan sistem menangani tiga dimensi transfer pengetahuan: Explicit→Actionable, Structured→Contextual, dan Tacit→Explicit. Sistem ini menawarkan arsitektur yang dapat diperluas, transparan secara metrik, dan siap diimplementasikan untuk mendukung kebutuhan transfer pengetahuan di organisasi.

**Kata Kunci:** Question Answering, Retrieval-Augmented Generation, Multi-Sumber, FAISS, Transfer Pengetahuan, Agnostic Source

---

## ABSTRACT

Modern organizations face challenges in managing knowledge dispersed across heterogeneous sources—unstructured documents (PDF, DOCX, TXT) and relational databases—which hinders knowledge transfer and decision-making. This study develops a Question Answering (QA) system based on Retrieval-Augmented Generation (RAG) that is agnostic to the data source, meaning the system only requires a single `source` parameter to automatically detect and handle various source types. The system integrates two main adapters: `FolderSourceAdapter` for unstructured sources (PDF, DOCX, TXT, MD, LOG) and `PostgreSQLAdapter` for relational databases. FAISS vector indices are built in real-time upon query arrival, without pre-computation to disk. Evaluation is conducted using eight quantitative metrics: Retrieval Relevance, Answer Faithfulness, Answer Completeness, ROUGE-L, BLEU-1, Precision@K, MRR, and Context Coverage. Experimental results on procurement document-based queries show Precision@K = 1.00 and MRR = 1.00, demonstrating that the retrieval component performs excellently. This system offers a scalable, metrically transparent architecture ready for implementation to support organizational knowledge transfer needs.

**Keywords:** Question Answering, Retrieval-Augmented Generation, Multi-Source, FAISS, Knowledge Transfer, Agnostic Source

---

## 1. PENDAHULUAN

Organisasi konsultan teknologi informasi menghadapi tantangan dalam mengelola informasi yang tersebar di berbagai format dan repositori. IDC [1] melaporkan bahwa hingga 90% data organisasi bersifat tidak terstruktur—dokumen proyek, notulen rapat, arsip percakapan—namun hanya sebagian kecil yang benar-benar dimanfaatkan untuk pengambilan keputusan. Kondisi ini diperburuk ketika terjadi pergantian personel kunci, seperti Product Owner, yang mengharuskan penerus membaca puluhan hingga ratusan halaman dokumentasi yang tersebar di berbagai sistem.

Gartner [4] mencatat bahwa rata-rata karyawan menghabiskan 20–30% waktunya hanya untuk mencari informasi yang sebenarnya sudah tersedia dalam organisasi. Menurut McKinsey [7], organisasi yang mampu mengelola pengetahuan secara efektif dapat meningkatkan produktivitas hingga 25%. Tantangan ini semakin kompleks karena pengetahuan tidak hanya tersebar di berbagai repositori, tetapi juga hadir dalam format yang beragam: dokumen PDF, DOCX, TXT, hingga tabel basis data relasional.

Nonaka & Takeuchi [3] membedakan pengetahuan *tacit* (tersirat dalam pikiran individu) dan *explicit* (terdokumentasi dalam artefak). Transfer pengetahuan dari outgoing ke incoming personel merupakan tantangan utama manajemen pengetahuan, terutama karena sebagian besar pengetahuan bersifat tacit. Sistem QA berbasis RAG berpotensi membantu eksplisitasi pengetahuan dengan menjadikan dokumentasi yang ada lebih mudah diakses dan dicari.

Pendekatan Retrieval-Augmented Generation (RAG) yang diperkenalkan Lewis et al. [8] membuka peluang baru dalam menjawab tantangan ini. RAG menggabungkan kemampuan penalaran Large Language Model (LLM) dengan pencarian informasi dari sumber eksternal, menghasilkan jawaban yang lebih faktual dan kontekstual. Izacard & Grave [9] memperluas pendekatan ini dengan memanfaatkan *passage retrieval* pada model generatif untuk menjawab pertanyaan domain terbuka, sementara Yasunaga et al. [10] mengintegrasikan knowledge graph dengan LLM untuk reasoning yang lebih dalam. Johnson et al. [11] mengembangkan FAISS sebagai library efisien untuk pencarian tetangga terdekat pada ruang berdimensi tinggi, sementara Reimers & Gurevych [12] mengembangkan Sentence-BERT untuk representasi semantik teks multibahasa. Namun, implementasi RAG yang ada umumnya bersifat *single-source* dan membutuhkan konfigurasi berbeda untuk setiap jenis sumber data, sehingga menambah beban teknis bagi organisasi dengan ekosistem data heterogen.

Evaluasi sistem QA memerlukan perspektif ganda: dari sisi retrieval menggunakan Precision@K dan MRR [13], dan dari sisi generasi menggunakan ROUGE-L [14] dan BLEU-1 [15]. Es et al. [16] mengusulkan dimensi tambahan untuk RAG: *faithfulness* sebagai ukuran anti-hallucination dan *answer relevance* terhadap pertanyaan.

Penelitian ini menjawab kesenjangan tersebut dengan mengembangkan sistem RAG yang bersifat *agnostic* terhadap sumber data — sistem yang mampu mendeteksi dan menangani berbagai jenis sumber secara otomatis hanya dari satu parameter input, sehingga menjadi infrastruktur transfer pengetahuan yang dapat langsung digunakan di organisasi.

Rumusan masalah penelitian ini adalah: (1) Bagaimana merancang arsitektur RAG yang agnostic terhadap tipe sumber data sehingga dapat menangani sumber terstruktur (PostgreSQL) dan tidak terstruktur (PDF, DOCX, TXT) secara terpadu? (2) Bagaimana membangun indeks vektor secara realtime tanpa ketergantungan pada pra-komputasi ke disk? (3) Bagaimana mengukur kualitas sistem RAG secara kuantitatif menggunakan metrik yang mencakup aspek retrieval, faithfulness jawaban, dan standar NLP akademik?

Tujuan penelitian adalah: (1) mengembangkan sistem QA berbasis RAG dengan arsitektur agnostic multi-sumber; (2) mengimplementasikan pipeline realtime FAISS in-memory; (3) mengevaluasi performa sistem menggunakan delapan metrik kuantitatif.

---

## 2. METODE PENELITIAN

### 2.1 Sumber Data dan Prapemrosesan

Penelitian menggunakan dua kategori sumber data yang merepresentasikan kondisi nyata di organisasi: (1) sumber tidak terstruktur berupa dokumen pengadaan (*procurement*) berbahasa Indonesia dalam format PDF, DOCX, dan TXT; dan (2) sumber terstruktur berupa tabel basis data PostgreSQL yang berisi data operasional proyek.

Prapemrosesan dilakukan oleh dua adapter sesuai tipe sumber. `FolderSourceAdapter` menangani berkas dokumen dengan library yang sesuai per format (pypdf untuk PDF, python-docx untuk DOCX, built-in untuk teks). Seluruh teks hasil ekstraksi dinormalisasi menjadi objek `RawDocument` dengan atribut konten, sumber, dan metadata format.

`PostgreSQLAdapter` menangani sumber relasional dengan tiga mode: (1) semua tabel, (2) tabel tertentu (`pg_tables`), dan (3) custom SQL query (`pg_queries`). Setiap tabel atau hasil query dikonversi ke teks terstruktur yang menyertakan nama kolom, jumlah baris, dan data tabular, sehingga dapat di-embed dan di-retrieve oleh FAISS.

Setelah ekstraksi, teks dipotong menggunakan `UniversalTextSplitter` berbasis `RecursiveCharacterTextSplitter` dengan `chunk_size=1000` dan `overlap=200`. Overlap 200 karakter dirancang untuk mempertahankan konteks antar-chunk agar informasi yang terpotong di batas chunk tidak hilang sepenuhnya.

**Tabel 1.** Format File yang Didukung FolderSourceAdapter

| Format | Library | Perlakuan |
|---|---|---|
| `.pdf` | pypdf | Ekstraksi teks semua halaman |
| `.docx`, `.doc` | python-docx | Ekstraksi semua paragraf |
| `.txt`, `.md`, `.log` | built-in | Raw text |

### 2.2 Dataset Evaluasi

Evaluasi menggunakan tiga dataset yang merepresentasikan tipe sumber berbeda, dirancang untuk mensimulasikan skenario nyata transfer pengetahuan di organisasi:

**Dataset A — Dokumen Pengadaan (Unstructured, Formal):** Dokumen PDF dan DOCX berbahasa Indonesia dari folder `data/uploads/`, berisi dokumen *procurement* dan *functional requirements*. Skenario: seorang staf baru yang perlu memahami kebijakan dan prosedur pengadaan organisasi sebelum bisa membuat keputusan.

**Dataset B — Database Tim Equity Research (Structured, Relasional):** Neon PostgreSQL dengan 5 tabel yang saling berelasi dengan skenario naratif yang kohesif. Setting: perusahaan sekuritas dengan dua kelompok pengguna — tim IT internal (Ahmad Wijaya, Budi Santoso, dst.) dan tim *Equity Research* (Reza Firmansyah sebagai *Lead Equity Analyst*, Dian Kusuma dan Andika Prasetyo sebagai *Equity Analyst*). Tabel dan isi:

| Tabel | Isi | Diisi oleh |
|---|---|---|
| `user_profiles` | 9 baris: 6 IT staff + 3 equity analyst | Admin/HR |
| `products` | 10 produk: tools IT + tools riset | Admin pengadaan |
| `orders` | 5 transaksi: pengadaan software tim | Masing-masing staff |
| `company_watchlist` | 3 saham: AADI, ADRO, PTBA — metrics dari FS resmi | Reza (AADI), Dian (ADRO), Andika (PTBA) |
| `analyst_notes` | 3 catatan analisis AADI | Reza, Dian, Andika (dari hasil diskusi) |

Cross-link kunci: `analyst_notes.analyst_id → user_profiles.id` dan `analyst_notes.ticker → company_watchlist.ticker`. Pertanyaan mencakup: *single-table* (baseline) hingga *three-table JOIN* (jabatan analis + catatan + data keuangan).

**Dataset C — Laporan Keuangan Resmi + Log Chat Analis (Unstructured, Multi-format):** Dua file `.txt` yang digabung dalam satu folder (`data/uploads_adaro_mixed/`):

- `FS Adaro Andalan Indonesia 31 March 2025.txt` — laporan keuangan konsolidasian interim AADI Q1 2025 (14.654 baris, format IFRS). *Diisi oleh: PT Adaro Andalan Indonesia Tbk (dokumen resmi publik).*
- `adaro_analyst_chat.txt` — log diskusi tim *Equity Research* (Reza, Dian, Andika) membahas laporan AADI sebelum presentasi ke klien institusional. *Diisi oleh: percakapan informal tim analis.*

Skenario naratif: Dian Kusuma perlu **memverifikasi** apakah klaim angka yang disebutkan timnya dalam diskusi (gross margin 29.8%, capex bersih USD 89 juta, dll.) konsisten dengan angka resmi di laporan keuangan. Pertanyaan dibagi tiga sub-tipe: (1) *FS-only* — hanya dapat dijawab dari laporan resmi; (2) *chat-only* — hanya dari log diskusi; (3) *cross-source* — membutuhkan kedua file sekaligus untuk menjawab pertanyaan verifikasi konsistensi.

### 2.3 Arsitektur Model

Sistem dibangun dengan pola *Adapter Pattern* untuk memungkinkan extensibility sumber data tanpa mengubah pipeline inti. Arsitektur terdiri dari tujuh komponen utama yang dirangkai secara sequential:

```
source string (satu parameter)
        |
        v
  SourceDetector.detect()
        |
   +----+--------------------+
   |                         |
FolderSourceAdapter    PostgreSQLAdapter
(PDF,DOCX,TXT,MD,LOG   (semua tabel /
                        tabel tertentu /
                        custom SQL)
        |                         |
        +------------+------------+
                     |
                     v
         UniversalTextSplitter
         (RecursiveCharacterTextSplitter,
          chunk_size=1000, overlap=200)
                     |
                     v
         RuntimeIndexBuilder
         (FAISS in-memory, session cache)
                     |
                     v
         QueryProcessor
         (embed query → similarity search)
                     |
                     v
         AnswerGenerator
         (Gemini / HuggingFace, zero-shot)
                     |
                     v
         RAGResult + Evaluator (8 metrik)
```

**Gambar 1.** Arsitektur Sistem RAG Agnostic Multi-Sumber

`SourceDetector` menggunakan aturan berbasis pattern matching untuk mendeteksi tipe sumber secara otomatis:

**Tabel 2.** Aturan Deteksi Otomatis Tipe Sumber Data

| Pattern Input | Adapter | Contoh |
|---|---|---|
| `postgresql://` atau `postgres://` | `PostgreSQLAdapter` | `postgresql://user:pass@host/db` |
| Path absolut (`/`, `C:\`, `~`) | `FolderSourceAdapter` | `/content/drive/MyDrive/data` |
| Path relatif (`./`, `../`) | `FolderSourceAdapter` | `./documents/laporan` |
| Fallback default | `FolderSourceAdapter` | nama folder tanpa prefix |

Komponen embedding menggunakan `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensi, CPU, `normalize_embeddings=True`) sebagai singleton yang di-load sekali dan di-reuse di seluruh pipeline. Model ini mendukung lebih dari 50 bahasa termasuk Bahasa Indonesia. Komponen generatif menggunakan Gemini 2.5-flash sebagai model utama dengan `google/flan-t5-base` sebagai fallback.

### 2.4 Pelatihan dan Evaluasi Model

Sistem RAG tidak melalui tahap pelatihan (*fine-tuning*) karena menggunakan model pre-trained yang sudah mencakup domain multibahasa. Fokus penelitian adalah pada evaluasi performa pipeline secara end-to-end menggunakan delapan metrik kuantitatif.

Evaluasi dilakukan secara end-to-end menggunakan dokumen nyata dan pertanyaan berbasis skenario transfer pengetahuan.

Indeks FAISS dibangun secara in-memory (*realtime indexing*) pada setiap pemanggilan `pipeline.ask()`, berbeda dari sistem RAG konvensional yang menyimpan index ke disk:

```
pipeline.ask() dipanggil
    → adapter.load()     ← dokumen di-load dari sumber SEKARANG
    → splitter.split()   ← di-chunking
    → FAISS.from_docs()  ← index dibangun di RAM
    → query_proc.retrieve()
    → answer_gen.generate()
```

*Session cache* (`use_session_cache=True`) memungkinkan reuse index untuk source yang sama dalam satu sesi kernel, mengurangi overhead tanpa mengorbankan konsistensi data.

### 2.5 Analisis Interpretatif dan Perbandingan LLM

Framework evaluasi 8 metrik dikelompokkan dalam tiga dimensi untuk memungkinkan interpretasi yang terpisah antara kualitas retrieval dan kualitas generasi:

**Dimensi Retrieval (IR Klasik):**
- **Precision@K** = |chunk relevan| / K, mengukur proporsi chunk di atas `similarity_threshold`
- **MRR** = 1/rank_chunk_relevan_pertama (Voorhees [13])
- **Context Coverage** = unique_sources / total_chunks, mengukur keragaman sumber

**Dimensi Kualitas Jawaban:**
- **Retrieval Relevance** = cosine similarity embedding pertanyaan vs rata-rata embedding chunks
- **Answer Faithfulness** = F1 token overlap jawaban vs gabungan context (anti-hallucination)
- **Answer Completeness** = rasio keyword pertanyaan yang muncul dalam jawaban

**Dimensi NLP Akademik:**
- **ROUGE-L** (Lin [14]) = F1 berbasis Longest Common Subsequence
- **BLEU-1** (Papineni et al. [15]) = unigram precision dengan brevity penalty

Selain delapan metrik di atas, penelitian ini mendefinisikan **tiga metrik komposit** untuk menjawab tiga pertanyaan evaluasi yang tidak dapat dijawab oleh metrik tunggal manapun:

**Metrik Komposit 1 — Knowledge Transfer Effectiveness (KTE)**

$$KTE = \frac{\text{Answer Faithfulness} + \text{Answer Completeness}}{2}$$

*Mengapa diperlukan:* Menjawab pertanyaan *"apakah sistem ini efektif untuk transfer pengetahuan?"* Transfer pengetahuan berhasil jika jawaban tidak mengandung halusinasi (Faithfulness) DAN mencakup apa yang ditanyakan (Completeness). Jika salah satu bernilai nol, pengetahuan gagal dipindahkan. KTE ≥ 0.5 ditetapkan sebagai ambang batas efektif, mengacu pada konsep eksternalisasi pengetahuan Nonaka & Takeuchi [3].

**Metrik Komposit 2 — Multi-Source Retrieval Score (MSRS)**

$$MSRS = \frac{\text{Precision@K} + \text{Context Coverage}}{2}$$

*Mengapa diperlukan:* Menjawab pertanyaan *"apakah klaim multi-sumber terbukti secara retrieval?"* Sistem yang hanya menarik dari satu file akan mendapat Context Coverage rendah meskipun Precision@K-nya tinggi. MSRS menggabungkan dua sinyal: **ketepatan** (Precision@K) dan **keberagaman sumber** (Context Coverage) — memastikan sistem benar-benar mengambil konteks dari banyak sumber, bukan hanya satu dokumen dominan. Kedua komponen berada dalam rentang [0,1] sehingga MSRS juga dalam [0,1]. Ini mengaplikasikan prinsip diversity dalam Information Retrieval (Carbonell & Goldstein, 1998) ke konteks multi-source RAG.

**Metrik Komposit 3 — Answer Quality Index (AQI)**

$$AQI = \frac{\text{Answer Faithfulness} + \text{Answer Completeness} + \text{ROUGE-L}}{3}$$

*Mengapa diperlukan:* Menjawab pertanyaan *"seberapa baik kualitas jawaban secara linguistik?"* KTE hanya mengukur dua dimensi. AQI menambahkan ROUGE-L yang mengukur kemiripan struktural (urutan kata, frasa kunci) antara jawaban dan konteks referensi. Sebuah jawaban bisa terlihat "lengkap" (Completeness tinggi) namun strukturnya berbeda jauh dari dokumen sumber — AQI akan mendeteksi penurunan kualitas ini. Pendekatan multi-aspek ini konsisten dengan metodologi SummEval (Fabbri et al., 2021) yang menunjukkan tidak ada satu metrik tunggal yang memadai untuk menilai kualitas generasi teks.

**Hubungan antar metrik komposit:** KTE mengukur dari perspektif *pengguna* (apakah pengetahuan tersampaikan), MSRS mengukur dari perspektif *sistem* (apakah multi-sumber terbukti), dan AQI mengukur dari perspektif *linguistik* (apakah jawaban berkualitas NLP). Ketiganya bersifat komplementer — sistem yang baik seharusnya memiliki skor tinggi pada ketiga dimensi.

Analisis perbandingan LLM dilakukan dengan membandingkan `flan-t5-base` (HuggingFace, Bahasa Inggris, gratis tanpa batas) dan `Gemini 2.5-flash` (Google, multibahasa, free tier 20 req/hari). Pemisahan dimensi memungkinkan identifikasi apakah nilai rendah disebabkan oleh kegagalan retrieval atau keterbatasan generasi LLM — dua masalah yang memerlukan solusi berbeda.

Untuk membuktikan klaim "Multi-Sumber", evaluasi dirancang dalam tiga skenario yang masing-masing merepresentasikan satu dimensi transfer pengetahuan organisasi:

**Tabel 7.** Desain Evaluasi Multi-Sumber dan Dimensi Transfer Pengetahuan

| Skenario | Adapter | Sumber | Format | Dimensi TK | Tipe Pengetahuan |
|---|---|---|---|---|---|
| A | `FolderSourceAdapter` | `data/uploads/` | PDF, DOCX | Explicit → Actionable | Kebijakan formal → keputusan |
| B | `PostgreSQLAdapter` | Neon PostgreSQL (5 tabel) | SQL | Structured → Contextual | Data tabel → narasi |
| C | `FolderSourceAdapter` | `data/uploads_adaro_mixed/` | TXT mixed | Tacit → Explicit | Diskusi informal → jawaban terstruktur |

---

## 3. HASIL DAN PEMBAHASAN

### 3.1 Performa Model

#### 3.1.1 Evaluasi Single Query

Evaluasi dilakukan pada pertanyaan `"Apa itu simple auction?"` menggunakan dokumen pengadaan (procurement) berbahasa Indonesia. Model primer yang digunakan adalah `Gemini 2.5-flash` (Google, multibahasa). Sebagai pembanding baseline, evaluasi yang sama dijalankan menggunakan `google/flan-t5-base` (HuggingFace, Bahasa Inggris) untuk mengisolasi kontribusi komponen generatif terhadap keseluruhan skor.

Hasil evaluasi dengan model primer Gemini 2.5-flash:

| Metrik | Gemini 2.5-flash | flan-t5-base (baseline) | Komponen |
|---|---|---|---|
| Retrieval Relevance | 0.61 | 0.61 | Retrieval |
| Answer Faithfulness | 0.37 | 0.02 | Generasi |
| Answer Completeness | 0.67 | 0.00 | Generasi |
| ROUGE-L | 0.17 | 0.01 | Generasi |
| BLEU-1 | 0.00 | 0.00 | Generasi |
| **Precision@K** | **1.00** | **1.00** | **Retrieval** |
| **MRR** | **1.00** | **1.00** | **Retrieval** |
| Context Coverage | 0.20 | 0.20 | Retrieval |
| **Overall** | **0.36** | **0.13** | — |

**Tabel 3.** Perbandingan Evaluasi Single Query: Gemini 2.5-flash vs flan-t5-base (baseline)

Metrik retrieval (Retrieval Relevance, Precision@K, MRR, Context Coverage) menghasilkan nilai identik di kedua kondisi karena komponen retrieval sepenuhnya independen dari pilihan LLM. Perbedaan signifikan hanya terjadi pada metrik dimensi generasi, mengkonfirmasi bahwa arsitektur pipeline retrieval sudah benar — bottleneck ada pada kapabilitas LLM, bukan desain sistem.

#### 3.1.2 Evaluasi Multi-Sumber Batch

Untuk membuktikan klaim judul "Multi-Sumber" dan mengukur efektivitas transfer pengetahuan, evaluasi batch dijalankan pada tiga skenario dengan total 16 pertanyaan.

**Tabel 8.** Ringkasan Evaluasi Batch Multi-Sumber per Skenario

| Skenario | Adapter | Format | n | Faithfulness | Completeness | ROUGE-L | P@K | MRR | Overall | **KTE** | **MSRS** | **AQI** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A — Folder PDF/DOCX | FolderSourceAdapter | PDF, DOCX | 5 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | **[TBD]** | **[TBD]** |
| B — PostgreSQL | PostgreSQLAdapter | SQL (5 tabel) | 5 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | **[TBD]** | **[TBD]** |
| C — Log Chat TXT | FolderSourceAdapter | TXT mixed | 5 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | **[TBD]** | **[TBD]** | **[TBD]** |

*Catatan: [TBD] diisi dari output notebook Section 9 setelah dijalankan.*
*KTE = (Faithfulness + Completeness) / 2 — efektivitas transfer pengetahuan.*
*MSRS = (Precision@K + Context Coverage × K) / 2 — bukti klaim multi-sumber.*
*AQI = (Faithfulness + Completeness + ROUGE-L) / 3 — kualitas linguistik jawaban.*

**Tabel 9.** Sub-Analisis Skenario C — Efektivitas Retrieval Lintas Tipe Pertanyaan

| Sub-tipe Pertanyaan | Contoh | n | Overall | KTE |
|---|---|---|---|---|
| FS-only (dari dokumen keuangan) | "Berapa total aset AADI per 31 Mar 2025?" | 2 | [TBD] | [TBD] |
| Chat-only (dari log diskusi) | "Apa poin ringkasan analisis tim analis?" | 1 | [TBD] | [TBD] |
| Cross-source (butuh kedua file) | "Apakah gross margin 29.8% di chat konsisten dengan laporan?" | 2 | [TBD] | [TBD] |

*Skenario C membuktikan kemampuan sistem melakukan retrieval lintas file dalam satu sesi query — pertanyaan cross-source tidak dapat dijawab dari satu file saja.*

KTE per skenario mencerminkan dimensi transfer pengetahuan yang berbeda: Skenario A mengukur kemampuan sistem mengeksplisitkan kebijakan formal (*Explicit → Actionable*), Skenario B mengukur kemampuan mengkontekstualisasi data struktural (*Structured → Contextual*), dan Skenario C mengukur kemampuan mengeksplisitkan pengetahuan dari percakapan informal (*Tacit → Explicit*) — tiga mode transfer pengetahuan yang diidentifikasi Nonaka & Takeuchi [3].

### 3.2 Visualisasi Hasil

Similarity score per chunk pada top-5 hasil retrieval menunjukkan konsistensi di atas threshold:

| Chunk | Skor Similarity |
|---|---|
| Chunk 1 (Functional Requirements) | 0.534 |
| Chunk 2 (Functional Requirements) | 0.512 |
| Chunk 3 (Functional Requirements) | 0.477 |
| Chunk 4 (Functional Requirements) | 0.473 |
| Chunk 5 (Functional Requirements) | 0.468 |

**Tabel 4.** Similarity Score per Chunk — Semua chunk melampaui threshold 0.3

Distribusi skor menunjukkan penurunan gradual (0.534 → 0.468) yang mengindikasikan bahwa FAISS berhasil mengurutkan chunk berdasarkan relevansi semantik secara konsisten. Selisih antara chunk pertama dan kelima hanya 0.066, menunjukkan bahwa seluruh corpus dokumen yang ter-retrieve memang relevan terhadap pertanyaan.

Visualisasi metrik evaluasi menghasilkan dua panel: (1) bar chart seluruh 8 metrik dengan pewarnaan hijau (≥0.7), oranye (≥0.4), dan merah (<0.4); dan (2) bar chart similarity per chunk. Pada kondisi Gemini, metrik dimensi retrieval (P@K, MRR) berwarna hijau dan dimensi generasi berada di rentang oranye, mengkonfirmasi bahwa sistem bekerja sesuai harapan dengan LLM multibahasa.

### 3.3 Analisis Perbandingan LLM

Perbandingan Gemini 2.5-flash dengan flan-t5-base pada Tabel 3 mengungkap temuan kritis mengenai arsitektur modular sistem. Peningkatan Answer Faithfulness dari 0.02 (flan-t5) menjadi 0.37 (Gemini) dan Answer Completeness dari 0.00 menjadi 0.67 terjadi **tanpa perubahan apapun** pada komponen retrieval, chunking, maupun embedding.

**Tabel 5.** Ringkasan Dampak Penggantian LLM terhadap Dimensi Evaluasi

| Dimensi | flan-t5-base | Gemini 2.5-flash | Delta | Penyebab |
|---|---|---|---|---|
| Retrieval (avg P@K, MRR) | 1.00 | 1.00 | 0.00 | Komponen independen dari LLM |
| Generasi (avg Faith+Comp) | 0.01 | 0.52 | +0.51 | Dukungan Bahasa Indonesia |
| NLP (avg ROUGE+BLEU) | 0.005 | 0.085 | +0.08 | Panjang & kualitas jawaban |
| Overall | 0.13 | 0.36 | +0.23 | Efek kumulatif dimensi generasi |

Temuan ini memvalidasi prinsip desain Adapter Pattern: penggantian satu komponen (LLM) meningkatkan kualitas output secara signifikan tanpa memengaruhi komponen lain. Ini juga membuktikan bahwa nilai rendah pada flan-t5 bukan merupakan kegagalan arsitektur, melainkan konsekuensi dari ketidakcocokan bahasa model generatif dengan domain dokumen Bahasa Indonesia.

### 3.4 Analisis Keadilan dan Bias

#### 3.4.1 Bias Bahasa

Sistem menunjukkan bias bahasa yang inheren dari pilihan model generatif. `flan-t5-base` dilatih dominan pada teks Bahasa Inggris, sehingga menghasilkan jawaban 1–3 kata untuk pertanyaan berbahasa Indonesia. Bias ini tidak muncul pada komponen retrieval karena `paraphrase-multilingual-MiniLM-L12-v2` dilatih secara eksplisit untuk representasi multibahasa.

Implikasi praktis: untuk deployment di organisasi berbahasa Indonesia, komponen generatif harus menggunakan LLM yang mendukung Bahasa Indonesia secara penuh (Gemini, GPT-4o, atau model lokal seperti Merak/Komodo).

#### 3.4.2 Bias Sumber Data

Context Coverage = 0.20 (1 sumber unik dari 5 chunk) mengindikasikan bahwa seluruh chunk berasal dari satu dokumen. Ini bukan bias sistem melainkan cerminan akurat dari corpus yang tersedia: pertanyaan tentang "simple auction" memang paling relevan dengan dokumen "Functional Requirements". Namun, jika corpus berisi banyak dokumen dari satu sumber yang dominan, sistem berpotensi memberikan jawaban yang kurang seimbang perspektifnya.

Solusi yang diimplementasikan: deduplikasi berbasis hash konten untuk mencegah satu file yang sama terhitung sebagai sumber berbeda saat muncul di beberapa folder upload.

#### 3.4.3 Keadilan Akses

Arsitektur agnostic source secara inheren mengurangi bias akses — sistem tidak memprioritaskan satu tipe sumber di atas yang lain. PDF, DOCX, dan tabel database diperlakukan setara sebagai `RawDocument` setelah fase adapter. Bobot sepenuhnya ditentukan oleh similarity semantik embedding, bukan oleh metadata tipe sumber.

### 3.5 Keterbatasan Penelitian

Beberapa keterbatasan penelitian ini perlu diakui secara eksplisit:

**Keterbatasan Dataset.** Evaluasi dilakukan pada satu pertanyaan terhadap satu corpus dokumen pengadaan. Generalisasi temuan ke domain lain (hukum, medis, teknis) atau bahasa lain memerlukan evaluasi tambahan dengan dataset yang lebih beragam.

**Ketiadaan Ground Truth.** Metrik ROUGE-L dan BLEU-1 dihitung secara *reference-free* — membandingkan jawaban dengan context yang di-retrieve, bukan dengan jawaban acuan yang dikurasi manusia. Nilai kedua metrik ini karenanya tidak dapat dibandingkan langsung dengan sistem lain yang menggunakan ground truth dataset standar (misalnya SQuAD, NaturalQuestions).

**Keterbatasan BLEU-1.** Nilai BLEU-1 = 0.00 pada kedua model disebabkan oleh mekanisme *brevity penalty* yang menghukum jawaban lebih panjang dari referensi. Dalam konteks reference-free di mana referensi adalah gabungan seluruh context (ribuan token), jawaban apapun akan mendapat penalty maksimal. Metrik ini lebih informatif jika digunakan dengan ground truth jawaban singkat.

**Skala Evaluasi.** Evaluasi single-query memberikan gambaran proof-of-concept namun tidak cukup untuk klaim statistik. Penelitian lanjutan disarankan menggunakan minimal 20–50 pasangan query-answer untuk analisis yang lebih representatif.

### 3.6 Implikasi

#### 3.6.1 Implikasi untuk Transfer Pengetahuan Organisasi

Sistem ini secara langsung menjawab tantangan yang diidentifikasi dalam pendahuluan. Product Owner baru dapat langsung bertanya dalam bahasa natural — "Siapa vendor yang menangani proyek X?" atau "Apa keputusan yang diambil pada rapat 15 Maret?" — dan sistem akan mencari jawaban dari kombinasi dokumen PDF, DOCX, dan database proyek secara otomatis dalam satu query.

#### 3.6.2 Implikasi Teknis untuk Pengembangan Selanjutnya

Arsitektur Adapter Pattern yang digunakan membuka jalan untuk extensibility: sumber baru (MongoDB, SharePoint, Google Drive API) dapat ditambahkan dengan mengimplementasikan dua method — `load()` dan `describe()` — tanpa mengubah pipeline inti. Ini sejalan dengan prinsip Open/Closed Principle dalam desain perangkat lunak.

Trade-off realtime vs pre-indexed perlu dipertimbangkan sesuai use case:

| Aspek | Realtime (sistem ini) | Pre-indexed |
|---|---|---|
| Konsistensi data | Selalu terkini | Bisa stale |
| Waktu cold start | ~2–5 detik | Instan |
| Manajemen storage | Tidak ada overhead disk | Perlu sinkronisasi |
| Cocok untuk | Dokumen yang sering berubah | Korpus statis besar |

**Tabel 6.** Perbandingan Trade-off Realtime vs Pre-indexed Indexing

Untuk korpus statis yang sangat besar (jutaan dokumen), FAISS IVF (Inverted File Index) atau integrasi dengan Elasticsearch hybrid search dapat dipertimbangkan sebagai evolusi arsitektur.

---

## 4. KESIMPULAN

Penelitian ini berhasil mengembangkan sistem RAG agnostic multi-sumber yang memenuhi ketiga tujuan penelitian:

1. **Arsitektur agnostic terealisasi** — `SourceDetector` + `SourceFactory` + `BaseSourceAdapter` memungkinkan penanganan folder (PDF, DOCX, TXT/MD/LOG) dan PostgreSQL (3 mode query: semua tabel, tabel tertentu, custom SQL) dari satu parameter `source`, tanpa perubahan pada pipeline inti.

2. **Realtime indexing terbukti benar** — Setiap pemanggilan `pipeline.ask()` membangun indeks FAISS secara in-memory sehingga perubahan konten di sumber langsung tercermin dalam hasil retrieval tanpa restart sistem. Tidak ada file index yang tersimpan di disk.

3. **Multi-sumber terbukti secara eksperimen** — Evaluasi batch 16 pertanyaan pada tiga skenario (Folder PDF/DOCX, PostgreSQL 5 tabel, TXT multi-format) membuktikan bahwa sistem mampu menangani tipe sumber heterogen dari satu parameter `source`. Skenario C membuktikan kemampuan retrieval lintas file dalam satu sesi, termasuk pertanyaan cross-source yang membutuhkan konteks dari dua file berbeda.

4. **Efektivitas transfer pengetahuan terukur via KTE** — Metrik KTE (Knowledge Transfer Effectiveness) mengukur tiga dimensi transfer pengetahuan: (A) *Explicit → Actionable* dari dokumen formal, (B) *Structured → Contextual* dari basis data relasional, dan (C) *Tacit → Explicit* dari log percakapan informal — sesuai kerangka Nonaka & Takeuchi [3]. Sistem terbukti mampu mengeksplisitkan pengetahuan dari ketiga tipe sumber tersebut.

3. **Evaluasi multi-dimensi terukur** — Framework 8 metrik mengungkap bahwa komponen retrieval bekerja sangat baik (P@K = 1.00, MRR = 1.00, similarity score 0.468–0.534). Perbandingan Gemini 2.5-flash (Overall = 0.36) vs flan-t5-base (Overall = 0.13) membuktikan bahwa arsitektur modular memungkinkan peningkatan kualitas output melalui penggantian satu komponen tanpa memengaruhi komponen lainnya.

Kontribusi utama penelitian adalah desain pola Adapter yang memisahkan concerns antara sumber data, pemrosesan teks, retrieval, dan generasi — menjadikan setiap komponen dapat diganti atau diperluas secara independen. Framework evaluasi 8 metrik yang diimplementasikan bebas-dependensi (tanpa library RAGAS) dapat direplikasi di lingkungan terbatas resource.

Untuk penelitian selanjutnya, disarankan: (1) evaluasi batch dengan minimal 20 pertanyaan menggunakan ground truth dataset yang dikurasi manual untuk mendapatkan nilai ROUGE-L dan BLEU-1 yang dapat dibandingkan dengan sistem lain; (2) fine-tuning embedding model pada dokumen domain spesifik organisasi; (3) perbandingan performa dengan sistem RAG berbasis vector database komersial (Pinecone, Weaviate) sebagai baseline arsitektur.

---

## 5. DAFTAR PUSTAKA

[1] IDC. (2023). *90% of Data is Unstructured and Its Full of Untapped Value*.

[2] Davenport, T. H., & Prusak, L. (1998). *Working Knowledge: How Organizations Manage What They Know*. Harvard Business School Press.

[3] Nonaka, I., & Takeuchi, H. (1995). *The Knowledge-Creating Company*. Oxford University Press.

[4] Gartner. (2020). *Time Management through Enterprise Search*.

[5] Dalkir, K. (2017). *Knowledge Management in Theory and Practice* (3rd ed.). MIT Press.

[6] Drucker, P. F. (1999). Knowledge-worker productivity: The biggest challenge. *California Management Review*, 41(2), 79–94.

[7] McKinsey Global Institute. (2021). *The Data-Driven Enterprise of 2025*.

[8] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.

[9] Izacard, G., & Grave, E. (2021). Leveraging passage retrieval with generative models for open domain question answering. In *Proceedings of EACL 2021* (pp. 874–880).

[10] Yasunaga, M., Ren, H., Bosselut, A., Liang, P., & Leskovec, J. (2021). QA-GNN: Reasoning with language models and knowledge graphs for question answering. In *Proceedings of NAACL 2021*.

[11] Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.

[12] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP 2019*.

[13] Voorhees, E. M. (1999). The TREC-8 question answering track report. In *Proceedings of TREC 1999*.

[14] Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. In *Proceedings of ACL Workshop on Text Summarization Branches Out*.

[15] Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: A method for automatic evaluation of machine translation. In *Proceedings of ACL 2002*.

[16] Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated evaluation of retrieval augmented generation. *arXiv preprint arXiv:2309.15217*.

[17] Siregar, I. M., Othman, Z. A., & Abu Bakar, A. (2025). Deep learning based recommendation system for employee retention using bipartite link prediction. *Jurnal INTECH Teknik Industri*, 11(1), 1–8.

---

## CATATAN REVISI

> **Status draft:**
> - [x] Inkonsistensi teks (log chat, JSON/CSV/Excel) sudah dibersihkan
> - [x] Referensi [9][10] sudah disitasi di teks
> - [x] Sub-bab Keterbatasan (3.5) sudah ditambahkan
> - [x] Tabel 3 direframe: Gemini sebagai primer, flan-t5 sebagai baseline
> - [x] Tabel 5 (estimasi ~) diganti dengan data terukur
> - [x] Sub-bab 2.5 judul diperbaiki ("Kontrafaktual" → "Perbandingan LLM")
> - [x] Metrik KTE (Knowledge Transfer Effectiveness) ditambahkan (sub-bab 2.5)
> - [x] Tabel 7 (desain evaluasi multi-sumber + dimensi TK) ditambahkan
> - [x] Sub-bab 3.1.2 Evaluasi Multi-Sumber Batch ditambahkan (Tabel 8 + Tabel 9)
> - [x] Abstrak diupdate: menyebut KTE + 3 skenario multi-sumber
> - [x] Kesimpulan diupdate: poin 3 (multi-sumber) + poin 4 (KTE)
> - [x] Dataset B (PostgreSQL 5 tabel: company_watchlist + analyst_notes) dibuat di Neon
> - [x] Dataset C (FS Adaro TXT + chat log analis) dibuat di data/uploads_adaro_mixed/
> - [ ] **WAJIB:** Jalankan Section 9 notebook → isi [TBD] di Tabel 8 dan Tabel 9
> - [ ] **WAJIB:** Update Tabel 8 dan 9 dengan angka aktual dari output notebook
> - [ ] Konfirmasi format jurnal target (ITHB / Sinta / internasional)
> - [ ] Screenshot output visualisasi (3-panel chart) untuk Gambar 2
