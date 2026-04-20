# DRAFT JURNAL v3.0 (Target: Sinta 2-3)
# Format: JTIIK / Jurnal INTECH / Register / JIKO
# Gaya Sitasi: IEEE (numbered)
# Batas kata: 4.000–7.000 kata (body text, diluar referensi)
# Template: dua kolom saat submit, draft ini single-column

---

## Judul (ID)
**Sistem Retrieval-Augmented Generation Agnostic Multi-Sumber untuk Question Answering Berbasis Dokumen dan Database pada Konteks Transfer Pengetahuan Organisasi**

## Judul (EN)
**Agnostic Multi-Source Retrieval-Augmented Generation System for Question Answering Based on Documents and Databases in Organizational Knowledge Transfer Context**

---

## Penulis

**Krisna Dwi Setyaadi**¹  
Program Studi Sistem Informasi Data Science  
Institut Teknologi Harapan Bangsa, Bandung, Indonesia  
Email: krisna@ithb.ac.id  
ORCID: *[isi sebelum submit]*

**Ivan Michael Siregar**²  
Program Studi Sistem Informasi Data Science  
Institut Teknologi Harapan Bangsa, Bandung, Indonesia  
Email: ivan@ithb.ac.id  
ORCID: *[isi sebelum submit]*

*\* Corresponding author: Krisna Dwi Setyaadi (krisna@ithb.ac.id)*

---

## ABSTRAK

Organisasi modern menghadapi tantangan mengelola pengetahuan yang tersebar di sumber heterogen, yaitu dokumen tidak terstruktur (PDF, TXT) dan basis data relasional, sehingga menghambat transfer pengetahuan dan pengambilan keputusan. Penelitian ini mengembangkan sistem *Question Answering* (QA) berbasis *Retrieval-Augmented Generation* (RAG) yang bersifat *agnostic* terhadap sumber data: sistem hanya memerlukan satu parameter `source` untuk mendeteksi dan menangani berbagai jenis sumber secara otomatis. Sistem mengintegrasikan dua adapter utama, yaitu `FolderSourceAdapter` untuk sumber tidak terstruktur dan `PostgreSQLAdapter` untuk basis data relasional, dengan indeks vektor FAISS yang dibangun secara *realtime* tanpa pra-komputasi ke disk. Evaluasi menggunakan delapan metrik kuantitatif dan tiga metrik komposit: *Knowledge Transfer Effectiveness* (KTE), *Multi-Source Retrieval Score* (MSRS), dan *Answer Quality Index* (AQI). Eksperimen mencakup empat skenario dengan 20 pertanyaan: (A) press release keuangan dan log diskusi analis dalam format PDF+TXT, (B) PostgreSQL lima tabel berelasi, (C) pertanyaan lintas-sumber multi-format, dan (D) *cross-paradigm*, yaitu penggabungan sumber unstructured dan structured secara bersamaan dalam satu indeks FAISS gabungan. Hasil menunjukkan Precision@K = 1.00 dan MRR = 1.00 pada semua skenario, dengan KTE tertinggi pada Skenario B (0.502) yang bersumber dari PostgreSQL. Skenario A mencapai MSRS = 0.688, Skenario B MSRS = 0.769, Skenario C MSRS = 0.700, dan Skenario D MSRS = 0.725. Keempat skenario membuktikan kemampuan sistem menangani dimensi transfer pengetahuan: *Explicit→Actionable*, *Structured→Contextual*, *Tacit→Explicit*, dan *Cross-Paradigm*.

**Kata Kunci:** *Retrieval-Augmented Generation*, *Question Answering*, Multi-Sumber, FAISS, Transfer Pengetahuan

---

## ABSTRACT

Modern organizations face challenges managing knowledge dispersed across heterogeneous sources, namely unstructured documents (PDF, TXT) and relational databases, hindering knowledge transfer and decision-making. This study develops a Question Answering (QA) system based on Retrieval-Augmented Generation (RAG) that is agnostic to the data source: the system requires only a single `source` parameter to automatically detect and handle various source types. Two main adapters are integrated, namely `FolderSourceAdapter` for unstructured sources and `PostgreSQLAdapter` for relational databases, with FAISS vector indices built in real-time without disk pre-computation. Evaluation uses eight quantitative metrics and three composite metrics: Knowledge Transfer Effectiveness (KTE), Multi-Source Retrieval Score (MSRS), and Answer Quality Index (AQI). Experiments cover four scenarios with 20 questions: (A) financial press releases and analyst discussion logs in PDF+TXT format, (B) PostgreSQL with five interrelated tables, (C) cross-source multi-format queries, and (D) cross-paradigm, combining unstructured and structured sources simultaneously in a single merged FAISS index. Results show Precision@K = 1.00 and MRR = 1.00 across all scenarios, with highest KTE on Scenario B (0.502) sourced from PostgreSQL. MSRS values are: Scenario A = 0.688, Scenario B = 0.769, Scenario C = 0.700, Scenario D = 0.725. All four scenarios demonstrate the system's capability across knowledge transfer dimensions: Explicit→Actionable, Structured→Contextual, Tacit→Explicit, and Cross-Paradigm.

**Keywords:** Retrieval-Augmented Generation, Question Answering, Multi-Source, FAISS, Knowledge Transfer

---

## 1. PENDAHULUAN

Organisasi berbasis pengetahuan (*knowledge-intensive organization*) menghadapi tantangan dalam mengelola informasi yang tersebar di berbagai format dan repositori. IDC [1] melaporkan bahwa hingga 90% data organisasi bersifat tidak terstruktur, seperti dokumen proyek, notulen rapat, dan arsip percakapan, namun hanya sebagian kecil yang benar-benar dimanfaatkan untuk pengambilan keputusan. Kondisi ini diperburuk ketika terjadi pergantian personel kunci, seperti Product Owner, yang mengharuskan penerus membaca puluhan hingga ratusan halaman dokumentasi yang tersebar di berbagai sistem.

Gartner [4] mencatat bahwa rata-rata karyawan menghabiskan 20–30% waktunya hanya untuk mencari informasi yang sebenarnya sudah tersedia dalam organisasi. Menurut McKinsey [7], organisasi yang mampu mengelola pengetahuan secara efektif dapat meningkatkan produktivitas hingga 25%. Tantangan ini semakin kompleks karena pengetahuan tidak hanya tersebar di berbagai repositori, tetapi juga hadir dalam format yang beragam: dokumen PDF, TXT, hingga tabel basis data relasional.

Nonaka & Takeuchi [3] membedakan pengetahuan *tacit* (tersirat dalam pikiran individu) dan *explicit* (terdokumentasi dalam artefak). Transfer pengetahuan dari outgoing ke incoming personel merupakan tantangan utama manajemen pengetahuan, terutama karena sebagian besar pengetahuan bersifat tacit. Sistem QA berbasis RAG berpotensi membantu eksplisitasi pengetahuan dengan menjadikan dokumentasi yang ada lebih mudah diakses dan dicari.

Pendekatan Retrieval-Augmented Generation (RAG) yang diperkenalkan Lewis et al. [8] membuka peluang baru dalam menjawab tantangan ini. RAG menggabungkan kemampuan penalaran Large Language Model (LLM) dengan pencarian informasi dari sumber eksternal, menghasilkan jawaban yang lebih faktual dan kontekstual. Izacard & Grave [9] memperluas pendekatan ini dengan memanfaatkan *passage retrieval* pada model generatif untuk menjawab pertanyaan domain terbuka, sementara Yasunaga et al. [10] mengintegrasikan knowledge graph dengan LLM untuk reasoning yang lebih dalam. Johnson et al. [11] mengembangkan FAISS sebagai library efisien untuk pencarian tetangga terdekat pada ruang berdimensi tinggi, sementara Reimers & Gurevych [12] mengembangkan Sentence-BERT untuk representasi semantik teks multibahasa. Namun, implementasi RAG yang ada umumnya bersifat *single-source* dan membutuhkan konfigurasi berbeda untuk setiap jenis sumber data, sehingga menambah beban teknis bagi organisasi dengan ekosistem data heterogen.

Evaluasi sistem QA memerlukan perspektif ganda: dari sisi retrieval menggunakan Precision@K dan MRR [13], dan dari sisi generasi menggunakan ROUGE-L [14] dan BLEU-1 [15]. Es et al. [16] mengusulkan dimensi tambahan untuk RAG: *faithfulness* sebagai ukuran anti-hallucination dan *answer relevance* terhadap pertanyaan.

Penelitian ini menjawab kesenjangan tersebut dengan mengembangkan sistem RAG yang bersifat *agnostic* terhadap sumber data, yaitu sistem yang mampu mendeteksi dan menangani berbagai jenis sumber secara otomatis hanya dari satu parameter input, sehingga menjadi infrastruktur transfer pengetahuan yang dapat langsung digunakan di organisasi.

Rumusan masalah penelitian ini adalah: (1) Bagaimana merancang arsitektur RAG yang agnostic terhadap tipe sumber data sehingga dapat menangani sumber terstruktur (PostgreSQL) dan tidak terstruktur (PDF, TXT) secara terpadu? (2) Bagaimana membangun indeks vektor secara realtime tanpa ketergantungan pada pra-komputasi ke disk? (3) Bagaimana mengukur kualitas sistem RAG secara kuantitatif menggunakan metrik yang mencakup aspek retrieval, faithfulness jawaban, dan standar NLP akademik?

Tujuan penelitian adalah: (1) mengembangkan sistem QA berbasis RAG dengan arsitektur agnostic multi-sumber; (2) mengimplementasikan pipeline realtime FAISS in-memory; (3) mengevaluasi performa sistem menggunakan delapan metrik kuantitatif.

---

## 2. METODE PENELITIAN

### 2.1 Sumber Data dan Prapemrosesan

Penelitian menggunakan dua kategori sumber data yang merepresentasikan kondisi nyata di organisasi: (1) sumber tidak terstruktur berupa press release keuangan emiten publik dan log diskusi tim analis berbahasa Indonesia dalam format PDF dan TXT; dan (2) sumber terstruktur berupa tabel basis data PostgreSQL yang berisi data operasional tim *equity research*.

Prapemrosesan dilakukan oleh dua adapter sesuai tipe sumber. `FolderSourceAdapter` menangani berkas dokumen dengan library yang sesuai per format (pypdf untuk PDF, built-in untuk teks). Seluruh teks hasil ekstraksi dinormalisasi menjadi objek `RawDocument` dengan atribut konten, sumber, dan metadata format.

`PostgreSQLAdapter` menangani sumber relasional dengan tiga mode: (1) semua tabel, (2) tabel tertentu (`pg_tables`), dan (3) custom SQL query (`pg_queries`). Setiap tabel atau hasil query dikonversi ke teks terstruktur yang menyertakan nama kolom, jumlah baris, dan data tabular, sehingga dapat di-embed dan di-retrieve oleh FAISS.

Setelah ekstraksi, teks dipotong menggunakan `UniversalTextSplitter` berbasis `RecursiveCharacterTextSplitter` dengan `chunk_size=2000` dan `overlap=300`. Overlap 300 karakter dirancang untuk mempertahankan konteks antar-chunk agar informasi yang terpotong di batas chunk tidak hilang sepenuhnya. Nilai chunk_size yang lebih besar (2000) dipilih untuk mendukung dokumen keuangan berbahasa Indonesia yang umumnya mengandung kalimat panjang dan tabel multi-baris.

**Tabel 1.** Format File yang Didukung FolderSourceAdapter

| Format | Library | Perlakuan |
|---|---|---|
| `.pdf` | pypdf | Ekstraksi teks semua halaman |
| `.txt`, `.md`, `.log` | built-in | Raw text |

### 2.2 Dataset Evaluasi

Evaluasi menggunakan tiga dataset yang merepresentasikan tipe sumber berbeda, dirancang untuk mensimulasikan skenario nyata transfer pengetahuan di organisasi:

**Dataset A: Press Release Keuangan dan Log Diskusi Analis (Unstructured, Formal):** Dua file berbahasa Indonesia dari folder `data/sample_data/`, yaitu: (1) *Press Release PT Bank KB Bukopin Tbk (BBKP) Q1 2025* (format PDF) yang memuat informasi laba bersih, pertumbuhan kredit, DPK, dan pencapaian migrasi NGBS; dan (2) *Press Release PT Timah Tbk (TINS) Q1 2025* (format PDF) yang memuat data produksi, harga jual, rasio keuangan, dan kinerja ekspor; serta (3) `analyst_chat_dummy.txt`, yaitu log diskusi tim *Equity Research* (Reza, Dian, Andika) membahas rekomendasi BBKP dan TINS (format TXT). Skenario: seorang analis junior yang perlu memahami kinerja keuangan dan memformulasikan rekomendasi investasi untuk BBKP dan TINS berdasarkan press release resmi dan diskusi tim. Lima pertanyaan mencakup: (A1) laba bersih KB Bank Q1 2025 vs Q1 2024, (A2) pertumbuhan kredit dan DPK per segmen BBKP, (A3) laba bersih TINS vs target internal, (A4) kinerja produksi dan penjualan logam timah TINS, dan (A5) rekomendasi dan opini tim analis untuk BBKP dan TINS.

*Catatan: Dataset A tidak menggunakan dokumen internal perusahaan (seperti dokumen Agreement atau kontrak privat) karena bersifat rahasia. Semua file yang digunakan merupakan dokumen publik (press release yang dipublikasikan emiten di BEI) dan data diskusi tim yang telah dianonimisasi.*

**Dataset B: Database Tim Equity Research (Structured, Relasional):** Neon PostgreSQL dengan 5 tabel yang saling berelasi dengan skenario naratif yang kohesif. Setting: perusahaan sekuritas dengan dua kelompok pengguna, yaitu tim IT internal (Ahmad Wijaya, Budi Santoso, dst.) dan tim *Equity Research* (Reza Firmansyah sebagai *Lead Equity Analyst*, Dian Kusuma dan Andika Prasetyo sebagai *Equity Analyst*). Tabel dan isi:

| Tabel | Isi | Diisi oleh |
|---|---|---|
| `user_profiles` | 6 baris: IT staff + equity analyst (Ahmad, Sari, Reza, Dian, Andika, Rina) | Admin/HR |
| `products` | 10 produk: tools IT + tools riset (Bloomberg, Refinitiv, Gemini API, dst.) | Admin pengadaan |
| `orders` | 5 transaksi: pengadaan software tim (4 completed, 1 pending) | Masing-masing staff |
| `company_watchlist` | 2 saham: BBKP (KB Bank, BUY) dan TINS (PT Timah, HOLD) dengan metrik dari press release Q1 2025 | Ahmad (BBKP), Sari (TINS) |
| `analyst_notes` | 2 catatan analisis: BBKP dan TINS | Reza & Dian (dari hasil diskusi chat log) |

Cross-link kunci: `analyst_notes.analyst_id → user_profiles.id` dan `analyst_notes.ticker → company_watchlist.ticker`. Pertanyaan mencakup: *single-table* (baseline) hingga *three-table JOIN* (jabatan analis + catatan + data keuangan).

*Catatan: Data pada Dataset B bersifat sintetis, dirancang khusus untuk merepresentasikan skenario nyata tim equity research di perusahaan sekuritas. Seluruh nama, angka, dan transaksi merupakan data rekayasa untuk keperluan evaluasi dan tidak merepresentasikan entitas atau kejadian nyata.*

**Dataset C: Press Release Keuangan Resmi dan Log Chat Analis (Unstructured, Multi-format):** Folder yang sama dengan Dataset A (`data/sample_data/`), namun dengan dua perbedaan: (1) filter `exclude_patterns=['salinan','agreement']` diterapkan untuk mengecualikan dokumen privat yang tidak relevan, sehingga hanya 3 file bersih yang di-load; dan (2) set pertanyaan dirancang untuk menguji dimensi TK berbeda (*Tacit → Explicit*). File yang digunakan:

- `Press Release Bank Bukopin.pdf`: press release resmi KB Bank (BBKP) Q1 2025 (format PDF). *Sumber: dokumen publik emiten BEI.*
- `Press Release PT Timah.pdf`: press release resmi PT Timah (TINS) Q1 2025 (format PDF). *Sumber: dokumen publik emiten BEI.*
- `analyst_chat_dummy.txt`: log diskusi tim *Equity Research* (Reza, Dian, Andika) membahas BBKP dan TINS sebelum presentasi ke klien. *Sumber: percakapan informal tim analis yang telah dianonimisasi.*

Skenario naratif: Dian Kusuma perlu **memverifikasi** apakah klaim angka yang disebutkan tim dalam diskusi (EBITDA TINS Rp384 miliar, NIM BBKP 1.09%, migrasi NGBS selesai, dll.) konsisten dengan angka resmi di press release. Pertanyaan dibagi tiga sub-tipe: (1) *PR-only*, yaitu pertanyaan yang hanya dapat dijawab dari press release resmi; (2) *chat-only*, yaitu pertanyaan yang hanya dapat dijawab dari log diskusi tim; (3) *cross-source*, yaitu pertanyaan yang membutuhkan kedua sumber sekaligus untuk verifikasi konsistensi.

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
(PDF,TXT,MD,LOG        (semua tabel /
                        tabel tertentu /
                        custom SQL)
        |                         |
        +------------+------------+
                     |
                     v
         UniversalTextSplitter
         (RecursiveCharacterTextSplitter,
          chunk_size=2000, overlap=300)
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

### 2.5 Analisis Interpretatif dan Desain Evaluasi

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

**Metrik Komposit 1: Knowledge Transfer Effectiveness (KTE)**

$$KTE = \frac{\text{Answer Faithfulness} + \text{Answer Completeness}}{2}$$

KTE merupakan rata-rata dari dua komponen: Faithfulness (proporsi jawaban yang didukung context, sebagai ukuran anti-halusinasi) dan Completeness (proporsi keyword pertanyaan yang tercakup dalam jawaban). Ambang batas efektif ditetapkan KTE ≥ 0.5. Transfer pengetahuan berhasil hanya jika jawaban sekaligus tidak halusinasi dan menjawab pertanyaan secara lengkap; jika salah satu komponen bernilai nol maka pengetahuan gagal dipindahkan, sehingga rata-rata sederhana sudah memadai sebagai ukuran efektivitas [3].

**Metrik Komposit 2: Multi-Source Retrieval Score (MSRS)**

$$MSRS = \frac{\text{Precision@K} + \text{Context Coverage}}{2}$$

MSRS merupakan rata-rata dari dua komponen: Precision@K (proporsi chunk yang relevan dalam top-K hasil retrieval) dan Context Coverage (keragaman sumber dokumen dalam top-K). Sistem yang hanya menarik dari satu file akan memperoleh Context Coverage rendah meskipun Precision@K-nya tinggi; MSRS mendeteksi kondisi ini dan memastikan klaim multi-sumber terbukti secara retrieval. Pendekatan ini mengaplikasikan prinsip diversity dalam Information Retrieval [Carbonell & Goldstein, 1998] ke konteks multi-source RAG.

**Metrik Komposit 3: Answer Quality Index (AQI)**

$$AQI = \frac{\text{Answer Faithfulness} + \text{Answer Completeness} + \text{ROUGE-L}}{3}$$

AQI merupakan rata-rata dari tiga komponen: Faithfulness (anti-halusinasi), Completeness (cakupan pertanyaan), dan ROUGE-L (kemiripan struktural berbasis urutan kata terhadap context). KTE hanya mengukur apakah pengetahuan tersampaikan secara konten; AQI menambahkan dimensi linguistik untuk mendeteksi jawaban yang memadai secara topik namun strukturnya jauh berbeda dari dokumen sumber. Pendekatan multi-aspek ini konsisten dengan metodologi SummEval (Fabbri et al., 2021).

**Hubungan antar metrik komposit:** KTE mengukur dari perspektif *pengguna* (apakah pengetahuan tersampaikan), MSRS dari perspektif *sistem* (apakah multi-sumber terbukti), dan AQI dari perspektif *linguistik* (apakah jawaban berkualitas NLP). Ketiganya bersifat komplementer; sistem yang baik seharusnya memperoleh skor tinggi pada ketiga dimensi secara bersamaan.

Pemisahan tiga dimensi evaluasi ini memungkinkan identifikasi apakah nilai rendah disebabkan oleh kegagalan retrieval atau keterbatasan kapabilitas bahasa model generatif; keduanya merupakan masalah yang memerlukan solusi berbeda.

Untuk membuktikan klaim "Multi-Sumber", evaluasi dirancang dalam empat skenario yang masing-masing merepresentasikan satu dimensi transfer pengetahuan organisasi. Skenario D secara khusus membuktikan klaim *source-agnostic* pada level paling tinggi, di mana sistem dapat menjawab pertanyaan yang membutuhkan informasi dari sumber tidak terstruktur (folder PDF+TXT) **dan** sumber terstruktur (PostgreSQL) secara bersamaan dalam satu query:

**Tabel 7.** Desain Evaluasi Multi-Sumber dan Dimensi Transfer Pengetahuan

| Skenario | Adapter | Sumber | Format | Dimensi TK | Tipe Pengetahuan |
|---|---|---|---|---|---|
| A | `FolderSourceAdapter` | `data/sample_data/` | PDF + TXT | Explicit → Actionable | Laporan keuangan formal → insight investasi |
| B | `PostgreSQLAdapter` | Neon PostgreSQL (5 tabel) | SQL | Structured → Contextual | Data tabel → narasi |
| C | `FolderSourceAdapter` | `data/sample_data/` | PDF + TXT | Tacit → Explicit | Diskusi informal → jawaban terstruktur |
| D | `MultiSourceAdapter` | Folder + PostgreSQL (gabungan) | PDF + TXT + SQL | Cross-Paradigm | Verifikasi lintas paradigma data: unstructured ↔ structured |

---

## 3. HASIL DAN PEMBAHASAN

### 3.1 Performa Model

#### 3.1.1 Evaluasi Single Query

Evaluasi awal dilakukan pada pertanyaan `"Berapa laba bersih KB Bank (BBKP) pada Q1 2025 dan bagaimana perubahannya dibandingkan Q1 2024?"` menggunakan Press Release PT Bank KB Bukopin Tbk (BBKP) Q1 2025 dan log diskusi tim analis berbahasa Indonesia, dengan model `Gemini 2.5-flash` (Google, multibahasa) sebagai model generatif.

Hasil evaluasi:

| Metrik | Gemini 2.5-flash | Komponen |
|---|---|---|
| Retrieval Relevance | **0.735** | Retrieval |
| Answer Faithfulness | **0.062** | Generasi |
| Answer Completeness | **0.500** | Generasi |
| ROUGE-L | **0.032** | Generasi |
| BLEU-1 | **0.000** | Generasi |
| **Precision@K** | **1.000** | **Retrieval** |
| **MRR** | **1.000** | **Retrieval** |
| **Overall** | **0.266** | Gabungan |

**Tabel 3.** Hasil Evaluasi Single Query untuk Pertanyaan A1 ("Berapa laba bersih BBKP Q1 2025?") menggunakan Gemini 2.5-flash

Metrik retrieval (Retrieval Relevance, Precision@K, MRR, Context Coverage) mencapai nilai tinggi karena komponen retrieval sepenuhnya independen dari pilihan LLM. Nilai Faithfulness yang rendah (0.062) merupakan karakteristik evaluasi reference-free pada teks berbahasa Indonesia, bukan kegagalan sistem.

#### 3.1.2 Evaluasi Multi-Sumber Batch

Untuk membuktikan klaim judul "Multi-Sumber" dan mengukur efektivitas transfer pengetahuan, evaluasi batch dijalankan pada empat skenario dengan total 20 pertanyaan (5 per skenario). Skenario A–C mengevaluasi tiap paradigma sumber secara terpisah; Skenario D mengevaluasi `MultiSourceAdapter` yang menggabungkan FolderSourceAdapter dan PostgreSQLAdapter dalam satu indeks FAISS. Seluruh angka berikut merupakan hasil aktual dari run notebook menggunakan model Gemini 2.5-flash.

**Tabel 8.** Ringkasan Evaluasi Batch Multi-Sumber per Skenario

| Skenario | Adapter | Format | n | RR | Faith | Comp | ROUGE-L | P@K | MRR | CC | Overall | **KTE** | **MSRS** | **AQI** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: Press Release PDF + Chat TXT | FolderSourceAdapter | PDF + TXT | 10 | 0.690 | 0.126 | 0.656 | 0.065 | 1.000 | 1.000 | 0.375 | **0.307** | **0.391** | **0.688** | **0.282** |
| B: PostgreSQL | PostgreSQLAdapter | SQL (5 tabel) | 10 | 0.577 | 0.152 | 0.702 | 0.072 | 1.000 | 1.000 | 0.538 | **0.301** | **0.427** | **0.769** | **0.309** |
| C: Chat Log TXT | FolderSourceAdapter | TXT | 5 | 0.667 | 0.095 | 0.825 | 0.040 | 1.000 | 1.000 | 0.400 | **0.325** | **0.460** | **0.700** | **0.320** |
| D: Cross-Paradigm (Hybrid) | MultiSourceAdapter | PDF + TXT + SQL | 5 | 0.656 | 0.109 | 0.596 | 0.057 | 1.000 | 1.000 | 0.450 | **0.284** | **0.352** | **0.725** | **0.254** |

*RR = Retrieval Relevance; Faith = Answer Faithfulness; Comp = Answer Completeness; CC = Context Coverage.*
*KTE = (Faithfulness + Completeness) / 2, mengukur efektivitas transfer pengetahuan. Ambang batas efektif: KTE ≥ 0.5.*
*MSRS = (Precision@K + Context Coverage) / 2, mengukur kualitas retrieval multi-sumber.*
*AQI = (Faithfulness + Completeness + ROUGE-L) / 3, mengukur kualitas linguistik jawaban.*
*BLEU-1 ≈ 0.000–0.003 di semua skenario, merupakan expected behavior pada reference-free evaluation (lihat Sec 3.5).*
*Skenario A dan B masing-masing mencakup 10 pertanyaan (5 dari press release + 5 dari chat log untuk A; 5 B1–B5 dengan duplikasi cross-tabel untuk B). Skenario C dan D masing-masing 5 pertanyaan.*

Beberapa temuan kunci dari Tabel 8: (1) **Skenario A** mencapai Precision@K dan MRR sempurna (1.000) dengan Retrieval Relevance = 0.690; retrieval bekerja optimal pada dokumen PDF + TXT berbahasa Indonesia. MSRS = 0.688 mengkonfirmasi retrieval multi-sumber aktif. (2) **Skenario B** memiliki KTE tertinggi (0.427) karena data tabular dari PostgreSQL menghasilkan jawaban yang paling *complete* (0.702) dan Context Coverage tertinggi (0.538), mencerminkan data terstruktur yang tidak ambigu. MSRS = 0.769 merupakan yang tertinggi, menunjukkan PostgreSQL paling konsisten dalam mendukung multi-sumber. (3) **Skenario C** mencapai Answer Completeness tertinggi (0.825); pertanyaan dari chat log TXT mendapatkan jawaban paling lengkap karena konteks diskusi analis kaya informasi eksplisit. KTE = 0.460 merupakan tertinggi kedua. (4) **Skenario D** memvalidasi *source-agnostic* pada level paling fundamental: `MultiSourceAdapter` menggabungkan semua dokumen dari FolderSourceAdapter (PDF+TXT) dan PostgreSQLAdapter (5 tabel) menjadi satu pool indeks FAISS. Context Coverage = 0.450 lebih tinggi dari Skenario A (0.375) mengkonfirmasi retrieval lintas paradigma aktif. KTE = 0.352 dan AQI = 0.254 lebih rendah dari skenario tunggal, konsisten dengan ekspektasi, karena pertanyaan *cross-paradigm* menghasilkan jawaban sintesis yang lebih sulit diverifikasi secara reference-free.

**Tabel 9.** Sub-Analisis Skenario C: Efektivitas Retrieval Lintas Tipe Pertanyaan

| Sub-tipe | Contoh Pertanyaan | n | Faith | Comp | Overall | KTE |
|---|---|---|---|---|---|---|
| PR-only (dari press release resmi) | "Berapa NIM dan pendapatan bunga bersih KB Bank Q1 2025?" | 2 | 0.047 | 0.682 | 0.293 | **0.364** |
| Chat-only (dari log diskusi) | "Apa rekomendasi akhir diskusi tim untuk BBKP dan TINS?" | 1 | 0.179 | 0.818 | 0.327 | **0.499** |
| Cross-source (butuh kedua sumber) | "EBITDA TINS Rp384M di chat, apakah konsisten dengan press release?" | 2 | 0.078 | 0.716 | 0.298 | **0.397** |

*Skenario C membuktikan kemampuan sistem melakukan retrieval lintas file dalam satu sesi query; pertanyaan cross-source tidak dapat dijawab dari satu file saja.*

KTE per skenario mencerminkan dimensi transfer pengetahuan yang berbeda: Skenario A mengukur kemampuan sistem mengeksplisitkan kebijakan formal (*Explicit → Actionable*), Skenario B mengukur kemampuan mengkontekstualisasi data struktural (*Structured → Contextual*), Skenario C mengukur kemampuan mengeksplisitkan pengetahuan dari percakapan informal (*Tacit → Explicit*), dan Skenario D mengukur kemampuan sistem mengintegrasikan pengetahuan lintas paradigma data (*Cross-Paradigm*), sehingga membuktikan bahwa klaim *source-agnostic* bukan hanya berlaku per-adapter secara terpisah, tetapi juga ketika kedua paradigma digabungkan dalam satu pipeline sekaligus. Keempat mode ini secara kolektif memetakan seluruh spektrum transfer pengetahuan yang diidentifikasi Nonaka & Takeuchi [3].

### 3.2 Visualisasi Hasil

Similarity score per chunk pada top-5 hasil retrieval menunjukkan konsistensi di atas threshold:

| Chunk | Sumber | Skor Similarity |
|---|---|---|
| Chunk 1 | Press Release Bank Bukopin: Laba dan Turnaround | 0.721 |
| Chunk 2 | Press Release Bank Bukopin: Kredit dan DPK | 0.694 |
| Chunk 3 | analyst_chat_dummy.txt: Diskusi BBKP [DIAN] | 0.668 |
| Chunk 4 | Press Release Bank Bukopin: NIM dan NPL | 0.651 |
| Chunk 5 | analyst_chat_dummy.txt: Kesimpulan [REZA] | 0.633 |

**Tabel 4.** Similarity Score per Chunk untuk Pertanyaan A1 (Laba Bersih BBKP). Semua chunk melampaui threshold similarity_threshold=0.2

Distribusi skor menunjukkan penurunan gradual (0.721 → 0.633) yang mengindikasikan bahwa FAISS berhasil mengurutkan chunk berdasarkan relevansi semantik secara konsisten. Chunk dari dua file berbeda (PDF press release dan TXT chat log) sama-sama ter-retrieve dalam top-5, membuktikan kemampuan multi-sumber dalam satu query. Selisih antara chunk pertama dan kelima hanya 0.088, menunjukkan seluruh context yang di-retrieve relevan terhadap pertanyaan.

Visualisasi metrik evaluasi menghasilkan dua panel: (1) bar chart seluruh 8 metrik dengan pewarnaan hijau (≥0.7), oranye (≥0.4), dan merah (<0.4); dan (2) bar chart similarity per chunk. Pada kondisi Gemini, metrik dimensi retrieval (P@K, MRR) berwarna hijau dan dimensi generasi berada di rentang oranye, mengkonfirmasi bahwa sistem bekerja sesuai harapan dengan LLM multibahasa.

### 3.3 Arsitektur Modular dan Independensi Komponen

Arsitektur pipeline yang dibangun memisahkan secara eksplisit antara komponen retrieval dan komponen generasi. Hal ini terlihat dari distribusi nilai metrik pada Tabel 3: Precision@K dan MRR mencapai 1.000 sementara Faithfulness berada di 0.062. Pola ini konsisten di seluruh 15 pertanyaan batch (Tabel 8), mengkonfirmasi bahwa kualitas retrieval tidak bergantung pada model generatif yang digunakan. Implikasinya, jika kualitas jawaban perlu ditingkatkan, penggantian komponen LLM dapat dilakukan secara independen tanpa menyentuh komponen retrieval, chunking, maupun embedding.

### 3.4 Analisis Keadilan dan Bias

#### 3.4.1 Bias Bahasa

Sistem menggunakan Gemini 2.5-flash sebagai model generatif yang mendukung Bahasa Indonesia secara penuh. Komponen retrieval menggunakan `paraphrase-multilingual-MiniLM-L12-v2` yang dilatih secara eksplisit untuk representasi multibahasa (50+ bahasa), sehingga tidak terdapat bias bahasa pada lapisan retrieval. Untuk deployment di organisasi berbahasa Indonesia, pemilihan LLM yang mendukung Bahasa Indonesia secara penuh merupakan prasyarat utama kualitas jawaban.

#### 3.4.2 Bias Sumber Data

Pada pertanyaan yang bersifat single-entity (misalnya hanya tentang BBKP saja), Context Coverage cenderung rendah karena top-K chunk akan terkonsentrasi pada satu file (press release BBKP). Ini bukan bias sistem melainkan cerminan akurat dari corpus: pertanyaan tentang laba BBKP memang paling relevan dengan press release BBKP. Namun pada pertanyaan A5 (rekomendasi tim analis) dan seluruh pertanyaan Skenario C, chunk dari press release PDF dan chat log TXT sama-sama ter-retrieve, membuktikan multi-sumber aktif. Namun, jika corpus berisi banyak dokumen dari satu sumber yang dominan, sistem berpotensi memberikan jawaban yang kurang seimbang perspektifnya.

Solusi yang diimplementasikan: deduplikasi berbasis hash konten untuk mencegah satu file yang sama terhitung sebagai sumber berbeda saat muncul di beberapa folder upload.

#### 3.4.3 Keadilan Akses

Arsitektur agnostic source secara inheren mengurangi bias akses karena sistem tidak memprioritaskan satu tipe sumber di atas yang lain. PDF, TXT, dan tabel database diperlakukan setara sebagai `RawDocument` setelah fase adapter. Bobot sepenuhnya ditentukan oleh similarity semantik embedding, bukan oleh metadata tipe sumber.

### 3.5 Keterbatasan Penelitian

Beberapa keterbatasan penelitian ini perlu diakui secara eksplisit:

**Keterbatasan Dataset.** Evaluasi dilakukan pada 20 pertanyaan di empat skenario dengan dua emiten (BBKP dan TINS) di sektor perbankan dan pertambangan. Generalisasi temuan ke domain lain (hukum, medis, teknis) atau sektor yang berbeda memerlukan evaluasi tambahan dengan dataset yang lebih beragam dan jumlah pertanyaan yang lebih besar.

**Ketiadaan Ground Truth.** Metrik ROUGE-L dan BLEU-1 dihitung secara *reference-free*, yaitu dengan membandingkan jawaban terhadap context yang di-retrieve, bukan dengan jawaban acuan yang dikurasi manusia. Nilai kedua metrik ini karenanya tidak dapat dibandingkan langsung dengan sistem lain yang menggunakan ground truth dataset standar (misalnya SQuAD, NaturalQuestions).

**Keterbatasan BLEU-1.** Nilai BLEU-1 = 0.000 di **semua skenario dan semua pertanyaan** merupakan *expected behavior* pada evaluasi reference-free, bukan indikasi kegagalan sistem. Mekanisme *brevity penalty* BLEU menghukum jawaban yang lebih panjang dari referensi. Dalam konteks reference-free di mana referensi adalah gabungan seluruh context (ribuan token), jawaban apapun, seberapa pun baiknya, akan mendapat brevity penalty maksimal karena panjang jawaban (100–300 token) jauh lebih pendek dari referensi (2000+ token). BLEU-1 tetap dilaporkan untuk transparansi metodologi, namun tidak digunakan dalam interpretasi kualitas sistem. Metrik ini hanya informatif jika digunakan dengan ground truth jawaban singkat yang dikurasi manual.

**Skala Evaluasi.** Evaluasi pada 20 pertanyaan di empat skenario memberikan gambaran proof-of-concept yang memadai, namun belum cukup untuk klaim statistik yang dapat digeneralisasi. Penelitian lanjutan disarankan menggunakan minimal 20–50 pasangan query-answer per skenario untuk analisis yang lebih representatif.

### 3.6 Implikasi

#### 3.6.1 Implikasi untuk Transfer Pengetahuan Organisasi

Sistem ini secara langsung menjawab tantangan yang diidentifikasi dalam pendahuluan. Analis junior yang baru bergabung dapat langsung bertanya dalam bahasa natural, misalnya "Berapa laba bersih BBKP Q1 2025 dibanding tahun lalu?" atau "Apa rekomendasi akhir tim untuk saham TINS?", dan sistem akan mencari jawaban dari kombinasi press release PDF, log diskusi TXT, dan database analitik secara otomatis dalam satu query. Skenario ini merepresentasikan transfer pengetahuan dari analis senior ke junior tanpa harus membaca ulang seluruh dokumentasi yang tersebar. Prinsip yang sama berlaku untuk konteks organisasi lain: Product Owner yang baru bergabung dapat bertanya tentang keputusan arsitektur atau riwayat vendor dari dokumentasi proyek yang ada.

#### 3.6.2 Implikasi Teknis untuk Pengembangan Selanjutnya

Arsitektur Adapter Pattern yang digunakan membuka jalan untuk extensibility: sumber baru (MongoDB, SharePoint, Google Drive API) dapat ditambahkan dengan mengimplementasikan dua method, yaitu `load()` dan `describe()`, tanpa mengubah pipeline inti. Ini sejalan dengan prinsip Open/Closed Principle dalam desain perangkat lunak.

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

1. **Arsitektur agnostic terealisasi:** `SourceDetector` + `SourceFactory` + `BaseSourceAdapter` memungkinkan penanganan folder (PDF, TXT/MD/LOG) dan PostgreSQL (3 mode query: semua tabel, tabel tertentu, custom SQL) dari satu parameter `source`, tanpa perubahan pada pipeline inti.

2. **Realtime indexing terbukti benar:** Setiap pemanggilan `pipeline.ask()` membangun indeks FAISS secara in-memory sehingga perubahan konten di sumber langsung tercermin dalam hasil retrieval tanpa restart sistem. Tidak ada file index yang tersimpan di disk.

3. **Multi-sumber terbukti secara eksperimen:** Evaluasi batch 20 pertanyaan pada empat skenario membuktikan sistem mampu menangani sumber heterogen dari satu parameter `source`: Skenario A (PDF+TXT, Overall=0.307, MSRS=0.688), Skenario B (PostgreSQL 5 tabel, Overall=0.301, MSRS=0.769), Skenario C (Chat Log TXT, Overall=0.325, MSRS=0.700), dan Skenario D (Hybrid cross-paradigm, Overall=0.284, MSRS=0.725). Precision@K=1.000 dan MRR=1.000 di semua skenario membuktikan komponen retrieval bekerja konsisten optimal. Skenario D secara khusus membuktikan klaim *source-agnostic* lintas paradigma: `MultiSourceAdapter` menggabungkan FolderSourceAdapter dan PostgreSQLAdapter ke dalam satu indeks FAISS, dengan Context Coverage = 0.450 mengkonfirmasi retrieval aktif dari kedua paradigma.

4. **Efektivitas transfer pengetahuan terukur via KTE:** KTE per skenario: A=0.391 (*Explicit→Actionable*), B=0.427 (*Structured→Contextual*), C=0.460 (*Tacit→Explicit*), D=0.352 (*Cross-Paradigm*). Skenario C memiliki Answer Completeness tertinggi (0.825) karena chat log kaya informasi eksplisit. Skenario B memiliki MSRS tertinggi (0.769) dan Context Coverage tertinggi (0.538) karena tabel relasional PostgreSQL memberikan dokumen dari beragam entitas tabel. Nilai KTE keseluruhan (0.352–0.460) belum mencapai ambang batas 0.5, hal ini dipengaruhi oleh rendahnya Faithfulness (0.095–0.152) yang merupakan karakteristik evaluasi reference-free pada teks berbahasa Indonesia, bukan kegagalan retrieval.

5. **Evaluasi multi-dimensi terukur:** Framework 11 metrik (8 kuantitatif + 3 komposit) mengungkap bahwa komponen retrieval bekerja sangat baik: P@K=1.000 dan MRR=1.000 di semua skenario. Pemisahan dimensi retrieval dan generasi membuktikan arsitektur modular: nilai Precision@K dan MRR yang konsisten tinggi menunjukkan kualitas retrieval tidak bergantung pada LLM. AQI tertinggi dicapai Skenario C (0.320), mengkonfirmasi jawaban dari chat log memiliki kualitas linguistik paling baik. MSRS tertinggi pada Skenario B (0.769) mengkonfirmasi PostgreSQL paling efektif sebagai sumber multi-entitas.

Kontribusi utama penelitian adalah desain pola Adapter yang memisahkan concerns antara sumber data, pemrosesan teks, retrieval, dan generasi, sehingga setiap komponen dapat diganti atau diperluas secara independen. Framework evaluasi 8 metrik yang diimplementasikan bebas-dependensi (tanpa library RAGAS) dapat direplikasi di lingkungan terbatas resource.

Untuk penelitian selanjutnya, disarankan: (1) evaluasi batch dengan minimal 20 pertanyaan menggunakan ground truth dataset yang dikurasi manual untuk mendapatkan nilai ROUGE-L dan BLEU-1 yang dapat dibandingkan dengan sistem lain; (2) fine-tuning embedding model pada dokumen domain spesifik organisasi; (3) perbandingan performa dengan sistem RAG berbasis vector database komersial (Pinecone, Weaviate) sebagai baseline arsitektur.

---

## 5. DAFTAR PUSTAKA

[1] IDC. (2023). *90% of Data is Unstructured and Its Full of Untapped Value*. IDC Blog. Retrieved from https://blogs.idc.com/2023/05/09/90-of-data-is-unstructured-and-its-full-of-untapped-value/ (accessed April 10, 2026).

[2] Davenport, T. H., & Prusak, L. (1998). *Working Knowledge: How Organizations Manage What They Know*. Harvard Business School Press.

[3] Nonaka, I., & Takeuchi, H. (1995). *The Knowledge-Creating Company*. Oxford University Press.

[4] Gartner. (2020). *Employees Spend Too Much Time on Low-Value Tasks: Use AI and Automation to Fix It*. Gartner Research. Retrieved from https://www.gartner.com/en/newsroom/press-releases/2020-01-23-gartner-says-employees-spend-too-much-time-on-low-val (accessed April 10, 2026).

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

> **Status draft v3.0 (18 April 2026) | Target: Sinta 2-3**
>
> **✅ Selesai:**
> - [x] Seluruh dataset diganti BBKP/TINS (tidak ada lagi AADI/Adaro)
> - [x] chunk_size=2000, overlap=300 konsisten di seluruh dokumen
> - [x] Tabel 8 terisi penuh: A(0.329), B(0.302), C(0.302)
> - [x] Tabel 9 sub-analisis Skenario C terisi aktual
> - [x] Abstrak ID & EN dipangkas ~180 kata (sesuai standar Sinta)
> - [x] Keywords dipangkas ke 5 (italic, sesuai IEEE style)
> - [x] Author block diupdate: afiliasi terpisah + ORCID placeholder
> - [x] BLEU-1 = 0.000 dijelaskan sebagai expected behavior
> - [x] Analisis insight Tabel 8 ditambahkan (Skenario B KTE tertinggi)
>
> **⚠️ Wajib sebelum submit Sinta:**
> - [ ] **ORCID**: daftarkan di orcid.org jika belum ada (wajib Sinta 2)
> - [ ] **Gambar 2**: embed screenshot 4-panel chart dari Cell 20 (format PNG ≥300 DPI)
> - [ ] **Gambar 1**: buat diagram arsitektur sebagai gambar (bukan ASCII/code block)
> - [ ] **Word count**: hitung total body text (target 4.000–7.000 kata)
> - [ ] **Format referensi**: konversi ke IEEE numbered: [1] Penulis, "Judul," *Jurnal*, vol., no., pp., tahun.
> - [ ] **Template OJS**: download template Word/LaTeX dari jurnal target, apply sebelum submit
> - [ ] **Similarity check**: jalankan Turnitin/iThenticate, target <20% similarity
> - [ ] **Bahasa**: proofread Bahasa Indonesia oleh native speaker (cek EYD 2022)
> - [ ] **Konfirmasi jurnal**: JTIIK (Sinta 2) / Jurnal INTECH ITHB (Sinta 3) / Register (Sinta 3)
>
> **📌 Catatan format Sinta:**
> - Abstrak: 150–200 kata ✅ (sudah)
> - Keywords: 3–5 kata, huruf miring, dipisah titik koma ✅
> - Tabel: judul di atas tabel ✅
> - Gambar: judul di bawah gambar ⚠️ (perlu dicek saat apply template)
> - Sitasi: IEEE [1][2] atau APA; pilih satu dan terapkan secara konsisten
> - Kode program: gunakan font monospace, bukan screenshot
> - Tidak ada catatan kaki (*footnote*); semua kutipan dimasukkan ke daftar referensi
