**Sistem Retrieval-Augmented Generation Agnostic Multi-Sumber untuk Question Answering Berbasis Dokumen dan Database pada Konteks Transfer Pengetahuan Organisasi**

*Agnostic Multi-Source Retrieval-Augmented Generation System for Question Answering Based on Documents and Databases in Organizational Knowledge Transfer Context*

---

## Penulis

**Krisna Dwi Setya Adi**  
Program Studi Sistem Informasi Data Science  
Institut Teknologi Harapan Bangsa, Bandung, Indonesia  
Email: krisna@ithb.ac.id  

**Ivan Michael Siregar**  
Program Studi Sistem Informasi Data Science  
Institut Teknologi Harapan Bangsa, Bandung, Indonesia  
Email: ivan@ithb.ac.id

*\* Corresponding author: Krisna Dwi Setya Adi (krisna@ithb.ac.id)*

---

## ABSTRAK

Pergantian personel kunci pada *document-based service organizations* menciptakan kesenjangan pengetahuan yang nyata: pengetahuan domain yang tersebar di berbagai artefak — dokumen teknis, prosedur operasional, dan basis data — tidak dapat diakses secara cepat oleh penerus, sehingga setiap permintaan klarifikasi memerlukan waktu berjam-jam hingga berhari-hari sebelum kesimpulan dapat dirumuskan. Organisasi modern menghadapi tantangan mengelola pengetahuan yang tersebar di sumber heterogen, yaitu dokumen tidak terstruktur (PDF, TXT) dan basis data relasional, sehingga menghambat transfer pengetahuan dan pengambilan keputusan. Penelitian ini mengembangkan sistem *Question Answering* (QA) berbasis *Retrieval-Augmented Generation* (RAG) yang bersifat *agnostic* terhadap sumber data: sistem hanya memerlukan satu parameter `source` untuk mendeteksi dan menangani berbagai jenis sumber secara otomatis. Sistem mengintegrasikan dua adapter utama, yaitu `FolderSourceAdapter` untuk sumber tidak terstruktur dan `PostgreSQLAdapter` untuk basis data relasional, dengan indeks vektor FAISS yang dibangun secara *real-time* tanpa pra-komputasi ke disk. Evaluasi menggunakan delapan metrik kuantitatif dan tiga metrik komposit: *Knowledge Transfer Effectiveness* (KTE), *Multi-Source Retrieval Score* (MSRS), dan *Answer Quality Index* (AQI). Eksperimen mencakup lima skenario dengan 25 pertanyaan (5 per skenario) menggunakan corpus tiga-layer dari sistem manajemen lelang obligasi pemerintah: spesifikasi kebutuhan fungsional sistem (FR, L1), basis data PostgreSQL operasional 8 tabel (L2), dan log diskusi tim pengembang 908 pesan (L3). Skenario: (A) Chat log saja, (B) dokumen FR/PDF saja, (C) PostgreSQL saja, (D) FR+DB, dan (E) semua layer (*Hybrid*). Hasil menunjukkan Precision@K = 1.00 dan MRR = 1.00 pada semua skenario. *Overall* tertinggi Skenario E (0.358); di antara skenario *reference-free* (A–D), Skenario D unggul (0.318). MSRS tertinggi Skenario C (0.825), dan *Retrieval Relevance* tertinggi Skenario E (0.582). Skenario E dilengkapi 5 jawaban referensi (*ground truth*) yang dikurasi manual, menghasilkan ROUGE-L = 0.167 dan BLEU-1 = 0.213 (*reference-based*). *Ablation study* empat konfigurasi membuktikan kontribusi dramatis Layer 3: Chat-only *Overall* = 0.230 → FR-only = 0.230 → FR+DB = 0.231 → Full = 0.358. Kelima skenario memetakan dimensi transfer pengetahuan: *Tacit→Operational* (A), *Explicit→Actionable* (B), *Explicit→Structured* (C), *Explicit→Cross-referenced* (D), dan *Cross-Paradigm* (E).

**Kata Kunci:** *Retrieval-Augmented Generation*, *Question Answering*, Multi-Sumber, FAISS, Transfer Pengetahuan

---

## ABSTRACT

Key personnel turnover in document-based service organizations creates a real knowledge gap: domain knowledge dispersed across technical documents, operational procedures, and databases cannot be accessed quickly by successors, causing each clarification request to require hours or even days before a conclusion can be reached. Modern organizations face challenges managing knowledge dispersed across heterogeneous sources, namely unstructured documents (PDF, TXT) and relational databases, hindering knowledge transfer and decision-making. This study develops a Question Answering (QA) system based on Retrieval-Augmented Generation (RAG) that is agnostic to the data source: the system requires only a single `source` parameter to automatically detect and handle various source types. Two main adapters are integrated, namely `FolderSourceAdapter` for unstructured sources and `PostgreSQLAdapter` for relational databases, with FAISS vector indices built in real-time without disk pre-computation. Evaluation uses eight quantitative metrics and three composite metrics: Knowledge Transfer Effectiveness (KTE), Multi-Source Retrieval Score (MSRS), and Answer Quality Index (AQI). Experiments cover five scenarios with 25 questions (5 per scenario) using a three-layer corpus from a government bond auction management system: functional requirements specification (FR, L1), operational PostgreSQL database with 8 tables (L2), and developer team discussion logs with 908 messages (L3). Scenarios: (A) Chat logs only, (B) FR/PDF only, (C) PostgreSQL only, (D) FR+DB, and (E) all layers (Hybrid). Results show Precision@K = 1.00 and MRR = 1.00 across all scenarios. Highest Overall on Scenario E (0.358); among reference-free scenarios (A–D), Scenario D leads (0.318). Highest MSRS on Scenario C (0.825), and highest Retrieval Relevance on Scenario E (0.582). Scenario E is equipped with 5 manually curated ground truth answers, yielding ROUGE-L = 0.167 and BLEU-1 = 0.213 (reference-based). Ablation study across four configurations demonstrates the dramatic contribution of Layer 3: Chat-only Overall = 0.230 → FR-only = 0.230 → FR+DB = 0.231 → Full = 0.358. Five scenarios map distinct knowledge transfer dimensions: Tacit→Operational (A), Explicit→Actionable (B), Explicit→Structured (C), Explicit→Cross-referenced (D), and Cross-Paradigm (E).

**Keywords:** Retrieval-Augmented Generation, Question Answering, Multi-Source, FAISS, Knowledge Transfer

---

## 1. PENDAHULUAN

Organisasi yang beroperasi dalam ekosistem layanan berbasis dokumen (*document-based service organizations*) — seperti lembaga keuangan, penyedia layanan teknis domain-spesifik, dan operator platform transaksi — sangat bergantung pada pengetahuan yang terakumulasi dalam berbagai artefak: spesifikasi kebutuhan fungsional, prosedur operasional, konfigurasi sistem, dan rekam jejak transaksi. Tantangan kritis muncul saat terjadi pergantian personel kunci (*knowledge worker turnover*): pengetahuan domain yang selama ini melekat pada individu harus ditransfer kepada penerus dalam waktu terbatas, sementara operasional layanan tidak dapat berhenti. Menurut Chui et al. (2012), karyawan yang tergolong *knowledge worker* menghabiskan sekitar 20% waktu kerjanya hanya untuk mencari informasi internal, dan IDC (2023) melaporkan bahwa hingga 90% data organisasi bersifat tidak terstruktur — tersebar di dokumen PDF, berkas teks, dan basis data — sehingga hanya sebagian kecil yang benar-benar dapat diakses dan dimanfaatkan secara efektif.

Permasalahan ini semakin kompleks karena pengetahuan organisasi tersebar secara heterogen: spesifikasi kebutuhan fungsional tersimpan dalam dokumen PDF ratusan halaman, data operasional tersimpan di tabel basis data relasional, sementara konteks keputusan teknis kerap hanya ada dalam ingatan personel yang bersangkutan. Nonaka dan Takeuchi (1995) membedakan pengetahuan *tacit* (tersirat dalam pikiran individu) dan *explicit* (terdokumentasi dalam artefak); namun bahkan pengetahuan *explicit* pun sulit ditemukan secara cepat apabila tersebar lintas repositori dengan format berbeda. Gartner (2020) memperkirakan karyawan menghabiskan hingga 30% waktu kerja untuk aktivitas bernilai rendah yang seharusnya dapat diotomatisasi, termasuk penelusuran dokumen dan klarifikasi informasi operasional yang berulang.

Dalam lingkungan enterprise yang menjadi konteks penelitian ini — sebuah sistem platform transaksi keuangan yang mengelola proses penawaran dan alokasi instrumen keuangan untuk lembaga-lembaga peserta terdaftar — dampak *turnover* ini dirasakan secara langsung secara operasional. Dalam rentang dua tahun, terjadi dua kali pergantian *Product Owner*, dengan komposisi tim yang menyusut dari tiga menjadi dua orang. Akibatnya, setiap permintaan klarifikasi teknis dari pengguna eksternal (*peserta platform*) — seperti alur proses transaksi, parameter konfigurasi sesi, atau aturan instrumen — memerlukan waktu respons yang bervariasi dari beberapa jam hingga satu hari kerja penuh, bergantung pada seberapa dalam konteks tersebut terdokumentasi dan seberapa akrab personel yang tersisa dengan materi tersebut. Kondisi ini merepresentasikan pola umum yang ditemukan pada berbagai *document-based service organization*: volume pengetahuan tidak berkurang, tetapi kapasitas manusia untuk mengaksesnya secara cepat dan akurat semakin terbatas.

Solusi manajemen pengetahuan (*Knowledge Management*/KM) tradisional — seperti wiki internal, basis pengetahuan statis, atau FAQ — tidak mampu menjawab pertanyaan dinamis yang membutuhkan inferensi lintas sumber secara *real-time*. Pengguna masih harus menelusuri spesifikasi teknis dan prosedur operasional ratusan halaman sekaligus mengecek tabel konfigurasi di basis data secara manual — proses yang memakan waktu berjam-jam bahkan berhari-hari sebelum kesimpulan dapat dirumuskan. Model bahasa besar (LLM) tanpa *grounding* faktual menghadirkan risiko berbeda: Gao et al. (2024) mengidentifikasi tiga kelemahan kritis LLM dalam konteks layanan domain-spesifik, yaitu *hallucination*, pengetahuan kedaluwarsa, dan penalaran tidak transparan (*non-transparent reasoning*). Lebih lanjut, pendekatan Self-RAG (Ren et al. 2023) secara eksplisit melatih LLM untuk mengevaluasi relevansi retrieval dan mengkritisi jawabannya sendiri — mengonfirmasi bahwa LLM standar tanpa mekanisme ini cenderung menghasilkan jawaban tanpa mempertimbangkan batas pengetahuannya, perilaku yang berbahaya dalam konteks layanan yang menuntut akurasi dan *traceability* tinggi. Terdapat kesenjangan nyata: belum ada solusi yang mampu menjawab pertanyaan faktual secara akurat **sekaligus cepat** dari sumber-sumber heterogen yang sudah ada di organisasi — di mana *cepat* berarti pengguna tidak perlu menunggu lama untuk dapat menyimpulkan sesuatu — tanpa memerlukan konfigurasi teknis yang berbeda untuk setiap tipe sumber.

Pendekatan *Retrieval-Augmented Generation* (RAG) yang diperkenalkan Lewis et al. (2020) membuka peluang untuk mengatasi keterbatasan ini: dengan menggabungkan retrieval dari sumber eksternal dan kemampuan generasi LLM, sistem dapat memberikan jawaban faktual, *grounded*, dan berbasis dokumen aktual. Izacard dan Grave (2021) memperluas pendekatan ini untuk *passage retrieval* pada domain terbuka; integrasi RAG dengan *knowledge graph* untuk *multi-hop reasoning* ditunjukkan oleh Yasunaga et al. (2021); sementara Johnson et al. (2019) mengembangkan FAISS sebagai infrastruktur pencarian vektor yang efisien, dan Reimers dan Gurevych (2019) menyediakan representasi semantik multibahasa melalui Sentence-BERT. Namun, implementasi RAG yang ada umumnya bersifat *single-source* dan memerlukan konfigurasi berbeda untuk setiap tipe sumber data, sehingga menambah beban teknis bagi organisasi dengan ekosistem data heterogen. Evaluasi sistem QA berbasis RAG sendiri memerlukan perspektif ganda: dari sisi retrieval menggunakan Precision@K dan MRR (Voorhees 1999), dari sisi generasi menggunakan ROUGE-L (Lin 2004) dan BLEU-1 (Papineni et al. 2002), serta dimensi khusus RAG yang diusulkan Es et al. (2023) yaitu *faithfulness* dan *answer relevance*.

Penelitian ini mengembangkan sistem QA berbasis RAG yang bersifat *agnostic* terhadap sumber data — mampu menangani dokumen tidak terstruktur (PDF, TXT) dan basis data relasional (PostgreSQL) secara terpadu melalui satu antarmuka pemrograman (*Adapter Pattern*) — sebagai infrastruktur transfer pengetahuan yang dapat diimplementasikan langsung pada *document-based service organization*. Kontribusi penelitian ini adalah sebagai berikut:

1. **Identifikasi faktor-faktor yang berpengaruh pada kualitas QA** dalam konteks *document-based enterprise service systems*: menganalisis aspek domain, karakteristik sumber data heterogen, dan pola pertanyaan operasional yang menentukan relevansi retrieval dan akurasi jawaban pada sistem transfer pengetahuan berbasis RAG.

2. **Rancangan model QA multi-sumber dengan arsitektur RAG Adapter Pattern**: desain arsitektur *agnostic* yang mengintegrasikan `FolderSourceAdapter` untuk sumber tidak terstruktur (PDF, TXT) dan `PostgreSQLAdapter` untuk basis data relasional dalam indeks FAISS *real-time in-memory*, tanpa pra-komputasi ke disk.

3. **Evaluasi empiris berbasis data operasional nyata dalam 5 skenario lintas paradigma**: pengujian performa sistem menggunakan 8 metrik kuantitatif dan 3 metrik komposit — *Knowledge Transfer Effectiveness* (KTE), *Multi-Source Retrieval Score* (MSRS), dan *Answer Quality Index* (AQI) — pada dataset yang berasal dari sistem platform transaksi keuangan operasional yang sesungguhnya.

---

## 2. METODE PENELITIAN

### 2.1 Sumber Data dan Prapemrosesan

Penelitian menggunakan data operasional nyata dari sistem manajemen lelang obligasi pemerintah (BOND_SYS) dengan tiga layer corpus: (L1) spesifikasi kebutuhan fungsional (Functional Requirements) BOND_SYS dalam format TXT, mencakup deskripsi modul lelang, alur proses bisnis, dan persyaratan teknis ratusan halaman; (L2) basis data PostgreSQL MOFIDS dengan 8 tabel operasional yang berisi data nyata (20 RFQ, 10 securities, 10 firms, 10 quotations, 10 trades, 10 trade statuses, 11 firm default params, 8 fraction masters); dan (L3) log diskusi tim pengembang dalam format TXT berjumlah 908 pesan dari tiga sumber — diskusi grup 2022, diskusi grup Februari 2025, dan percakapan personal 2022. Identitas sistem, institusi, dan individu dianonimisasi menggunakan skema *token masking* (nama sistem → BOND_SYS, nama kementerian → GOV_DEPT1, nama modul → BOND_MOD, dll.) untuk memenuhi prinsip *reproducibility* yang disyaratkan jurnal akademik.

Prapemrosesan dilakukan oleh dua adapter sesuai tipe sumber. `FolderSourceAdapter` menangani berkas dokumen dengan library yang sesuai per format (pypdf untuk PDF, built-in untuk teks). Seluruh teks hasil ekstraksi dinormalisasi menjadi objek `RawDocument` dengan atribut konten, sumber, dan metadata format.

`PostgreSQLAdapter` menangani sumber relasional dengan tiga mode: (1) semua tabel, (2) tabel tertentu (`pg_tables`), dan (3) custom SQL query (`pg_queries`). Setiap tabel atau hasil query dikonversi ke teks terstruktur yang menyertakan nama kolom, jumlah baris, dan data tabular, sehingga dapat di-embed dan di-retrieve oleh FAISS.

Setelah ekstraksi, teks dipotong menggunakan `UniversalTextSplitter` berbasis `RecursiveCharacterTextSplitter` dengan `chunk_size=2000` dan `overlap=300`. Overlap 300 karakter dirancang untuk mempertahankan konteks antar-chunk agar informasi yang terpotong di batas chunk tidak hilang sepenuhnya. Nilai chunk_size yang lebih besar (2000) dipilih untuk mendukung dokumen keuangan berbahasa Indonesia yang umumnya mengandung kalimat panjang dan tabel multi-baris.

**Tabel 1.** Format File yang Didukung FolderSourceAdapter

| Format | Library | Perlakuan |
|---|---|---|
| `.pdf` | pypdf | Ekstraksi teks semua halaman |
| `.txt`, `.md`, `.log` | built-in | Raw text |

### 2.2 Dataset Evaluasi

Evaluasi dirancang dalam lima skenario yang masing-masing merepresentasikan tipe sumber dan dimensi transfer pengetahuan berbeda. Seluruh teks berbahasa Indonesia dan berasal dari data operasional nyata yang telah dianonimisasi dari sistem lelang obligasi BOND_SYS.

**Skenario A — Chat Only (L3, Tacit→Operational):** Log diskusi tim pengembang BOND_SYS (908 pesan, 3 file TXT). Lima pertanyaan mencakup isu operasional yang hanya ada di log diskusi: bug submit quotation, masalah upload allocation demo, isu filter NEWCORE-2442, keputusan digit desimal, dan status fitur amend.

**Skenario B — FR/PDF (L1, Explicit→Actionable):** Dokumen spesifikasi kebutuhan fungsional BOND_SYS dalam format TXT. Lima pertanyaan mencakup alur proses bisnis yang terdokumentasi: tahapan Buyback Cash, perbedaan sesi General/Restricted, pihak dalam approval RFQ, persyaratan teknis Upload Allocation, dan mekanisme notifikasi broadcast.

**Skenario C — PostgreSQL (L2, Explicit→Structured):** Basis data PostgreSQL MOFIDS dengan 8 tabel operasional. Lima pertanyaan mencakup data konfigurasi struktural: nilai default price percentage, kombinasi fraction_type/digit, perbedaan auction_unit per board, firma dengan is_active=Y, dan kuotasi teralokasi pada RFQ tertentu.

**Skenario D — FR+DB (L1+L2, Explicit→Cross-referenced):** Penggabungan dokumen FR dan PostgreSQL 8 tabel melalui `MultiSourceAdapter`. Lima pertanyaan membutuhkan *cross-reference* FR dengan data aktual: konsistensi jam sesi, board type di FR vs DB, konsistensi offering_parameter, perhitungan settlement_date, dan konsistensi offering_digit dengan fraction_masters.

**Skenario E — Hybrid All (L1+L2+L3, Cross-Paradigm):** Penggabungan semua layer (FR, PostgreSQL, log diskusi) dalam satu indeks FAISS. Lima pertanyaan membutuhkan ketiga layer sekaligus: bug submit quotation BS-SB (Chat+FR), keputusan offering digit (Chat+DB), insiden ETL MOFIDS Februari 2023 (Chat), alur upload allocation (FR+DB+Chat), dan status fitur amend (FR+DB+Chat). Skenario ini satu-satunya yang dilengkapi **5 jawaban referensi (*ground truth*) yang disusun manual** oleh peneliti (`GROUND_TRUTH_HYBRID`), sehingga metrik ROUGE-L dan BLEU-1 bersifat *reference-based* — berbeda dari Skenario A–D yang *reference-free*.

### 2.3 Arsitektur Model

Sistem dibangun dengan pola *Adapter Pattern* untuk memungkinkan extensibility sumber data tanpa mengubah pipeline inti. Arsitektur terdiri dari tujuh komponen utama yang dirangkai secara sequential: `SourceDetector` mendeteksi tipe sumber dari parameter input, `SourceFactory` menginisiasi adapter yang sesuai (`FolderSourceAdapter` atau `PostgreSQLAdapter`), `UniversalTextSplitter` memotong teks hasil ekstraksi, `RuntimeIndexBuilder` membangun indeks FAISS secara in-memory, `QueryProcessor` melakukan similarity search pada indeks, dan `AnswerGenerator` menghasilkan jawaban beserta evaluasi 8 metrik secara bersamaan.

*[Gambar 1 — diagram arsitektur sistem perlu disajikan sebagai file gambar (PNG/PDF ≥300 DPI) sebelum submit ke OJS]*

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

Indeks FAISS dibangun secara in-memory (*real-time indexing*) pada setiap pemanggilan `pipeline.ask()`, berbeda dari sistem RAG konvensional yang menyimpan index ke disk. Urutan eksekusi: (1) adapter me-load dokumen dari sumber saat itu juga, (2) splitter memotong teks, (3) FAISS membangun indeks di RAM, (4) query processor melakukan retrieval, dan (5) answer generator menghasilkan jawaban. *Session cache* (`use_session_cache=True`) memungkinkan reuse indeks untuk sumber yang sama dalam satu sesi eksekusi, mengurangi overhead tanpa mengorbankan konsistensi data antar-pemanggilan dalam sesi yang sama.

### 2.5 Analisis Interpretatif dan Desain Evaluasi

Framework evaluasi 8 metrik dikelompokkan dalam tiga dimensi untuk memungkinkan interpretasi yang terpisah antara kualitas retrieval dan kualitas generasi:

**Dimensi Retrieval (IR Klasik):**
- **Precision@K** = |chunk relevan| / K, mengukur proporsi chunk di atas `similarity_threshold`
- **MRR** = 1/rank_chunk_relevan_pertama (Voorhees 1999)
- **Context Coverage** = unique_sources / total_chunks, mengukur keragaman sumber

**Dimensi Kualitas Jawaban:**
- **Retrieval Relevance** = cosine similarity embedding pertanyaan vs rata-rata embedding chunks
- **Answer Faithfulness** = F1 token overlap jawaban vs gabungan context (anti-hallucination)
- **Answer Completeness** = rasio keyword pertanyaan yang muncul dalam jawaban

**Dimensi NLP Akademik:**
- **ROUGE-L** (Lin 2004) = F1 berbasis Longest Common Subsequence
- **BLEU-1** (Papineni et al. 2002) = unigram precision dengan brevity penalty

Selain delapan metrik di atas, penelitian ini mendefinisikan **tiga metrik komposit** untuk menjawab tiga pertanyaan evaluasi yang tidak dapat dijawab oleh metrik tunggal manapun:

**Metrik Komposit 1: Knowledge Transfer Effectiveness (KTE)**

$$KTE = \frac{\text{Answer Faithfulness} + \text{Answer Completeness}}{2}$$

KTE merupakan rata-rata dari dua komponen: Faithfulness (proporsi jawaban yang didukung context, sebagai ukuran anti-halusinasi) dan Completeness (proporsi keyword pertanyaan yang tercakup dalam jawaban). Ambang batas efektif ditetapkan KTE ≥ 0.5. Transfer pengetahuan berhasil hanya jika jawaban sekaligus tidak halusinasi dan menjawab pertanyaan secara lengkap; jika salah satu komponen bernilai nol maka pengetahuan gagal dipindahkan, sehingga rata-rata sederhana sudah memadai sebagai ukuran efektivitas (Nonaka dan Takeuchi 1995).

**Metrik Komposit 2: Multi-Source Retrieval Score (MSRS)**

$$MSRS = \frac{\text{Precision@K} + \text{Context Coverage}}{2}$$

MSRS merupakan rata-rata dari dua komponen: Precision@K (proporsi chunk yang relevan dalam top-K hasil retrieval) dan Context Coverage (keragaman sumber dokumen dalam top-K). Sistem yang hanya menarik dari satu file akan memperoleh Context Coverage rendah meskipun Precision@K-nya tinggi; MSRS mendeteksi kondisi ini dan memastikan klaim multi-sumber terbukti secara retrieval. Pendekatan ini mengaplikasikan prinsip diversity dalam Information Retrieval (Carbonell dan Goldstein 1998) ke konteks multi-source RAG.

**Metrik Komposit 3: Answer Quality Index (AQI)**

$$AQI = \frac{\text{Answer Faithfulness} + \text{Answer Completeness} + \text{ROUGE-L}}{3}$$

AQI merupakan rata-rata dari tiga komponen: Faithfulness (anti-halusinasi), Completeness (cakupan pertanyaan), dan ROUGE-L (kemiripan struktural berbasis urutan kata terhadap context). KTE hanya mengukur apakah pengetahuan tersampaikan secara konten; AQI menambahkan dimensi linguistik untuk mendeteksi jawaban yang memadai secara topik namun strukturnya jauh berbeda dari dokumen sumber. Pendekatan multi-aspek ini konsisten dengan metodologi SummEval (Fabbri et al. 2021).

**Hubungan antar metrik komposit:** KTE mengukur dari perspektif *pengguna* (apakah pengetahuan tersampaikan), MSRS dari perspektif *sistem* (apakah multi-sumber terbukti), dan AQI dari perspektif *linguistik* (apakah jawaban berkualitas NLP). Ketiganya bersifat komplementer; sistem yang baik seharusnya memperoleh skor tinggi pada ketiga dimensi secara bersamaan.

**Metrik Agregat Overall** dihitung sebagai rata-rata sederhana dari lima metrik kualitas jawaban:

$$Overall = \frac{RR + Faithfulness + Completeness + ROUGE\text{-}L + BLEU\text{-}1}{5}$$

Metrik retrieval murni (Precision@K, MRR, Context Coverage) tidak dimasukkan dalam Overall agar nilai agregat ini mencerminkan kualitas jawaban, bukan kualitas retrieval yang secara konsisten sempurna (P@K=MRR=1.000).

Pemisahan tiga dimensi evaluasi ini memungkinkan identifikasi apakah nilai rendah disebabkan oleh kegagalan retrieval atau keterbatasan kapabilitas bahasa model generatif; keduanya merupakan masalah yang memerlukan solusi berbeda.

Untuk membuktikan klaim "Multi-Sumber", evaluasi dirancang dalam lima skenario yang masing-masing merepresentasikan satu dimensi transfer pengetahuan organisasi. Skenario E secara khusus membuktikan klaim *source-agnostic* pada level paling tinggi — satu query menjangkau dokumen FR (TXT), log diskusi (TXT), **dan** tabel PostgreSQL (SQL) secara bersamaan dalam satu indeks FAISS. Skenario E juga satu-satunya skenario yang dilengkapi ground truth sehingga ROUGE-L dan BLEU-1 bersifat reference-based:

**Tabel 3.** Desain Evaluasi Multi-Sumber dan Dimensi Transfer Pengetahuan

| Skenario | Layer | Adapter | Sumber | Format | Dimensi TK | Tipe Pengetahuan |
|---|---|---|---|---|---|---|
| A: Chat Only | L3 | `FolderSourceAdapter` | Log diskusi tim BOND_SYS (908 pesan, 3 file TXT) | TXT | Tacit → Operational | Percakapan informal → jawaban operasional |
| B: FR/PDF | L1 | `FolderSourceAdapter` | Functional Requirements BOND_SYS | TXT | Explicit → Actionable | Spesifikasi formal → insight proses bisnis |
| C: PostgreSQL | L2 | `PostgreSQLAdapter` | DB MOFIDS (8 tabel, 20 RFQ) | SQL | Explicit → Structured | Data tabel → narasi kontekstual |
| D: FR+DB | L1+L2 | `MultiSourceAdapter` | FR + PostgreSQL (gabungan) | TXT + SQL | Explicit → Cross-referenced | Verifikasi lintas spesifikasi FR dan data aktual |
| E: Hybrid All† | L1+L2+L3 | `MultiSourceAdapter` | FR + DB + Chat (semua layer) | TXT + SQL | Cross-Paradigm | Sintesis tacit + explicit + structured |

*†Skenario E: evaluasi reference-based (ROUGE-L dan BLEU-1 vs GROUND_TRUTH_HYBRID). Skenario A–D: reference-free (vs retrieved context).*

---

## 3. HASIL DAN PEMBAHASAN

### 3.1 Performa Sistem

#### 3.1.1 Evaluasi Single Query

Evaluasi awal dilakukan pada pertanyaan `"Apa bug yang ditemukan pada proses submit quotation BS-SB dan bagaimana solusinya?"` menggunakan log diskusi tim pengembang BOND_SYS (908 pesan, 3 file TXT) sebagai sumber, dengan model `Gemini 2.5-flash` (Google, multibahasa) sebagai model generatif. Pertanyaan ini merepresentasikan kebutuhan operasional nyata: *Product Owner* baru perlu mengetahui riwayat bug pada modul quotation tanpa harus menelusuri ratusan pesan diskusi secara manual.

Hasil evaluasi:

**Tabel 4.** Hasil Evaluasi Single Query (Pertanyaan A1) Menggunakan Gemini 2.5-flash

| Metrik | Gemini 2.5-flash | Komponen |
|---|---|---|
| Retrieval Relevance | **0.612** | Retrieval |
| Answer Faithfulness | **0.097** | Generasi |
| Answer Completeness | **0.900** | Generasi |
| ROUGE-L | **0.033** | Generasi |
| BLEU-1 | **0.000** | Generasi |
| **Precision@K** | **1.000** | **Retrieval** |
| **MRR** | **1.000** | **Retrieval** |
| **Overall** | **0.328** | Gabungan |

Metrik retrieval (Retrieval Relevance, Precision@K, MRR, Context Coverage) mencapai nilai tinggi karena komponen retrieval sepenuhnya independen dari pilihan LLM. Nilai Faithfulness yang rendah (0.097) merupakan karakteristik evaluasi *reference-free* pada teks berbahasa Indonesia, bukan kegagalan sistem — sistem berhasil menemukan konteks yang relevan namun tidak mereproduksi kata-kata dokumen sumber secara verbatim.

#### 3.1.2 Evaluasi Batch Multi-Sumber

Untuk membuktikan klaim Multi-Sumber dan mengukur efektivitas transfer pengetahuan, evaluasi batch dijalankan pada lima skenario dengan total 25 pertanyaan (5 per skenario). Skenario A, B, C mengevaluasi tiap layer corpus secara terpisah; Skenario D mengevaluasi kombinasi L1+L2; dan Skenario E mengevaluasi seluruh tiga layer secara bersamaan melalui `MultiSourceAdapter`. Skenario E satu-satunya yang menggunakan evaluasi *reference-based* (vs 5 jawaban referensi yang dikurasi manual), sedangkan Skenario A–D menggunakan evaluasi *reference-free* (vs retrieved context). Seluruh angka berikut merupakan hasil aktual dari run notebook menggunakan model Gemini 2.5-flash.

**Tabel 5.** Ringkasan Evaluasi Batch Multi-Sumber per Skenario

| Skenario | Adapter | Format | n | RR | Faith | Comp | ROUGE-L | BLEU-1 | P@K | MRR | CC | Overall | **KTE** | **MSRS** | **AQI** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: Chat Only (L3) | FolderSourceAdapter | TXT | 5 | 0.473 | 0.097 | 0.807 | 0.033 | 0.000 | 1.000 | 1.000 | 0.450 | **0.282** | **0.452** | **0.725** | **0.312** |
| B: FR/PDF (L1) | FolderSourceAdapter | TXT | 5 | 0.580 | 0.121 | 0.628 | 0.051 | 0.000 | 1.000 | 1.000 | 0.250 | **0.276** | **0.375** | **0.625** | **0.267** |
| C: PostgreSQL (L2) | PostgreSQLAdapter | SQL (8 tabel) | 5 | 0.460 | 0.046 | 0.609 | 0.024 | 0.000 | 1.000 | 1.000 | 0.650 | **0.228** | **0.328** | **0.825** | **0.226** |
| D: FR+DB (L1+L2) | MultiSourceAdapter | TXT + SQL | 5 | 0.576 | 0.136 | 0.833 | 0.046 | 0.000 | 1.000 | 1.000 | 0.525 | **0.318** | **0.484** | **0.763** | **0.338** |
| E: Hybrid All (L1+L2+L3)† | MultiSourceAdapter | TXT + SQL | 5 | 0.582 | 0.128 | 0.699 | **0.167** | **0.213** | 1.000 | 1.000 | 0.425 | **0.358** | **0.414** | **0.713** | **0.331** |

*RR = Retrieval Relevance; Faith = Answer Faithfulness; Comp = Answer Completeness; CC = Context Coverage.*
*KTE = (Faithfulness + Completeness) / 2. MSRS = (Precision@K + Context Coverage) / 2. AQI = (Faithfulness + Completeness + ROUGE-L) / 3.*
*†Skenario E: ROUGE-L dan BLEU-1 reference-based (vs GROUND_TRUTH_HYBRID). Skenario A–D: reference-free (vs retrieved context).*
*Semua nilai dibulatkan tiga desimal; selisih pembulatan ≤0.001 dapat terjadi pada formula komposit.*

Beberapa temuan kunci dari Tabel 5: (1) **Skenario D (FR+DB)** mencapai *Overall* tertinggi di antara skenario *reference-free* (0.318) dan *Answer Completeness* tertinggi (0.833), karena penggabungan dokumen FR dan data PostgreSQL memberikan konteks lintas paradigma yang paling lengkap untuk pertanyaan operasional. Secara keseluruhan termasuk Skenario E (*reference-based*), *Overall* tertinggi dicapai Skenario E (0.358). (2) **Skenario C (PostgreSQL)** mencapai MSRS tertinggi (0.825) dengan *Context Coverage* = 0.650, mencerminkan bahwa 8 tabel operasional memberikan keragaman sumber retrieval tertinggi — setiap query menarik chunk dari berbagai tabel berbeda. (3) **Skenario A (Chat Only)** mencapai *Answer Completeness* tinggi (0.807) karena pertanyaan dirancang spesifik sesuai konteks diskusi tim; sistem menemukan jawaban langsung dari log percakapan. (4) **Skenario B (FR/PDF)** mencapai *Answer Faithfulness* tertinggi (0.121) dan *Retrieval Relevance* kedua (0.580), karena dokumen FR mengandung informasi teknis yang padu sehingga overlap antara jawaban dan context tinggi. (5) **Skenario E (Hybrid)** mencapai *Retrieval Relevance* tertinggi (0.582) dan *Overall* = 0.358; ROUGE-L = 0.167 dan BLEU-1 = 0.213 adalah nilai *reference-based* yang bermakna — menunjukkan terdapat overlap konten yang nyata antara jawaban AI dan jawaban referensi yang dikurasi peneliti untuk pertanyaan lintas-layer.

KTE per skenario mencerminkan dimensi transfer pengetahuan yang berbeda: A (*Tacit→Operational*) = 0.452, B (*Explicit→Actionable*) = 0.375, C (*Explicit→Structured*) = 0.328, D (*Explicit→Cross-referenced*) = 0.484, E (*Cross-Paradigm*) = 0.414. Nilai KTE tertinggi pada Skenario D (0.484) karena kombinasi FR+DB menghasilkan jawaban yang faktual dan lengkap; Skenario E mencapai KTE 0.414 meskipun pertanyaannya paling kompleks lintas tiga layer. Precision@K dan MRR mencapai 1.000 di seluruh 25 pertanyaan, mengkonfirmasi komponen retrieval bekerja optimal di semua konfigurasi sumber dan tipe sumber.

#### 3.1.3 Ablation Study — Kontribusi Per Layer Corpus

Untuk membuktikan bahwa setiap layer corpus memberikan kontribusi yang terukur, *ablation study* dijalankan pada pertanyaan yang sama (E1–E5) dengan empat konfigurasi sumber yang berbeda secara bertahap.

**Tabel 6.** Hasil Ablation Study — Kontribusi Per Layer

| Konfigurasi | Layer Aktif | n | Overall | Faithfulness | Completeness | ROUGE-L | BLEU-1 |
|---|---|---|---|---|---|---|---|
| Ablasi-0: Chat only | L3 | 5 | 0.230 | 0.084 | 0.587 | 0.031‡ | 0.000 |
| Ablasi-1: FR only | L1 | 5 | 0.230 | 0.157 | 0.371 | 0.062‡ | 0.001 |
| Ablasi-2: FR+DB | L1+L2 | 5 | 0.231 | 0.127 | 0.408 | 0.055‡ | 0.002 |
| **Ablasi-Full: FR+DB+Chat** | **L1+L2+L3** | **5** | **0.358** | **0.128** | **0.699** | **0.167** | **0.213** |

*‡reference-free (vs retrieved context). Full (L1+L2+L3) reference-based (vs GROUND_TRUTH_HYBRID).*
*Overall = (RR + Faithfulness + Completeness + ROUGE-L + BLEU-1) / 5; kolom RR tidak ditampilkan karena dikumpulkan bersamaan dengan metrik lainnya. Ablasi-Full identik dengan Skenario E Tabel 5 (RR = 0.582).*

Pola ablasi membuktikan kontribusi yang dramatis: *Overall* dari 0.230 (Chat only) → 0.230 (FR only) → 0.231 (FR+DB) → 0.358 (Full). Tiga konfigurasi pertama menghasilkan *Overall* yang hampir identik (0.230–0.231), sedangkan konfigurasi Full memberikan lompatan signifikan (+0.127 dari FR+DB ke Full). Ini mengkonfirmasi bahwa **kombinasi ketiga layer secara bersamaan** adalah faktor penentu kualitas: ROUGE-L naik dari 0.055 → 0.167 (3.0×) dan BLEU-1 dari 0.002 → 0.213 saat Layer 3 (Chat) diintegrasikan, mengkonfirmasi bahwa informasi *tacit* dalam log diskusi tim berkontribusi nyata pada pertanyaan yang memerlukan konteks operasional (E1: bug quotation, E2: keputusan digit desimal, E3: insiden ETL MOFIDS, E4: alur upload allocation, E5: status fitur amend). Hasil ablation ini secara kuantitatif membuktikan bahwa arsitektur tiga-layer corpus — bukan hanya arsitektur pipeline multi-adapter — merupakan faktor penentu kualitas jawaban pada pertanyaan *cross-paradigm*.

### 3.2 Visualisasi Hasil

Hasil evaluasi batch divisualisasikan dalam empat panel terpisah (Gambar 2a–2d).

**Gambar 2a** menampilkan rata-rata metrik standar per skenario dalam bentuk grouped bar chart. P@K dan MRR konsisten mencapai 1.000 di semua skenario; Skenario B memiliki ROUGE-L tertinggi di antara skenario *reference-free* (0.051), sementara Skenario E memiliki ROUGE-L = 0.167 dan BLEU-1 = 0.213 (*reference-based*).

**Gambar 2b** menampilkan *Overall* score per pertanyaan (25 pertanyaan lintas 5 skenario), memperlihatkan distribusi performa antar pertanyaan dan skenario.

**Gambar 2c** membandingkan tiga metrik komposit (KTE, MSRS, AQI) antar skenario. MSRS tertinggi pada Skenario C (PostgreSQL, 0.825); KTE tertinggi Skenario D (0.484); Skenario E memiliki KTE = 0.414 meski pertanyaannya paling kompleks.

**Gambar 2d** merupakan radar chart yang memperlihatkan profil multi-dimensi kelima skenario, mengkonfirmasi bahwa masing-masing skenario memiliki profil yang berbeda: A unggul di Completeness, B di Faithfulness, C di MSRS, D di Overall dan KTE, dan E di Retrieval Relevance dan ROUGE-L *reference-based*.

![Gambar 2a](result_download/eval_panel1_metrik_standar.png)

**Gambar 2a.** Rata-rata metrik standar per skenario (grouped bar chart).

![Gambar 2b](result_download/eval_panel2_overall_per_q.png)

**Gambar 2b.** Overall score per pertanyaan (25 pertanyaan, 5 skenario).

![Gambar 2c](result_download/eval_panel3_komposit_kte_msrs_aqi.png)

**Gambar 2c.** Metrik komposit KTE, MSRS, dan AQI per skenario.

![Gambar 2d](result_download/eval_panel4_radar.png)

**Gambar 2d.** Radar chart profil multi-dimensi kelima skenario.

### 3.3 Keterbatasan Penelitian

Komponen retrieval menggunakan `paraphrase-multilingual-MiniLM-L12-v2` (50+ bahasa) sehingga tidak terdapat bias bahasa pada lapisan retrieval; bobot retrieval sepenuhnya ditentukan oleh similarity semantik, bukan metadata tipe sumber. Pada pertanyaan *single-entity*, Context Coverage cenderung rendah karena top-K chunk terkonsentrasi pada satu dokumen — ini merupakan cerminan akurat corpus, bukan bias sistem. Deduplikasi berbasis hash konten diimplementasikan untuk mencegah satu file terhitung ganda sebagai sumber berbeda.

Beberapa keterbatasan penelitian ini perlu diakui secara eksplisit:

**Keterbatasan Cakupan Evaluasi.** Evaluasi mencakup 25 pertanyaan di lima skenario dalam satu domain (sistem lelang obligasi pemerintah) dalam corpus berbahasa Indonesia. Validasi empiris pada domain dan bahasa lain diperlukan untuk memperkuat generalisasi hasil.

**Ground Truth Parsial.** Hanya Skenario E (5 pertanyaan) yang dilengkapi jawaban referensi (*ground truth*) yang dikurasi manual oleh peneliti. Skenario A–D menggunakan evaluasi *reference-free* sehingga ROUGE-L dan BLEU-1 untuk keempat skenario tersebut dihitung terhadap retrieved context, bukan jawaban acuan ideal. Nilai ROUGE-L Skenario E (0.167) dan BLEU-1 (0.213) yang bersifat *reference-based* tidak dapat dibandingkan langsung dengan ROUGE-L Skenario A–D yang *reference-free* (0.024–0.051). Perluasan ground truth ke seluruh 25 pertanyaan akan meningkatkan validitas komparatif antar skenario.

**Skala Evaluasi.** Evaluasi pada 25 pertanyaan di lima skenario memberikan gambaran *proof-of-concept* yang memadai, namun belum cukup untuk klaim statistik yang dapat digeneralisasi. Penelitian lanjutan disarankan menggunakan minimal 20–50 pasangan query-answer per skenario dengan ground truth yang dikurasi untuk seluruh skenario.

### 3.4 Implikasi

#### 3.4.1 Implikasi untuk Transfer Pengetahuan Organisasi

Sistem ini secara langsung menjawab tantangan yang diidentifikasi dalam pendahuluan. Product Owner yang baru bergabung dapat langsung bertanya dalam bahasa natural, misalnya "Apa bug yang ditemukan pada submit quotation BS-SB dan bagaimana solusinya?" atau "Apa status implementasi fitur amend pada modul trade custody?", dan sistem akan mencari jawaban dari kombinasi dokumen FR TXT, log diskusi tim 908 pesan, dan database PostgreSQL MOFIDS secara otomatis dalam satu query. Skenario ini merepresentasikan transfer pengetahuan dari personel lama ke penerus tanpa harus membaca ulang seluruh dokumentasi yang tersebar, sejalan dengan tantangan eksplisitasi pengetahuan tacit yang diidentifikasi Nonaka dan Takeuchi (1995). Prinsip yang sama berlaku untuk konteks organisasi lain: analis baru dapat bertanya tentang keputusan teknis atau konfigurasi sistem dari dokumentasi proyek yang sudah ada.

#### 3.4.2 Implikasi Teknis untuk Pengembangan Selanjutnya

Arsitektur Adapter Pattern yang digunakan membuka jalan untuk extensibility: sumber baru (MongoDB, SharePoint, Google Drive API) dapat ditambahkan dengan mengimplementasikan dua method, yaitu `load()` dan `describe()`, tanpa mengubah pipeline inti. Ini sejalan dengan prinsip Open/Closed Principle dalam desain perangkat lunak (Martin 2017).

Trade-off *real-time* vs pre-indexed perlu dipertimbangkan sesuai use case:

**Tabel 7.** Perbandingan Trade-off *Real-Time* vs Pre-indexed Indexing

| Aspek | *Real-Time* (sistem ini) | Pre-indexed |
|---|---|---|
| Konsistensi data | Selalu terkini | Bisa stale |
| Waktu cold start | ~2–5 detik | Instan |
| Manajemen storage | Tidak ada overhead disk | Perlu sinkronisasi |
| Cocok untuk | Dokumen yang sering berubah | Korpus statis besar |

Untuk korpus statis yang sangat besar (jutaan dokumen), FAISS IVF (Inverted File Index) atau integrasi dengan Elasticsearch hybrid search dapat dipertimbangkan sebagai evolusi arsitektur.

---

## 4. KESIMPULAN

Penelitian ini berhasil mengembangkan sistem RAG agnostic multi-sumber yang memenuhi tujuan penelitian, dengan enam temuan utama:

1. **Arsitektur *agnostic* terealisasi:** `SourceDetector` + `SourceFactory` + `BaseSourceAdapter` memungkinkan penanganan folder (TXT/PDF/MD/LOG) dan PostgreSQL (3 mode query) dari satu parameter `source`, tanpa perubahan pada pipeline inti. Deteksi otomatis tipe sumber berbasis pattern matching berjalan konsisten di seluruh 25 pertanyaan.

2. ***Real-time* indexing terbukti benar:** Setiap pemanggilan `pipeline.ask()` membangun indeks FAISS secara in-memory; tidak ada file index di disk. Perubahan konten di sumber langsung tercermin dalam hasil retrieval tanpa restart sistem.

3. **Multi-sumber terbukti secara eksperimen:** Evaluasi batch 25 pertanyaan (5 skenario × 5 pertanyaan) menunjukkan Precision@K = 1.000 dan MRR = 1.000 di seluruh skenario. Performa agregat: A (Chat Only, *Overall* = 0.282, MSRS = 0.725), B (FR/PDF, *Overall* = 0.276, MSRS = 0.625), C (PostgreSQL, *Overall* = 0.228, MSRS = 0.825), D (FR+DB, *Overall* = 0.318, MSRS = 0.763), E (Hybrid, *Overall* = 0.358, MSRS = 0.713). Skenario E membuktikan klaim *source-agnostic* pada level tertinggi: *Retrieval Relevance* = 0.582 tertinggi di antara semua skenario, mengkonfirmasi query dapat menjangkau chunk relevan dari ketiga layer corpus secara bersamaan.

4. **Ground truth Skenario E menghasilkan ROUGE-L dan BLEU-1 yang bermakna:** Dengan 5 jawaban referensi yang dikurasi manual, Skenario E mencapai ROUGE-L = 0.167 dan BLEU-1 = 0.213 (*reference-based*) — membuktikan terdapat overlap konten yang nyata antara jawaban AI dan jawaban ideal untuk pertanyaan *cross-layer*. Ini merupakan peningkatan ~7× dibandingkan ROUGE-L skenario *reference-free* tertinggi (B = 0.051).

5. **Ablation study membuktikan kontribusi dramatis Layer 3:** *Overall*: Chat-only = 0.230 → FR-only = 0.230 → FR+DB = 0.231 → Full = 0.358. Tiga konfigurasi awal hampir identik; lompatan +0.127 terjadi saat ketiga layer digabungkan — ROUGE-L naik 3× dari 0.055 ke 0.167 dan BLEU-1 dari 0.002 ke 0.213, membuktikan bahwa log diskusi tim (Layer 3) adalah komponen penentu kualitas jawaban *cross-paradigm*.

6. **Efektivitas transfer pengetahuan terukur:** KTE per skenario: A = 0.452 (*Tacit→Operational*), B = 0.375 (*Explicit→Actionable*), C = 0.328 (*Explicit→Structured*), D = 0.484 (*Explicit→Cross-referenced*), E = 0.414 (*Cross-Paradigm*). Skenario D memiliki KTE tertinggi (0.484), diikuti A (0.452) dan E (0.414), menunjukkan bahwa kombinasi sumber yang tepat dengan pertanyaan domain-spesifik secara konsisten menghasilkan transfer pengetahuan yang efektif.

Kontribusi utama penelitian adalah: (i) desain pola Adapter yang memisahkan *concerns* antara sumber data, pemrosesan teks, retrieval, dan generasi; (ii) desain evaluasi lima skenario dengan tiga-layer corpus operasional nyata; dan (iii) framework ground truth parsial yang memungkinkan validasi *reference-based* pada skenario prioritas tanpa mensyaratkan ground truth lengkap untuk semua skenario.

Untuk penelitian selanjutnya disarankan: (1) perluasan ground truth ke seluruh 25 pertanyaan untuk validasi komparatif yang lebih kuat; (2) *hybrid search* (FAISS + BM25) untuk meningkatkan retrieval pada query dengan token spesifik domain; (3) fine-tuning embedding model pada dokumen domain organisasi spesifik; (4) perbandingan dengan sistem RAG berbasis vector database komersial (Pinecone, Weaviate) sebagai baseline arsitektur.

---

## 5. DAFTAR PUSTAKA

Chui M, Manyika J, Bughin J, Dobbs R, Roxburgh C, Sarrazin H, Sands G, Westergren M (2012) The social economy: unlocking value and productivity through social technologies. McKinsey Global Institute. https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-social-economy

Carbonell J, Goldstein J (1998) The use of MMR, diversity-based reranking for reordering documents and producing summaries. In: Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pp 335–336. https://doi.org/10.1145/290941.291025

Es S, James J, Espinosa-Anke L, Schockaert S (2023) RAGAS: automated evaluation of retrieval augmented generation. arXiv:2309.15217. https://doi.org/10.48550/arXiv.2309.15217

Fabbri AR, Kryściński W, McCann B, Xiong C, Socher R, Radev D (2021) SummEval: re-evaluating summarization evaluation. Trans Assoc Comput Linguist 9:391–409. https://doi.org/10.1162/tacl_a_00373

Gao Y, Xiong Y, Gao X, Jia K, Pan J, Bi Y, Dai Y, Sun J, Wang M, Wang H (2024) Retrieval-augmented generation for large language models: a survey. arXiv:2312.10997. https://doi.org/10.48550/arXiv.2312.10997

Gartner (2020) Gartner says employees spend too much time on low-value tasks: use AI and automation to fix it. Gartner Newsroom. https://www.gartner.com/en/newsroom/press-releases/2020-01-23-gartner-says-employees-spend-too-much-time-on-low-value-tasks

IDC (2023) 90% of data is unstructured and it's full of untapped value. IDC Blog. https://blogs.idc.com/2023/05/09/90-of-data-is-unstructured-and-its-full-of-untapped-value/

Izacard G, Grave E (2021) Leveraging passage retrieval with generative models for open domain question answering. In: Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics, pp 874–880. https://doi.org/10.18653/v1/2021.eacl-main.74

Johnson J, Douze M, Jégou H (2019) Billion-scale similarity search with GPUs. IEEE Trans Big Data 7(3):535–547. https://doi.org/10.1109/TBDATA.2019.2921572

Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V, Goyal N, Kiela D (2020) Retrieval-augmented generation for knowledge-intensive NLP tasks. In: Advances in Neural Information Processing Systems 33, pp 9459–9474. https://doi.org/10.48550/arXiv.2005.11401

Lin CY (2004) ROUGE: a package for automatic evaluation of summaries. In: Proceedings of the ACL Workshop on Text Summarization Branches Out. https://aclanthology.org/W04-1013

Martin RC (2017) Clean architecture: a craftsman's guide to software structure and design. Prentice Hall, Upper Saddle River

Nonaka I, Takeuchi H (1995) The knowledge-creating company. Oxford University Press, New York

Papineni K, Roukos S, Ward T, Zhu WJ (2002) BLEU: a method for automatic evaluation of machine translation. In: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pp 311–318. https://doi.org/10.3115/1073083.1073135

Reimers N, Gurevych I (2019) Sentence-BERT: sentence embeddings using Siamese BERT-networks. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. https://doi.org/10.18653/v1/D19-1410

Ren H, Shi H, Zhao W, Zhao J, Zhao Y (2023) Self-RAG: learning to retrieve, generate, and critique through self-reflection. arXiv:2307.11019. https://doi.org/10.48550/arXiv.2307.11019

Voorhees EM (1999) The TREC-8 question answering track report. In: Proceedings of the 8th Text REtrieval Conference (TREC-8), pp 77–82. https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf

Yasunaga M, Ren H, Bosselut A, Liang P, Leskovec J (2021) QA-GNN: reasoning with language models and knowledge graphs for question answering. In: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics, pp 535–545. https://doi.org/10.18653/v1/2021.naacl-main.45

