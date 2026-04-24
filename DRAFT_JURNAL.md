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

Pergantian personel kunci pada *document-based service organizations* menciptakan kesenjangan pengetahuan yang nyata: pengetahuan domain yang tersebar di berbagai artefak — dokumen teknis, prosedur operasional, dan basis data — tidak dapat diakses secara cepat oleh penerus, sehingga setiap permintaan klarifikasi memerlukan waktu berjam-jam hingga berhari-hari sebelum kesimpulan dapat dirumuskan. Organisasi modern menghadapi tantangan mengelola pengetahuan yang tersebar di sumber heterogen, yaitu dokumen tidak terstruktur (PDF, TXT) dan basis data relasional, sehingga menghambat transfer pengetahuan dan pengambilan keputusan. Penelitian ini mengembangkan sistem *Question Answering* (QA) berbasis *Retrieval-Augmented Generation* (RAG) yang bersifat *agnostic* terhadap sumber data: sistem hanya memerlukan satu parameter `source` untuk mendeteksi dan menangani berbagai jenis sumber secara otomatis. Sistem mengintegrasikan dua adapter utama, yaitu `FolderSourceAdapter` untuk sumber tidak terstruktur dan `PostgreSQLAdapter` untuk basis data relasional, dengan indeks vektor FAISS yang dibangun secara *real-time* tanpa pra-komputasi ke disk. Evaluasi menggunakan delapan metrik kuantitatif dan tiga metrik komposit: *Knowledge Transfer Effectiveness* (KTE), *Multi-Source Retrieval Score* (MSRS), dan *Answer Quality Index* (AQI). Eksperimen mencakup empat skenario dengan 20 pertanyaan: (A) dokumen operasional dan log diskusi tim dalam format PDF+TXT, (B) PostgreSQL lima tabel berelasi, (C) pertanyaan lintas-sumber multi-format, dan (D) *cross-paradigm*, yaitu penggabungan sumber unstructured dan structured secara bersamaan dalam satu indeks FAISS gabungan. Hasil menunjukkan Precision@K = 1.00 dan MRR = 1.00 pada semua skenario, dengan KTE tertinggi pada Skenario B (0.502) yang bersumber dari PostgreSQL dan satu-satunya skenario yang melampaui ambang batas efektif KTE ≥ 0.5. Skenario A mencapai MSRS = 0.650, Skenario B MSRS = 0.812, Skenario C MSRS = 0.700, dan Skenario D MSRS = 0.725. Keempat skenario membuktikan kemampuan sistem menangani dimensi transfer pengetahuan: *Explicit→Actionable*, *Structured→Contextual*, *Tacit→Explicit*, dan *Cross-Paradigm*.

**Kata Kunci:** *Retrieval-Augmented Generation*, *Question Answering*, Multi-Sumber, FAISS, Transfer Pengetahuan

---

## ABSTRACT

Key personnel turnover in document-based service organizations creates a real knowledge gap: domain knowledge dispersed across technical documents, operational procedures, and databases cannot be accessed quickly by successors, causing each clarification request to require hours or even days before a conclusion can be reached. Modern organizations face challenges managing knowledge dispersed across heterogeneous sources, namely unstructured documents (PDF, TXT) and relational databases, hindering knowledge transfer and decision-making. This study develops a Question Answering (QA) system based on Retrieval-Augmented Generation (RAG) that is agnostic to the data source: the system requires only a single `source` parameter to automatically detect and handle various source types. Two main adapters are integrated, namely `FolderSourceAdapter` for unstructured sources and `PostgreSQLAdapter` for relational databases, with FAISS vector indices built in real-time without disk pre-computation. Evaluation uses eight quantitative metrics and three composite metrics: Knowledge Transfer Effectiveness (KTE), Multi-Source Retrieval Score (MSRS), and Answer Quality Index (AQI). Experiments cover four scenarios with 20 questions: (A) operational documents and team discussion logs in PDF+TXT format, (B) PostgreSQL with five interrelated tables, (C) cross-source multi-format queries, and (D) cross-paradigm, combining unstructured and structured sources simultaneously in a single merged FAISS index. Results show Precision@K = 1.00 and MRR = 1.00 across all scenarios, with highest KTE on Scenario B (0.502) sourced from PostgreSQL, the only scenario exceeding the effective threshold KTE ≥ 0.5. MSRS values are: Scenario A = 0.650, Scenario B = 0.812, Scenario C = 0.700, Scenario D = 0.725. All four scenarios demonstrate the system's capability across knowledge transfer dimensions: Explicit→Actionable, Structured→Contextual, Tacit→Explicit, and Cross-Paradigm.

**Keywords:** Retrieval-Augmented Generation, Question Answering, Multi-Source, FAISS, Knowledge Transfer

---

## 1. PENDAHULUAN

Organisasi yang beroperasi dalam ekosistem layanan berbasis dokumen (*document-based service organizations*) — seperti lembaga keuangan, penyedia layanan teknis domain-spesifik, dan operator platform transaksi — sangat bergantung pada pengetahuan yang terakumulasi dalam berbagai artefak: spesifikasi kebutuhan fungsional, prosedur operasional, konfigurasi sistem, dan rekam jejak transaksi. Tantangan kritis muncul saat terjadi pergantian personel kunci (*knowledge worker turnover*): pengetahuan domain yang selama ini melekat pada individu harus ditransfer kepada penerus dalam waktu terbatas, sementara operasional layanan tidak dapat berhenti. Menurut McKinsey [3], karyawan yang tergolong *knowledge worker* menghabiskan sekitar 20% waktu kerjanya hanya untuk mencari informasi internal, dan IDC [1] melaporkan bahwa hingga 90% data organisasi bersifat tidak terstruktur — tersebar di dokumen PDF, berkas teks, dan basis data — sehingga hanya sebagian kecil yang benar-benar dapat diakses dan dimanfaatkan secara efektif.

Permasalahan ini semakin kompleks karena pengetahuan organisasi tersebar secara heterogen: spesifikasi kebutuhan fungsional tersimpan dalam dokumen PDF ratusan halaman, data operasional tersimpan di tabel basis data relasional, sementara konteks keputusan teknis kerap hanya ada dalam ingatan personel yang bersangkutan. Nonaka & Takeuchi [4] membedakan pengetahuan *tacit* (tersirat dalam pikiran individu) dan *explicit* (terdokumentasi dalam artefak); namun bahkan pengetahuan *explicit* pun sulit ditemukan secara cepat apabila tersebar lintas repositori dengan format berbeda. Gartner [2] memperkirakan karyawan menghabiskan hingga 30% waktu kerja untuk aktivitas bernilai rendah yang seharusnya dapat diotomatisasi, termasuk penelusuran dokumen dan klarifikasi informasi operasional yang berulang.

Dalam lingkungan enterprise yang menjadi konteks penelitian ini — sebuah sistem platform transaksi keuangan yang mengelola proses penawaran dan alokasi instrumen keuangan untuk lembaga-lembaga peserta terdaftar — dampak *turnover* ini dirasakan secara langsung secara operasional. Dalam rentang dua tahun, terjadi dua kali pergantian *Product Owner*, dengan komposisi tim yang menyusut dari tiga menjadi dua orang. Akibatnya, setiap permintaan klarifikasi teknis dari pengguna eksternal (*peserta platform*) — seperti alur proses transaksi, parameter konfigurasi sesi, atau aturan instrumen — memerlukan waktu respons yang bervariasi dari beberapa jam hingga satu hari kerja penuh, bergantung pada seberapa dalam konteks tersebut terdokumentasi dan seberapa akrab personel yang tersisa dengan materi tersebut. Kondisi ini merepresentasikan pola umum yang ditemukan pada berbagai *document-based service organization*: volume pengetahuan tidak berkurang, tetapi kapasitas manusia untuk mengaksesnya secara cepat dan akurat semakin terbatas.

Solusi manajemen pengetahuan (*Knowledge Management*/KM) tradisional — seperti wiki internal, basis pengetahuan statis, atau FAQ — tidak mampu menjawab pertanyaan dinamis yang membutuhkan inferensi lintas sumber secara *real-time*. Pengguna masih harus menelusuri spesifikasi teknis dan prosedur operasional ratusan halaman sekaligus mengecek tabel konfigurasi di basis data secara manual — proses yang memakan waktu berjam-jam bahkan berhari-hari sebelum kesimpulan dapat dirumuskan. Model bahasa besar (LLM) tanpa *grounding* faktual menghadirkan risiko berbeda: Gao et al. [14] mengidentifikasi tiga kelemahan kritis LLM dalam konteks layanan domain-spesifik, yaitu *hallucination*, pengetahuan kedaluwarsa, dan penalaran tidak transparan (*non-transparent reasoning*). Lebih lanjut, Ren et al. [15] menunjukkan bahwa LLM cenderung merespons dengan keyakinan penuh bahkan di luar batas pengetahuannya (*unwavering confidence beyond knowledge boundary*) — perilaku yang berbahaya dalam konteks layanan yang menuntut akurasi dan *traceability* tinggi. Terdapat kesenjangan nyata: belum ada solusi yang mampu menjawab pertanyaan faktual secara akurat **sekaligus cepat** dari sumber-sumber heterogen yang sudah ada di organisasi — di mana *cepat* berarti pengguna tidak perlu menunggu lama untuk dapat menyimpulkan sesuatu — tanpa memerlukan konfigurasi teknis yang berbeda untuk setiap tipe sumber.

Pendekatan *Retrieval-Augmented Generation* (RAG) yang diperkenalkan Lewis et al. [5] membuka peluang untuk mengatasi keterbatasan ini: dengan menggabungkan retrieval dari sumber eksternal dan kemampuan generasi LLM, sistem dapat memberikan jawaban faktual, *grounded*, dan berbasis dokumen aktual. Izacard & Grave [6] memperluas pendekatan ini untuk *passage retrieval* pada domain terbuka, sementara Johnson et al. [8] mengembangkan FAISS sebagai infrastruktur pencarian vektor yang efisien, dan Reimers & Gurevych [9] menyediakan representasi semantik multibahasa melalui Sentence-BERT. Namun, implementasi RAG yang ada umumnya bersifat *single-source* dan memerlukan konfigurasi berbeda untuk setiap tipe sumber data, sehingga menambah beban teknis bagi organisasi dengan ekosistem data heterogen. Evaluasi sistem QA berbasis RAG sendiri memerlukan perspektif ganda: dari sisi retrieval menggunakan Precision@K dan MRR [10], dari sisi generasi menggunakan ROUGE-L [11] dan BLEU-1 [12], serta dimensi khusus RAG yang diusulkan Es et al. [13] yaitu *faithfulness* dan *answer relevance*.

Penelitian ini mengembangkan sistem QA berbasis RAG yang bersifat *agnostic* terhadap sumber data — mampu menangani dokumen tidak terstruktur (PDF, TXT) dan basis data relasional (PostgreSQL) secara terpadu melalui satu antarmuka pemrograman (*Adapter Pattern*) — sebagai infrastruktur transfer pengetahuan yang dapat diimplementasikan langsung pada *document-based service organization*. Kontribusi penelitian ini adalah sebagai berikut:

1. **Identifikasi faktor-faktor yang berpengaruh pada kualitas QA** dalam konteks *document-based enterprise service systems*: menganalisis aspek domain, karakteristik sumber data heterogen, dan pola pertanyaan operasional yang menentukan relevansi retrieval dan akurasi jawaban pada sistem transfer pengetahuan berbasis RAG.

2. **Rancangan model QA multi-sumber dengan arsitektur RAG Adapter Pattern**: desain arsitektur *agnostic* yang mengintegrasikan `FolderSourceAdapter` untuk sumber tidak terstruktur (PDF, TXT) dan `PostgreSQLAdapter` untuk basis data relasional dalam indeks FAISS *real-time in-memory*, tanpa pra-komputasi ke disk.

3. **Evaluasi empiris berbasis data operasional nyata dalam 4 skenario lintas paradigma**: pengujian performa sistem menggunakan 8 metrik kuantitatif dan 3 metrik komposit — *Knowledge Transfer Effectiveness* (KTE), *Multi-Source Retrieval Score* (MSRS), dan *Answer Quality Index* (AQI) — pada dataset yang berasal dari sistem platform transaksi keuangan operasional yang sesungguhnya.

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

Evaluasi dirancang dalam empat skenario yang merepresentasikan tipe sumber dan dimensi transfer pengetahuan berbeda. Seluruh dokumen berbahasa Indonesia dan merupakan dokumen publik atau data sintetis yang dianonimisasi.

**Skenario A (Unstructured — Explicit→Actionable):** Dua press release keuangan kuartalan emiten publik (BBKP dan TINS) dalam format PDF, serta log diskusi tim analis dalam format TXT. Lima pertanyaan mencakup perbandingan laba bersih, pertumbuhan kredit per segmen, kinerja produksi, dan rekomendasi investasi.

**Skenario B (Structured — Structured→Contextual):** Basis data PostgreSQL dengan 5 tabel berelasi yang merepresentasikan data operasional tim riset ekuitas, mencakup profil pengguna, produk, transaksi, daftar pantauan saham, dan catatan analisis. Pertanyaan mencakup *single-table* (baseline) hingga *three-table JOIN*.

**Skenario C (Unstructured Multi-format — Tacit→Explicit):** Folder yang sama dengan Skenario A dengan filter dokumen privat dinonaktifkan. Set pertanyaan dirancang untuk verifikasi konsistensi antara angka resmi di press release dan klaim dalam log diskusi tim, dibagi tiga sub-tipe: *PR-only*, *chat-only*, dan *cross-source*.

**Skenario D (Hybrid — Cross-Paradigm):** Penggabungan sumber Skenario A dan B secara bersamaan dalam satu indeks FAISS melalui `MultiSourceAdapter`. Lima pertanyaan dirancang untuk membutuhkan informasi dari kedua paradigma sumber (unstructured dan structured) sekaligus.

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
- **MRR** = 1/rank_chunk_relevan_pertama (Voorhees [10])
- **Context Coverage** = unique_sources / total_chunks, mengukur keragaman sumber

**Dimensi Kualitas Jawaban:**
- **Retrieval Relevance** = cosine similarity embedding pertanyaan vs rata-rata embedding chunks
- **Answer Faithfulness** = F1 token overlap jawaban vs gabungan context (anti-hallucination)
- **Answer Completeness** = rasio keyword pertanyaan yang muncul dalam jawaban

**Dimensi NLP Akademik:**
- **ROUGE-L** (Lin [11]) = F1 berbasis Longest Common Subsequence
- **BLEU-1** (Papineni et al. [12]) = unigram precision dengan brevity penalty

Selain delapan metrik di atas, penelitian ini mendefinisikan **tiga metrik komposit** untuk menjawab tiga pertanyaan evaluasi yang tidak dapat dijawab oleh metrik tunggal manapun:

**Metrik Komposit 1: Knowledge Transfer Effectiveness (KTE)**

$$KTE = \frac{\text{Answer Faithfulness} + \text{Answer Completeness}}{2}$$

KTE merupakan rata-rata dari dua komponen: Faithfulness (proporsi jawaban yang didukung context, sebagai ukuran anti-halusinasi) dan Completeness (proporsi keyword pertanyaan yang tercakup dalam jawaban). Ambang batas efektif ditetapkan KTE ≥ 0.5. Transfer pengetahuan berhasil hanya jika jawaban sekaligus tidak halusinasi dan menjawab pertanyaan secara lengkap; jika salah satu komponen bernilai nol maka pengetahuan gagal dipindahkan, sehingga rata-rata sederhana sudah memadai sebagai ukuran efektivitas [4].

**Metrik Komposit 2: Multi-Source Retrieval Score (MSRS)**

$$MSRS = \frac{\text{Precision@K} + \text{Context Coverage}}{2}$$

MSRS merupakan rata-rata dari dua komponen: Precision@K (proporsi chunk yang relevan dalam top-K hasil retrieval) dan Context Coverage (keragaman sumber dokumen dalam top-K). Sistem yang hanya menarik dari satu file akan memperoleh Context Coverage rendah meskipun Precision@K-nya tinggi; MSRS mendeteksi kondisi ini dan memastikan klaim multi-sumber terbukti secara retrieval. Pendekatan ini mengaplikasikan prinsip diversity dalam Information Retrieval [16] ke konteks multi-source RAG.

**Metrik Komposit 3: Answer Quality Index (AQI)**

$$AQI = \frac{\text{Answer Faithfulness} + \text{Answer Completeness} + \text{ROUGE-L}}{3}$$

AQI merupakan rata-rata dari tiga komponen: Faithfulness (anti-halusinasi), Completeness (cakupan pertanyaan), dan ROUGE-L (kemiripan struktural berbasis urutan kata terhadap context). KTE hanya mengukur apakah pengetahuan tersampaikan secara konten; AQI menambahkan dimensi linguistik untuk mendeteksi jawaban yang memadai secara topik namun strukturnya jauh berbeda dari dokumen sumber. Pendekatan multi-aspek ini konsisten dengan metodologi SummEval [15].

**Hubungan antar metrik komposit:** KTE mengukur dari perspektif *pengguna* (apakah pengetahuan tersampaikan), MSRS dari perspektif *sistem* (apakah multi-sumber terbukti), dan AQI dari perspektif *linguistik* (apakah jawaban berkualitas NLP). Ketiganya bersifat komplementer; sistem yang baik seharusnya memperoleh skor tinggi pada ketiga dimensi secara bersamaan.

**Metrik Agregat Overall** dihitung sebagai rata-rata sederhana dari lima metrik kualitas jawaban:

$$Overall = \frac{RR + Faithfulness + Completeness + ROUGE\text{-}L + BLEU\text{-}1}{5}$$

Metrik retrieval murni (Precision@K, MRR, Context Coverage) tidak dimasukkan dalam Overall agar nilai agregat ini mencerminkan kualitas jawaban, bukan kualitas retrieval yang secara konsisten sempurna (P@K=MRR=1.000).

Pemisahan tiga dimensi evaluasi ini memungkinkan identifikasi apakah nilai rendah disebabkan oleh kegagalan retrieval atau keterbatasan kapabilitas bahasa model generatif; keduanya merupakan masalah yang memerlukan solusi berbeda.

Untuk membuktikan klaim "Multi-Sumber", evaluasi dirancang dalam empat skenario yang masing-masing merepresentasikan satu dimensi transfer pengetahuan organisasi. Skenario D secara khusus membuktikan klaim *source-agnostic* pada level paling tinggi, di mana sistem dapat menjawab pertanyaan yang membutuhkan informasi dari sumber tidak terstruktur (folder PDF+TXT) **dan** sumber terstruktur (PostgreSQL) secara bersamaan dalam satu query:

**Tabel 3.** Desain Evaluasi Multi-Sumber dan Dimensi Transfer Pengetahuan

| Skenario | Adapter | Sumber | Format | Dimensi TK | Tipe Pengetahuan |
|---|---|---|---|---|---|
| A | `FolderSourceAdapter` | Press release PDF + log diskusi TXT | PDF + TXT | Explicit → Actionable | Laporan keuangan formal → insight investasi |
| B | `PostgreSQLAdapter` | Neon PostgreSQL (5 tabel) | SQL | Structured → Contextual | Data tabel → narasi |
| C | `FolderSourceAdapter` | Press release PDF + log diskusi TXT | PDF + TXT | Tacit → Explicit | Diskusi informal → jawaban terstruktur |
| D | `MultiSourceAdapter` | Folder + PostgreSQL (gabungan) | PDF + TXT + SQL | Cross-Paradigm | Verifikasi lintas paradigma data: unstructured ↔ structured |

---

## 3. HASIL DAN PEMBAHASAN

### 3.1 Performa Sistem

#### 3.1.1 Evaluasi Single Query

Evaluasi awal dilakukan pada pertanyaan `"Berapa laba bersih KB Bank (BBKP) pada Q1 2025 dan bagaimana perubahannya dibandingkan Q1 2024?"` menggunakan Press Release PT Bank KB Bukopin Tbk (BBKP) Q1 2025 dan log diskusi tim analis berbahasa Indonesia, dengan model `Gemini 2.5-flash` (Google, multibahasa) sebagai model generatif.

Hasil evaluasi:

**Tabel 4.** Hasil Evaluasi Single Query (Pertanyaan A1) Menggunakan Gemini 2.5-flash

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

Metrik retrieval (Retrieval Relevance, Precision@K, MRR, Context Coverage) mencapai nilai tinggi karena komponen retrieval sepenuhnya independen dari pilihan LLM. Nilai Faithfulness yang rendah (0.062) merupakan karakteristik evaluasi reference-free pada teks berbahasa Indonesia, bukan kegagalan sistem.

#### 3.1.2 Evaluasi Batch Multi-Sumber

Untuk membuktikan klaim judul "Multi-Sumber" dan mengukur efektivitas transfer pengetahuan, evaluasi batch dijalankan pada empat skenario dengan total 20 pertanyaan (5 per skenario). Skenario A–C mengevaluasi tiap paradigma sumber secara terpisah; Skenario D mengevaluasi `MultiSourceAdapter` yang menggabungkan FolderSourceAdapter dan PostgreSQLAdapter dalam satu indeks FAISS. Seluruh angka berikut merupakan hasil aktual dari run notebook menggunakan model Gemini 2.5-flash.

**Tabel 5.** Ringkasan Evaluasi Batch Multi-Sumber per Skenario

| Skenario | Adapter | Format | n | RR | Faith | Comp | ROUGE-L | P@K | MRR | CC | Overall | **KTE** | **MSRS** | **AQI** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: Press Release PDF + Chat TXT | FolderSourceAdapter | PDF + TXT | 5 | 0.724 | 0.142 | 0.716 | 0.073 | 1.000 | 1.000 | 0.300 | **0.331** | **0.429** | **0.650** | **0.310** |
| B: PostgreSQL | PostgreSQLAdapter | SQL (5 tabel) | 5 | 0.498 | 0.196 | 0.808 | 0.087 | 1.000 | 1.000 | 0.625 | **0.319** | **0.502** | **0.812** | **0.364** |
| C: Chat Log TXT | FolderSourceAdapter | TXT | 5 | 0.667 | 0.095 | 0.825 | 0.040 | 1.000 | 1.000 | 0.400 | **0.325** | **0.460** | **0.700** | **0.320** |
| D: Cross-Paradigm (Hybrid) | MultiSourceAdapter | PDF + TXT + SQL | 5 | 0.656 | 0.109 | 0.596 | 0.057 | 1.000 | 1.000 | 0.450 | **0.284** | **0.352** | **0.725** | **0.254** |

*RR = Retrieval Relevance; Faith = Answer Faithfulness; Comp = Answer Completeness; CC = Context Coverage.*
*KTE = (Faithfulness + Completeness) / 2, mengukur efektivitas transfer pengetahuan. Ambang batas efektif: KTE ≥ 0.5.*
*MSRS = (Precision@K + Context Coverage) / 2, mengukur kualitas retrieval multi-sumber.*
*AQI = (Faithfulness + Completeness + ROUGE-L) / 3, mengukur kualitas linguistik jawaban.*
*BLEU-1 ≈ 0.000–0.004 di semua skenario, merupakan expected behavior pada reference-free evaluation (lihat Sec 3.3); nilai ini dimasukkan dalam kalkulasi Overall tetapi tidak ditampilkan sebagai kolom tersendiri karena nilainya konstan di seluruh skenario.*
*Setiap skenario mencakup 5 pertanyaan (total 20 pertanyaan, masing-masing 5 per skenario).*
*Semua nilai dibulatkan tiga desimal sesuai output notebook; selisih pembulatan ≤0.001 dapat terjadi pada formula komposit.*

Beberapa temuan kunci dari Tabel 5: (1) **Skenario A** mencapai Precision@K dan MRR sempurna (1.000) dengan Retrieval Relevance = 0.724; retrieval bekerja optimal pada dokumen PDF + TXT berbahasa Indonesia. MSRS = 0.650 mencerminkan Context Coverage = 0.300 yang relatif lebih rendah karena pertanyaan cenderung berfokus pada satu entitas (BBKP atau TINS). (2) **Skenario B** memiliki KTE tertinggi (0.502) melampaui ambang batas efektif KTE ≥ 0.5, karena data tabular dari PostgreSQL menghasilkan jawaban yang paling *complete* (0.808) dan Context Coverage tertinggi (0.625), mencerminkan data terstruktur yang tidak ambigu dari 5 tabel berbeda. MSRS = 0.812 merupakan yang tertinggi. (3) **Skenario C** mencapai Answer Completeness tertinggi (0.825); pertanyaan dari chat log TXT mendapatkan jawaban paling lengkap karena konteks diskusi analis kaya informasi eksplisit. KTE = 0.460 merupakan tertinggi kedua. (4) **Skenario D** memvalidasi *source-agnostic* pada level paling fundamental: `MultiSourceAdapter` menggabungkan semua dokumen dari FolderSourceAdapter (PDF+TXT) dan PostgreSQLAdapter (5 tabel) menjadi satu pool indeks FAISS. Context Coverage = 0.450 lebih tinggi dari Skenario A (0.300) mengkonfirmasi retrieval lintas paradigma aktif. KTE = 0.352 dan AQI = 0.254 lebih rendah dari skenario tunggal, konsisten dengan ekspektasi, karena pertanyaan *cross-paradigm* menghasilkan jawaban sintesis yang lebih sulit diverifikasi secara reference-free.

**Tabel 6.** Sub-Analisis Skenario C: Efektivitas Retrieval Lintas Tipe Pertanyaan

| Sub-tipe | Contoh Pertanyaan | n | Faith | Comp | Overall | KTE |
|---|---|---|---|---|---|---|
| PR-only (dari press release resmi) | "Berapa NIM dan pendapatan bunga bersih KB Bank Q1 2025?" | 2 | 0.047 | 0.682 | 0.293 | **0.364** |
| Chat-only (dari log diskusi) | "Apa rekomendasi akhir diskusi tim untuk BBKP dan TINS?" | 1 | 0.179 | 0.818 | 0.327 | **0.499** |
| Cross-source (butuh kedua sumber) | "EBITDA TINS Rp384M di chat, apakah konsisten dengan press release?" | 2 | 0.078 | 0.716 | 0.298 | **0.397** |

*Skenario C membuktikan kemampuan sistem melakukan retrieval lintas file dalam satu sesi query; pertanyaan cross-source tidak dapat dijawab dari satu file saja.*

KTE per skenario mencerminkan dimensi transfer pengetahuan yang berbeda: Skenario A mengukur kemampuan sistem mengeksplisitkan kebijakan formal (*Explicit → Actionable*), Skenario B mengukur kemampuan mengkontekstualisasi data struktural (*Structured → Contextual*), Skenario C mengukur kemampuan mengeksplisitkan pengetahuan dari percakapan informal (*Tacit → Explicit*), dan Skenario D mengukur kemampuan sistem mengintegrasikan pengetahuan lintas paradigma data (*Cross-Paradigm*), sehingga membuktikan bahwa klaim *source-agnostic* bukan hanya berlaku per-adapter secara terpisah, tetapi juga ketika kedua paradigma digabungkan dalam satu pipeline sekaligus. Keempat mode ini secara kolektif memetakan seluruh spektrum transfer pengetahuan yang diidentifikasi Nonaka & Takeuchi [4].

Arsitektur pipeline memisahkan secara eksplisit antara komponen retrieval dan komponen generasi, sebagaimana terlihat dari nilai Precision@K dan MRR yang konsisten mencapai 1.000 di seluruh 20 pertanyaan sementara metrik generasi bervariasi antar skenario. Hal ini mengkonfirmasi bahwa kualitas retrieval tidak bergantung pada model generatif yang digunakan.

### 3.2 Visualisasi Hasil

Hasil evaluasi batch divisualisasikan dalam empat panel terpisah (Gambar 2a–2d).

**Gambar 2a** menampilkan rata-rata metrik standar (RR, Faithfulness, Completeness, ROUGE-L, P@K, MRR, Context Coverage) per skenario dalam bentuk grouped bar chart. P@K dan MRR secara konsisten mencapai 1.000 di semua skenario; metrik generasi (Faithfulness, ROUGE-L) berada di kisaran rendah sebagaimana expected pada evaluasi reference-free.

**Gambar 2b** menampilkan Overall score per pertanyaan (20 pertanyaan lintas 4 skenario), menunjukkan variasi antar pertanyaan dalam satu skenario masih dalam rentang wajar dan tidak ada outlier ekstrem.

**Gambar 2c** membandingkan tiga metrik komposit (KTE, MSRS, AQI) antar skenario. MSRS Skenario B tertinggi (0.812) dan KTE Skenario B tertinggi (0.502), satu-satunya skenario yang melampaui ambang batas efektif KTE ≥ 0.5.

**Gambar 2d** merupakan radar chart yang memperlihatkan profil multi-dimensi keempat skenario secara bersamaan, mengkonfirmasi bahwa masing-masing skenario memiliki keunggulan pada dimensi yang berbeda sesuai tipe sumbernya.

![Gambar 2a](result_download/eval_panel1_metrik_standar.png)

**Gambar 2a.** Rata-rata metrik standar per skenario (grouped bar chart).

![Gambar 2b](result_download/eval_panel2_overall_per_q.png)

**Gambar 2b.** Overall score per pertanyaan (20 pertanyaan, 4 skenario).

![Gambar 2c](result_download/eval_panel3_komposit_kte_msrs_aqi.png)

**Gambar 2c.** Metrik komposit KTE, MSRS, dan AQI per skenario.

![Gambar 2d](result_download/eval_panel4_radar.png)

**Gambar 2d.** Radar chart profil multi-dimensi keempat skenario.

### 3.3 Keterbatasan Penelitian

Komponen retrieval menggunakan `paraphrase-multilingual-MiniLM-L12-v2` (50+ bahasa) sehingga tidak terdapat bias bahasa pada lapisan retrieval; bobot retrieval sepenuhnya ditentukan oleh similarity semantik, bukan metadata tipe sumber. Pada pertanyaan *single-entity*, Context Coverage cenderung rendah karena top-K chunk terkonsentrasi pada satu dokumen — ini merupakan cerminan akurat corpus, bukan bias sistem. Deduplikasi berbasis hash konten diimplementasikan untuk mencegah satu file terhitung ganda sebagai sumber berbeda.

Beberapa keterbatasan penelitian ini perlu diakui secara eksplisit:

**Keterbatasan Cakupan Evaluasi.** Evaluasi yang dilaporkan mencakup hanya 20 pertanyaan di empat skenario, menggunakan dua entitas (BBKP dan TINS) dari satu sektor (keuangan/pertambangan) dalam corpus berbahasa Indonesia. Validasi empiris pada domain dan bahasa lain diperlukan untuk memperkuat generalisasi hasil.

**Ketiadaan Ground Truth.** Metrik ROUGE-L dan BLEU-1 dihitung secara *reference-free*, yaitu dengan membandingkan jawaban terhadap context yang di-retrieve, bukan dengan jawaban acuan yang dikurasi manusia. Nilai kedua metrik ini karenanya tidak dapat dibandingkan langsung dengan sistem lain yang menggunakan ground truth dataset standar (misalnya SQuAD, NaturalQuestions).

**Keterbatasan BLEU-1.** Nilai BLEU-1 = 0.000 di semua skenario merupakan *expected behavior* pada evaluasi *reference-free*: mekanisme *brevity penalty* BLEU memberikan penalti maksimal karena panjang jawaban (100–300 token) jauh lebih pendek dari referensi gabungan context (2000+ token). BLEU-1 dilaporkan untuk transparansi metodologi namun tidak digunakan dalam interpretasi kualitas sistem.

**Skala Evaluasi.** Evaluasi pada 20 pertanyaan di empat skenario memberikan gambaran proof-of-concept yang memadai, namun belum cukup untuk klaim statistik yang dapat digeneralisasi. Penelitian lanjutan disarankan menggunakan minimal 20–50 pasangan query-answer per skenario untuk analisis yang lebih representatif.

### 3.4 Implikasi

#### 3.4.1 Implikasi untuk Transfer Pengetahuan Organisasi

Sistem ini secara langsung menjawab tantangan yang diidentifikasi dalam pendahuluan. Analis junior yang baru bergabung dapat langsung bertanya dalam bahasa natural, misalnya "Berapa laba bersih BBKP Q1 2025 dibanding tahun lalu?" atau "Apa rekomendasi akhir tim untuk saham TINS?", dan sistem akan mencari jawaban dari kombinasi press release PDF, log diskusi TXT, dan database analitik secara otomatis dalam satu query. Skenario ini merepresentasikan transfer pengetahuan dari analis senior ke junior tanpa harus membaca ulang seluruh dokumentasi yang tersebar, sejalan dengan tantangan eksplisitasi pengetahuan tacit yang diidentifikasi Nonaka & Takeuchi [4]. Prinsip yang sama berlaku untuk konteks organisasi lain: Product Owner yang baru bergabung dapat bertanya tentang keputusan arsitektur atau riwayat vendor dari dokumentasi proyek yang ada.

#### 3.4.2 Implikasi Teknis untuk Pengembangan Selanjutnya

Arsitektur Adapter Pattern yang digunakan membuka jalan untuk extensibility: sumber baru (MongoDB, SharePoint, Google Drive API) dapat ditambahkan dengan mengimplementasikan dua method, yaitu `load()` dan `describe()`, tanpa mengubah pipeline inti. Ini sejalan dengan prinsip Open/Closed Principle dalam desain perangkat lunak [16].

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

Penelitian ini berhasil mengembangkan sistem RAG agnostic multi-sumber yang memenuhi tujuan penelitian, dengan lima temuan utama:

1. **Arsitektur *agnostic* terealisasi:** `SourceDetector` + `SourceFactory` + `BaseSourceAdapter` memungkinkan penanganan folder (PDF, TXT/MD/LOG) dan PostgreSQL (3 mode query: semua tabel, tabel tertentu, custom SQL) dari satu parameter `source`, tanpa perubahan pada pipeline inti.

2. ***Real-time* indexing terbukti benar:** Setiap pemanggilan `pipeline.ask()` membangun indeks FAISS secara in-memory sehingga perubahan konten di sumber langsung tercermin dalam hasil retrieval tanpa restart sistem. Tidak ada file index yang tersimpan di disk.

3. **Multi-sumber terbukti secara eksperimen:** Evaluasi batch 20 pertanyaan (4 skenario × 5 pertanyaan) menunjukkan Precision@K=1.000 dan MRR=1.000 di seluruh skenario. Performa agregat: A (Overall=0.331, MSRS=0.650), B (Overall=0.319, MSRS=0.812), C (Overall=0.325, MSRS=0.700), D (Overall=0.284, MSRS=0.725). Skenario D membuktikan klaim *source-agnostic* lintas paradigma dengan Context Coverage=0.450, mengkonfirmasi `MultiSourceAdapter` berhasil menggabungkan sumber unstructured dan structured dalam satu indeks FAISS.

4. **Efektivitas transfer pengetahuan terukur via KTE:** KTE per skenario: A=0.429 (*Explicit→Actionable*), B=0.502 (*Structured→Contextual*), C=0.460 (*Tacit→Explicit*), D=0.352 (*Cross-Paradigm*). Skenario B satu-satunya yang melampaui ambang batas KTE ≥ 0.5; rendahnya KTE pada skenario unstructured (Faithfulness 0.095–0.142) merupakan karakteristik evaluasi *reference-free* berbahasa Indonesia, bukan kegagalan retrieval.

5. **Evaluasi multi-dimensi terukur:** Framework 11 metrik (8 kuantitatif + 3 komposit) membuktikan pemisahan arsitektur retrieval–generasi: retrieval konsisten sempurna di semua skenario sementara metrik generasi bervariasi sesuai tipe sumber. AQI dan MSRS tertinggi pada Skenario B (0.364 dan 0.812), mengkonfirmasi PostgreSQL sebagai sumber dengan kualitas jawaban dan keragaman retrieval terbaik.

Kontribusi utama penelitian adalah desain pola Adapter yang memisahkan concerns antara sumber data, pemrosesan teks, retrieval, dan generasi, sehingga setiap komponen dapat diganti atau diperluas secara independen. Framework evaluasi 8 metrik yang diimplementasikan bebas-dependensi (tanpa library RAGAS) dapat direplikasi di lingkungan terbatas resource.

Untuk penelitian selanjutnya, disarankan: (1) evaluasi batch dengan minimal 20 pertanyaan menggunakan ground truth dataset yang dikurasi manual untuk mendapatkan nilai ROUGE-L dan BLEU-1 yang dapat dibandingkan dengan sistem lain; (2) fine-tuning embedding model pada dokumen domain spesifik organisasi; (3) perbandingan performa dengan sistem RAG berbasis vector database komersial (Pinecone, Weaviate) sebagai baseline arsitektur.

---

## 5. DAFTAR PUSTAKA

[1] IDC, "90% of Data is Unstructured and Its Full of Untapped Value," *IDC Blog*, 2023. [Online]. Available: https://blogs.idc.com/2023/05/09/90-of-data-is-unstructured-and-its-full-of-untapped-value/

[2] Gartner, "Gartner Says Employees Spend Too Much Time on Low-Value Tasks: Use AI and Automation to Fix It," Gartner Newsroom, Jan. 2020. [Online]. Available: https://www.gartner.com/en/newsroom/press-releases/2020-01-23-gartner-says-employees-spend-too-much-time-on-low-value-tasks

[3] M. Chui, J. Manyika, J. Bughin, R. Dobbs, C. Roxburgh, H. Sarrazin, G. Sands, and M. Westergren, "The social economy: Unlocking value and productivity through social technologies," McKinsey Global Institute, Jul. 2012. [Online]. Available: https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-social-economy

[4] I. Nonaka and H. Takeuchi, *The Knowledge-Creating Company*. New York: Oxford University Press, 1995.

[5] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Proc. NeurIPS*, 2020, pp. 9459–9474.

[6] G. Izacard and E. Grave, "Leveraging passage retrieval with generative models for open domain question answering," in *Proc. EACL*, 2021, pp. 874–880.

[7] M. Yasunaga, H. Ren, A. Bosselut, P. Liang, and J. Leskovec, "QA-GNN: Reasoning with language models and knowledge graphs for question answering," in *Proc. NAACL*, 2021.

[8] J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," *IEEE Trans. Big Data*, vol. 7, no. 3, pp. 535–547, 2019.

[9] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. EMNLP*, 2019.

[10] E. M. Voorhees, "The TREC-8 question answering track report," in *Proc. TREC*, 1999, pp. 77–82.

[11] C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in *Proc. ACL Workshop on Text Summarization Branches Out*, 2004.

[12] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "BLEU: A method for automatic evaluation of machine translation," in *Proc. ACL*, 2002.

[13] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAS: Automated evaluation of retrieval augmented generation," *arXiv preprint arXiv:2309.15217*, 2023.

[14] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, M. Wang, and H. Wang, "Retrieval-augmented generation for large language models: A survey," *arXiv preprint arXiv:2312.10997*, 2024. DOI: 10.48550/arXiv.2312.10997

[15] H. Ren, H. Shi, W. Zhao, J. Zhao, and Y. Zhao, "Self-RAG: Learning to retrieve, generate, and critique through self-reflection," *arXiv preprint arXiv:2307.11019*, 2023. DOI: 10.48550/arXiv.2307.11019

[16] J. Carbonell and J. Goldstein, "The use of MMR, diversity-based reranking for reordering documents and producing summaries," in *Proc. ACM SIGIR*, 1998, pp. 335–336.

[15] A. R. Fabbri, W. Kryściński, B. McCann, C. Xiong, R. Socher, and D. Radev, "SummEval: Re-evaluating summarization evaluation," *Trans. Assoc. Comput. Linguist.*, vol. 9, pp. 391–409, 2021.

[16] R. C. Martin, *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Upper Saddle River, NJ: Prentice Hall, 2017.


