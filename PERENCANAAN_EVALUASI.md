# Perencanaan Evaluasi RAG - Jurnal SINTA

> **Versi:** Pre-planning v3.5 — 6 Mei 2026
> **Status:** Draft jurnal selesai — MYJOURNAL.md (EN Springer-style) siap review dosen

---

## Arahan Dosen (Pak Ivan Siregar) — Chat 21-22 April 2026

### 1. Revisi Introduction — Urgency "Mengapa QA diperlukan?"
> *"Coba gali sedikit terkait: AI assistants, and customer support applications. Temukan limitation mereka terkait kebutuhan perusahaan sehingga QA menjadi dianggap perlu. Kita harus munculkan urgency, agar penelitian dianggap significant. Ulas di Introduction untuk menjawab: Mengapa harus QA?"*

**Action:** Tambahkan sub-argumen di Bab 1 tentang limitasi AI assistant umum (hallucination, lack of domain knowledge, no grounding) → sitasi paper terkait → QA dengan RAG sebagai solusi.

### 2. Munculkan 3 Kontribusi Eksplisit (gaya Springer)
> *"Setelah itu, munculkan 3 contributions."*

1. **Kontribusi 1:** Identifikasi faktor yang berpengaruh pada kualitas QA dalam **document-based enterprise service systems** (aspek domain, heterogenitas sumber, pola pertanyaan operasional)
2. **Kontribusi 2:** Rancangan model QA dengan RAG **Adapter Pattern multi-source**: `FolderSourceAdapter` (PDF, TXT) + `PostgreSQLAdapter` (relasional) + FAISS real-time in-memory
3. **Kontribusi 3:** Evaluasi empiris berbasis **data operasional nyata** — **5 skenario**, 8 metrik kuantitatif, 3 metrik komposit (KTE, MSRS, AQI)

> *"Coba lihat paper yang saya tulis, seperti itu gayanya"*
> → Referensi: https://link.springer.com/article/10.1007/s44443-025-00263-4

### 3. Data — Keputusan Final
> *"Jangan [sintetis], ada aspek Reproducible yang menjadi constraint lolos jurnal. Kalau sintetik dia tidak reproducible."*

> *"Masking data personal bisa?"* → **Jawaban: Bisa**
> *"Sometimes jurnal minta data diopen"* → Pertimbangan pemilihan jurnal target

> *"Atau encode saja ke ID autonumber, yang penting relationship bisa dimaintain"*
> → **Keputusan: Gunakan data proyek nyata (MOFIDS) + encode autonumber di akhir**

---

## Status
- [x] Pilih tabel database relevan
- [x] Dapat sample data nyata dari DBeaver (rfq, securities, firm_masters) → `data/mofids_sample.sql`
- [x] **Introduction baru ditulis** — urgency framing: *document-based service organizations*, turnover 2x/tahun, motivating scenario, 3 kontribusi eksplisit
- [x] Referensi [14] Gao et al. 2024 dan [15] Ren et al. 2023 ditambahkan ke DRAFT_JURNAL.md
- [x] Tentukan 5 skenario evaluasi — **finalisasi selesai**, 25 pertanyaan draft v2 tersedia
- [x] Siapkan layer ketiga — **keputusan final: gunakan chat Teams** (3 file: `cuplikan`, `cuplikan2`, `cuplikan-personal-message`). `change_logs` **tidak digunakan**.
- [x] Buat 25 pertanyaan evaluasi (5 per skenario) — draft v2 selesai
- [x] Jalankan evaluasi RAG — selesai 2 Mei 2026, hasil di `newresult/`
- [x] **Terminologi FR → PDF** diterapkan di seluruh draft (3 Mei 2026)
- [x] **Tabel 5 dipecah → Tabel 5a (retrieval) + Tabel 5b (generation/composite)** (3 Mei 2026)
- [x] **DRAFT_JURNAL_EN.md** dibuat sebagai versi bahasa Inggris lengkap
- [x] **Citation style** dikonversi ke author-year + DOI link di seluruh bibliography
- [x] **Domain masking** "lelang obligasi pemerintah" → BOND_SYS diterapkan
- [x] **MYJOURNAL** — content review selesai 6 Mei 2026: seluruh konten (5 skenario, 8 metrik, 3 komposit, Tabel 5a/5b, ablation, BOND_SYS masking) ✅ sesuai rencana
- [x] **MYJOURNAL body language** dikonversi ke **English** penuh (Springer-style) — 6 Mei 2026
- [x] **Token masking final** diterapkan di MYJOURNAL.md — 6 Mei 2026 (lihat tabel masking di bawah)
- [ ] Gambar 1 arsitektur → PNG ≥300 DPI (draft Mermaid tersedia di `GAMBAR_1_ARSITEKTUR.md`)
- [ ] Encode/masking data untuk lampiran jurnal
- [ ] Integrasi RAGAS (Faithfulness + Answer Relevance)
- [ ] Tambah std deviation di tabel ringkasan

---

## Token Masking Log — MYJOURNAL.md

> Catatan: Masking hanya berlaku untuk **MYJOURNAL.md** (versi publikasi). Google Colab dan notebook internal tetap menggunakan nama asli untuk keperluan demo.

| Nama Asli | Token di Paper | Alasan |
|---|---|---|
| MOFIDS | BOND_SYS | Nama sistem internal platform lelang obligasi |
| Direktorat Jenderal Pengelolaan Pembiayaan dan Risiko (DJPPR) | GOV_DEPT1 | Nama institusi pemerintah pemilik platform |
| LPDU (Lembaga Penjual Dealer Utama) / nama modul nyata | BOND_MOD | Nama modul internal platform |
| BS-SB | BOARD_TYPE_A | Kode board type internal platform |
| Buyback Cash | INSTRUMENT_TYPE_A | Nama produk instrumen internal |
| trade custody module | BOND_MOD_CUSTODY | Nama modul internal (pola BOND_MOD) |
| ETL incident February 2023 | ETL incident (anonymized period) | Tanggal spesifik dapat mengidentifikasi organisasi |
| NEWCORE-2442 | ISSUE_REF_2442 | Nomor tiket internal (sudah sebagian ter-mask) |

---

## Corpus 3-Layer

| Layer | Sumber | Status |
|---|---|---|
| **PDF/TXT** | `data/functional-reqiurement` (FR LPDU) | ✅ Ada |
| **Database** | MOFIDS (PostgreSQL) → `data/mofids_sample.sql` | ✅ Ada (20 RFQ, 10 Securities, 10 Firms, 10 Quotations, 10 Trades, 10 Trade Statuses, 11 Firm Default Params, 8 Fractions) |
| **Chat** | `cuplikan` (Grup besar 2022) + `cuplikan2` (Grup sedang 2025) + `cuplikan-personal-message` (PM Krisna↔Patresia 2022) | ✅ Ada — siap diparse |

---

## Tabel Database Relevan (dari MOFIDS)

### Core Lelang
- `request_for_quotations` — RFQ Preparation
- `rfq_source_securities` — Seri Source
- `rfq_destination_securities` — Seri Destination
- `rfq_firm_lists` — Daftar DU per lelang
- `rfq_user_lists` — User DU
- `rfq_broadcasts` — Notifikasi alokasi
- `quotations` — Entri kuotasi DU
- `trades` — Data settlement
- `trade_statuses` — Status transaksi

### Securities / Instrumen
- `securities_master_statics` — Data SUN/SBN (coupon, maturity, dll)
- `securities_master_dynamics` — Last price, accrued interest
- `securities_types` — Jenis SBN
- `benchmark_historicals` — Seri Benchmark (Staple Bonds)
- `coupon_details` — Detail kupon

### Firm / Investor
- `firm_masters` — Data Dealer Utama
- `firm_default_params` — Default parameter per DU
- `investors` — Data investor (Sub Registry)

### Konfigurasi / Admin
- `parameters` — Parameter lelang (waktu sesi, max/min price %)
- `workday_settings` — Settlement date
- `change_logs` — Audit trail
- `schedulers` / `scheduler_jobs` — Job otomatis ke PLTE/BI

## Insight dari Data Nyata (mofids_sample.sql)

### request_for_quotations (20 records, 2014–2023)
- `board_id`: `BC` (Buyback Cash), `BS` (Buyback Switch), `BS-SB` (Staple Bonds)
- `status`: mayoritas `Closed`, 1 record `Pending` (20180322-01)
- `auction_unit`: `Bio` (mayoritas) atau `Mio` (hanya BS-SB)
- `offering_parameter`: semua `Price`
- `max_price_percentage` / `min_price_percentage`: konsisten 150 / 30
- Jadwal sesi: 10:00–12:29 (sesi 1), 12:30–13:00 (sesi 2) — **konsisten dengan FR**
- `chat_flag`: 2 record `Y` (2014), sisanya `N`

### securities_master_statics (10 records)
- Mix obligasi korporat (BMRI, BJBR, WIKA, dll) + SPN pemerintah
- `listing_type`: Listed/Unlisted/Retail
- `coupon_type_id`: fixed rate
- `year_basis`: `30/360` atau `Actual/Actual`

### firm_masters (10 records)
- `firm_type_id`: 1 = Sekuritas, 2 = Bank, 3 = Kustodian
- `firm_id` format: `S-xxx` (Sekuritas), `B-xxx` (Bank), `C-xxx` (Kustodian)
- Sebagian besar `active_status = N` (data lama/historis)
- `firm_sid = 'SIDDUMMYPLTE001'` → sudah anonymous/dummy untuk PLTE

### Catatan untuk Encoding (saran dosen)
- `firm_id` sudah dalam format kode (`S-ASTRA`, `B-PANIN`) — **tidak perlu di-encode ulang**
- `firm_name` (nama perusahaan publik: BEI-listed) → **tidak sensitif**, bisa dipakai langsung
- `created_by` / `updated_by` berformat `XX_DJPU_ADM` / `XX_DJPU_MON` → username internal, perlu di-encode
- `contact_person` (nama individu) → **perlu di-encode** ke autonumber

---



### Skenario A — PDF Only
- **Sumber:** `functional-reqiurement` saja
- **Fokus:** Pertanyaan tentang alur proses bisnis lelang
- **Contoh Q:** *"Apa saja tahapan proses Buyback Cash?"*, *"Apa perbedaan sesi General dan Restricted?"*

### Skenario B — Database Only
- **Sumber:** Tabel MOFIDS saja
- **Fokus:** Pertanyaan data struktural/konfigurasi
- **Contoh Q:** *"Berapa default max price percentage?"*, *"Siapa saja DU yang terdaftar?"*

### Skenario C — PDF + Database (Multi-source)
- **Sumber:** FR dokumen + tabel MOFIDS
- **Fokus:** Pertanyaan yang butuh cross-reference dokumen dan data
- **Contoh Q:** *"Apakah konfigurasi sistem sudah sesuai spesifikasi FR untuk waktu sesi lelang?"*

### Skenario D — PDF + Database + Chat (Full Multi-source)
- **Sumber:** Semua layer
- **Fokus:** Pertanyaan kontekstual yang melibatkan diskusi tim
- **Contoh Q:** *"Berdasarkan diskusi tim dan FR, apa isu yang belum terimplementasi?"*

---

## Masking Nama Lembaga — `data/functional-reqiurement`

> **Status:** ✅ Diterapkan — April 2026  
> **Tujuan:** Menjaga anonimitas identitas sistem dan institusi agar dokumen FR dapat digunakan sebagai corpus publik jurnal.

### Skema Masking (applied)

| Term Asli | Token Masking | Keterangan |
|---|---|---|
| `MOFIDS` / `MOFiDS` | `BOND_SYS` | Nama sistem lelang obligasi |
| `MOFIDS Admin` | `BOND_SYS Admin` | Role admin sistem |
| `CTP` / `NEWCTP` | `CORE_MODULE` / `NEWCORE` | Platform/komponen inti |
| `LPDU` (prefix fungsi) | `BOND_MOD` | Modul Layanan Perdagangan |
| `BEI` / `IDX` | `EXCHANGE_ORG` | Bursa Efek Indonesia |
| `DJPPR` | `GOV_DEPT1` | Direktorat Jenderal terkait |
| `DJPU` | `GOV_DEPT2` | Direktorat Jenderal terkait |
| `Kementrian Keuangan` | `GOV_MINISTRY` | Kementerian penerbit kebijakan |
| `PLTE` | `EXT_SYS_1` | Sistem eksternal pelaporan |
| `DSS` | `EXT_SYS_2` | Sistem alokasi eksternal |
| `Bank Indonesia` | `CENTRAL_BANK` | Bank sentral settlement |
| `LPKSBN` / `LPKSUN` | `AUCTION_RP/AUCTION_RP2` | Laporan lelang |
| `Puspita Pratiwi` | `AUTHOR_01` | Nama penulis dokumen FR |

### Tidak Di-mask (terminologi domain umum)

`SUN`, `SBN`, `Dealer Utama`, `DU`, `Buyback Cash`, `Buyback Debt Switch`, `Simple Auction`, `Staple Bonds`, `settlement`, `maker-checker`, `RFQ`, `quotation`

---

## Strategi Encoding Data (saran dosen)

> *"Encode saja ke ID autonumber, yang penting relationship bisa dimaintain"*

Pendekatan:
- Nama firm → `firm_id` (integer autonumber)
- Nama securities → `sec_id` (integer autonumber)
- Nama instansi (DJPPR, dll) → `org_id` (integer autonumber)
- Relasi antar tabel **tetap valid** via foreign key ID
- Dikerjakan **setelah** evaluasi selesai, sebelum submit jurnal

---

## Arsitektur Corpus — 3 Layer Final

```
Corpus
├── Layer 1 — Dokumen (PDF/TXT)
│   └── data/functional-reqiurement        ← FR LPDU v1.0 (file TXT tunggal)
│       └── → FolderSourceAdapter → FAISS index: data/indices/
│
├── Layer 2 — Database (PostgreSQL → SQL)
│   └── data/mofids_sample.sql             ← 20 RFQ, 10 Securities, 10 Firms
│       └── → PostgreSQLAdapter (real-time query)
│
└── Layer 3 — Chat (Teams copy-paste → plain .txt)
    ├── Sumber mentah (RAW):
    │   ├── cuplikan                       ← Grup MOFIDS 2022 (raw Teams export)
    │   ├── cuplikan2                      ← Grup MOFIDS 2025 (raw Teams export)
    │   └── cuplikan-personal-message      ← Personal Dev_B↔PO_1 2022 (raw Teams export)
    │
    ├── Diproses oleh parse_chat_corpus.py:
    │   ├── data/processed_chats/chat_group_mofids.txt    ← 833 pesan, anonimisasi ✅
    │   ├── data/processed_chats/chat_group_mofids2.txt   ← 36 pesan, anonimisasi ✅
    │   └── data/processed_chats/chat_personal_mofids.txt ← 39 pesan, anonimisasi ✅
    │
    └── → FolderSourceAdapter (sama dengan Layer 1) → FAISS index: data/indices/

Catatan: Layer 1 + Layer 3 (chat processed) di-load bersama oleh FolderSourceAdapter.
Layer 2 di-load terpisah oleh PostgreSQLAdapter.
File mofids-personal-message.txt di root = backup sumber mentah (identik dengan cuplikan-personal-message).
```

### Keputusan Desain Corpus Layer 3

| Aspek | Keputusan | Alasan |
|---|---|---|
| **Format output** | Plain `.txt` (bukan FAISS terpisah) | Konsisten dengan format `adaro_analyst_chat.txt`; langsung di-load FolderSourceAdapter |
| **Folder output** | `data/processed_chats/` | 3 file `.txt` bersih per sumber |
| **Metadata header** | `--- GRUP: ---`, `--- Anggota: ---`, `--- Konteks: ---` | Konteks sumber tertanam di file, bisa diindeks sebagai bagian dokumen |
| **Format per baris** | `[DD/MM/YYYY, HH:MM:SS] Pseudonim: isi pesan` | Konsisten, mudah dibaca model |
| **Masking nama** | Lihat tabel **Daftar Anonimisasi Individu** di bawah | Anonimisasi pribadi sesuai `parse_chat_corpus.py` |
| **Script** | `parse_chat_corpus.py` | Parse Teams raw export → plain `.txt` bersih |

### Daftar Anonimisasi Individu (dari `parse_chat_corpus.py`)

> Semua nama asli dalam file chat diganti dengan pseudonim sebelum diindeks ke FAISS.

| Pseudonim | Nama Asli | Role |
|---|---|---|
| `PO_1` | Patresia Ratu Wetti Sitanggang | Product Owner |
| `PO_2` | Mesakh Dwi Putra | Product Owner |
| `PO_3` | Ivena Chindy Claudia | Product Owner |
| `PM_1` | Bondan Chaya Nugraha / Bondan Chahya Nugraha | Project Manager |
| `Dev_A` | Ardy Maulana | Developer |
| `Dev_B` | Krisna Dwi Setyaadi | Developer |
| `Dev_C` | Sheldy Rivaldi | Developer |
| `Dev_D` | Ezra Hutapea | Developer |
| `Dev_E` | Dhifa Irawan | Developer |
| `Dev_F` | Julio Lemena | Developer |
| `Dev_G` | Leslie Aula | Developer |
| `DevOps_1` | Sandy Agustinus Suherman / moonlay | DevOps |

> **Catatan:** Username alias Teams (misal `moonlay-sheldy`, `moonlay-krisna`) juga di-mask ke pseudonim yang sesuai. Nama panggilan tunggal (misal `krisna`, `ardy`) dalam isi pesan **tidak di-mask** agar konteks percakapan tetap terbaca — hanya nama lengkap (2+ kata) yang di-replace.

---

## 4 Skenario Evaluasi

> ⚠️ **REVISI v3.0 — Redesign menjadi 5 skenario**
> Lihat section "5 Skenario Evaluasi (Revised)" di bawah.

| Skenario | Sumber Aktif | Fokus |
|---|---|---|
| ~~**A**~~ | ~~PDF/TXT saja~~ | ~~Pemahaman alur bisnis dari dokumen FR~~ |
| ~~**B**~~ | ~~Database saja~~ | ~~Query data struktural/konfigurasi sistem~~ |
| ~~**C**~~ | ~~PDF + Database~~ | ~~Cross-reference dokumen & data aktual~~ |
| ~~**D**~~ | ~~PDF + Database + Chat~~ | ~~Konteks operasional + diskusi tim~~ |

---

## 5 Skenario Evaluasi (Revised — v3.0)

> **Reasoning redesign:** Skenario Chat-only diperlukan untuk membuktikan secara empiris bahwa
> Layer 3 (tacit knowledge) **tidak dapat berdiri sendiri** — membutuhkan FR dan DB sebagai fondasi.
> Tanpa isolasi ini, kontribusi incremental Layer 3 tidak dapat dibuktikan di hadapan reviewer.
> Urutan A→E mencerminkan eskalasi dari sumber paling lemah ke paling lengkap.

| Skenario | Adapter | Layer Aktif | n | Tipe Pertanyaan | Dimensi TK |
|---|---|---|---|---|---|
| **A** *(baru)* | FolderSourceAdapter | Layer 3 saja (Chat) | 5 | Tacit knowledge dari log diskusi tim | Tacit → Operational |
| **B** *(dulu A)* | FolderSourceAdapter | Layer 1 saja (FR PDF) | 5 | Alur bisnis & spesifikasi teknis FR | Explicit → Actionable |
| **C** *(dulu B)* | PostgreSQLAdapter | Layer 2 saja (DB) | 5 | Query data struktural/konfigurasi | Explicit → Structured |
| **D** *(dulu C)* | MultiSourceAdapter | Layer 1 + 2 (FR + DB) | 5 | Cross-reference dokumen & data aktual | Explicit → Cross-referenced |
| **E** *(dulu D)* | MultiSourceAdapter | Layer 1 + 2 + 3 (All) | 5 | Konteks operasional + diskusi tim | Cross-Paradigm |

**Source path per skenario:**
- **A:** `sample_data/chat/` — chat logs saja
- **B:** `sample_data/pdf/` — FR PDF saja
- **C:** `postgresql://...` — DB saja
- **D:** `sample_data/pdf/|postgresql://...` — FR + DB
- **E:** `sample_data/|postgresql://...` — semua layer

---

## 25 Pertanyaan Evaluasi (Draft v2 — Revised)

> Format: **[ID]** Pertanyaan → *[Jawaban referensi singkat]*

### Skenario A *(baru)* — Chat Only (Tacit Knowledge Layer 3)

> Pertanyaan yang **hanya bisa dijawab** dari corpus chat tim — tidak ada jawabannya di FR atau DB.
> Sumber: `chat_group_mofids.txt`, `chat_group_mofids2.txt`, `chat_personal_mofids.txt`

| ID | Pertanyaan | Referensi Jawaban Singkat |
|---|---|---|
| A1 | Apa masalah yang ditemukan tim saat proses submit quotation pada board BS-SB menjelang demo, dan bagaimana workaround sementara yang disepakati? | NEWCORE-2406: tombol Quote disable karena data belum load; workaround: tunggu network request selesai (semua 200), pastikan package terpilih |
| A2 | Berdasarkan log diskusi tim, mengapa fitur upload allocation tidak dapat didemonstrasikan pada sesi demo 27 Juli 2022? | File download allocation bermasalah setelah update; tim sepakat pakai file upload manual yang disiapkan sendiri sebagai workaround demo |
| A3 | Apa yang didiskusikan tim terkait isu NEWCORE-2442 dan apa status penyelesaiannya berdasarkan log percakapan? | Filter status WAITING belum ada di backend; NEWCORE-2423 dan NEWCORE-2420 sudah DONE; NEWCORE-2442 masih kurang backend untuk terima filter status WAITING |
| A4 | Apa keputusan teknis yang didiskusikan tim terkait jumlah desimal (digit) untuk last price dan offering price? | Last price & offering price bisa desimal sampai 5 angka di belakang koma; yang diinput = yang disimpan; offering parameter tidak mempengaruhi input, hanya berpengaruh ke sistem PDS |
| A5 | Apa status implementasi fitur amend pada modul trade custody berdasarkan diskusi tim Februari 2025? | Fitur amend trade custody masih IN PROGRESS; bug confirm amend: logic row baru vs update row lama tidak konsisten; kolom CASH in/CASH out hilang di env dev |

### Skenario B *(dulu A)* — FR Only (Alur Bisnis FR LPDU)

| ID | Pertanyaan | Referensi Jawaban Singkat |
|---|---|---|
| B1 | Apa saja tahapan utama proses Buyback Cash dalam sistem MOFIDS? | Preparation → Session → Quotation → Allocation |
| B2 | Apa perbedaan antara sesi General dan sesi Restricted dalam lelang LPKSBN? | General: semua DU; Restricted: DU tertentu saja |
| B3 | Siapa saja pihak yang terlibat dalam proses persetujuan (approval) pembuatan RFQ? | Maker (GOV_DEPT) + Checker (GOV_DEPT) |
| B4 | Apa persyaratan teknis untuk fitur Upload Allocation berdasarkan FR? | Format file, validasi kolom, status RFQ harus active |
| B5 | Bagaimana mekanisme pengiriman notifikasi broadcast kepada Dealer Utama dalam FR? | Otomatis setelah allocation disetujui; via sistem notifikasi internal |

### Skenario C *(dulu B)* — Database Only (Data Struktural MOFIDS)

| ID | Pertanyaan | Referensi Jawaban Singkat |
|---|---|---|
| C1 | Berapa nilai default max_price_percentage dan min_price_percentage pada lelang MOFIDS? | 150% dan 30% |
| C2 | Apa saja kombinasi fraction_type dan fraction_digit yang tersedia untuk tipe Price? | Price: digit 2 (0.05), digit 3 (0.002), digit 5 (0.03125) |
| C3 | Apa perbedaan auction_unit antara board BS-SB dibanding board BS dan BC? | BS-SB: `Mio`; BS/BC: `Bio` |
| C4 | Firma mana saja yang memiliki `is_active = Y` di firm_default_params dan apa kode custody-nya? | BBTN→BTANIDJA, BANZ→ANZBIDJX, BBCA→CENAIDJA, BBII→IBBKIDJA, BBNI→BNINIDJA, BDMN→BDINIDJA, BHNS→BNIAIDJA, BMDR→BMRIIDJA |
| C5 | Dari semua quotation pada RFQ 20140327-01, berapa yang status `is_allocated = Y`? | 5 quotation allocated (CBNA 3x, SCBI 2x) |

### Skenario D *(dulu C)* — FR + DB (Cross-reference)

| ID | Pertanyaan | Referensi Jawaban Singkat |
|---|---|---|
| D1 | Apakah jam sesi pada data RFQ aktual sudah sesuai spesifikasi FR untuk jam operasional? | Ya: FR menyebut 10:00–12:29 (S1), 12:30–13:00 (S2); DB konsisten |
| D2 | Board type apa saja yang didefinisikan dalam FR, dan berapa yang sudah ada di database? | FR: BC, BS, BS-SB, SA; DB: BC (1), BS (17), BS-SB (1), BS-SB (1) |
| D3 | Apakah konfigurasi offering_parameter pada semua RFQ konsisten dengan ketentuan FR? | Ya: semua `Price`, FR mendefinisikan `Price` sebagai parameter standar |
| D4 | Berdasarkan FR dan data aktual, bagaimana settlement_date dihitung untuk board BS? | FR: T+2 dari event_date; DB: `event_date + 1` trading day via workday_settings |
| D5 | Berdasarkan FR dan data aktual, apakah offering_digit pada RFQ konsisten dengan fraction_masters? | Ya: semua RFQ pakai digit=2, fraction_masters mendefinisikan Price digit=2 → fraction=0.05 (konsisten) |

### Skenario E *(dulu D)* — FR + DB + Chat (Full Multi-source)

| ID | Pertanyaan | Referensi Jawaban Singkat |
|---|---|---|
| E1 | Berdasarkan diskusi tim dan FR, apa bug yang ditemukan pada submit quotation BS-SB dan bagaimana solusinya? | NEWCORE-2406: tombol Quote disable karena data belum load; solusi: tunggu network request selesai |
| E2 | Apa keputusan teknis terkait `offering digit` dan validasi hardcode yang didiskusikan tim? | Decimal bisa 5 digit; nilai yang diinput = yang disimpan; tidak tergantung offering_parameter |
| E3 | Berdasarkan log diskusi, apa isu pada NEWCORE-2442 dan bagaimana penyelesaiannya? | Filter status WAITING belum ada di backend; solusi: filter di UI saja, butuh backend update |
| E4 | Berdasarkan FR, data sistem, dan diskusi tim — bagaimana alur upload allocation, dan apa perbedaan antara spesifikasi FR dengan kondisi aktual yang ditemukan tim? | FR: format file + validasi + status RFQ harus active; DB: quotations dengan `is_allocated = Y`; Chat: bug file download bermasalah saat demo — tim siapkan file upload manual sebagai workaround |
| E5 | Berdasarkan seluruh sumber, apa status implementasi fitur amend pada modul trade custody? | FR mendefinisikan amend; DB: trade_statuses semua `Success Report` (normal flow); Chat (2025): bug amend logic row baru vs update — IN PROGRESS, belum resolve |

---

---

## 20 Pertanyaan Evaluasi (Draft v1)

> ⚠️ **OBSOLETE — diganti oleh "25 Pertanyaan Evaluasi (Draft v2 — Revised)" di atas**



| ID | Pertanyaan | Referensi Jawaban Singkat |
|---|---|---|
| A1 | Apa saja tahapan utama proses Buyback Cash dalam sistem MOFIDS? | Preparation → Session → Quotation → Allocation |
| A2 | Apa perbedaan antara sesi General dan sesi Restricted dalam lelang LPKSBN? | General: semua DU; Restricted: DU tertentu saja |
| A3 | Siapa saja pihak yang terlibat dalam proses persetujuan (approval) pembuatan RFQ? | Maker (DJPPR) + Checker (DJPPR) |
| A4 | Apa persyaratan teknis untuk fitur Upload Allocation berdasarkan FR? | Format file, validasi kolom, status RFQ harus active |
| A5 | Bagaimana mekanisme pengiriman notifikasi broadcast kepada Dealer Utama dalam FR? | Otomatis setelah allocation disetujui; via sistem notifikasi internal |

### Skenario B — Database Only (Data Struktural MOFIDS)

| ID | Pertanyaan | Referensi Jawaban Singkat |
|---|---|---|
| B1 | Berapa nilai default max_price_percentage dan min_price_percentage pada lelang MOFIDS? | 150% dan 30% |
| B2 | Apa saja kombinasi fraction_type dan fraction_digit yang tersedia untuk tipe Price? | Price: digit 2 (0.05), digit 3 (0.002), digit 5 (0.03125) |
| B3 | Apa perbedaan auction_unit antara board BS-SB dibanding board BS dan BC? | BS-SB: `Mio`; BS/BC: `Bio` |
| B4 | Firma mana saja yang memiliki `is_active = Y` di firm_default_params dan apa kode custody-nya? | BBTN→BTANIDJA, BANZ→ANZBIDJX, BBCA→CENAIDJA, BBII→IBBKIDJA, BBNI→BNINIDJA, BDMN→BDINIDJA, BHNS→BNIAIDJA, BMDR→BMRIIDJA |
| B5 | Dari semua quotation pada RFQ 20140327-01, berapa yang status `is_allocated = Y`? | 5 quotation allocated (CBNA 3x, SCBI 2x) |

### Skenario C — PDF + Database (Cross-reference)

| ID | Pertanyaan | Referensi Jawaban Singkat |
|---|---|---|
| C1 | Apakah jam sesi pada data RFQ aktual sudah sesuai spesifikasi FR untuk jam operasional? | Ya: FR menyebut 10:00–12:29 (S1), 12:30–13:00 (S2); DB konsisten |
| C2 | Board type apa saja yang didefinisikan dalam FR, dan berapa yang sudah ada di database? | FR: BC, BS, BS-SB, SA; DB: BC (1), BS (17), BS-SB (1), BS-SB (1) |
| C3 | Apakah konfigurasi offering_parameter pada semua RFQ konsisten dengan ketentuan FR? | Ya: semua `Price`, FR mendefinisikan `Price` sebagai parameter standar |
| C4 | Berdasarkan FR dan data aktual, bagaimana settlement_date dihitung untuk board BS? | FR: T+2 dari event_date; DB: `event_date + 1` trading day via workday_settings |
| C5 | Berdasarkan FR dan data aktual, apakah offering_digit pada RFQ konsisten dengan fraction_masters? | Ya: semua RFQ pakai digit=2, fraction_masters mendefinisikan Price digit=2 → fraction=0.05 (konsisten) |

### Skenario D — PDF + Database + Chat (Full Multi-source)

| ID | Pertanyaan | Referensi Jawaban Singkat |
|---|---|---|
| D1 | Berdasarkan diskusi tim dan FR, apa bug yang ditemukan pada submit quotation BS-SB dan bagaimana solusinya? | NEWCTP-2406: tombol Quote disable karena data belum load; solusi: tunggu network request selesai |
| D2 | Apa keputusan teknis terkait `offering digit` dan validasi hardcode yang didiskusikan tim? | Decimal di-hardcode 2 digit; seharusnya mengikuti `offering_parameter` dari DB — perlu fix |
| D3 | Berdasarkan log diskusi, apa isu pada NEWCORE-2442 dan bagaimana penyelesaiannya? | Filter status WAITING belum ada di backend; solusi: filter di UI saja (PDS), butuh backend update |
| D4 | Berdasarkan FR, data sistem, dan diskusi tim — bagaimana alur upload allocation, dan apa perbedaan antara spesifikasi FR dengan kondisi aktual yang ditemukan tim? | FR: format file + validasi + status RFQ harus active; DB: quotations dengan `is_allocated = Y`; Chat: bug file download bermasalah saat demo — tim siapkan file upload manual sebagai workaround |
| D5 | Berdasarkan seluruh sumber, apa status implementasi fitur amend pada modul trade custody? | FR mendefinisikan amend; DB: trade_statuses semua `Success Report` (normal flow); Chat (2025): bug amend logic row baru vs update — IN PROGRESS, belum resolve |

---

## Metrik Evaluasi

> **Status saat ini:** Metrik yang diimplementasikan di notebook adalah token-overlap based (tidak pakai RAGAS).
> Gap kritis: tidak ada ground truth dan tidak ada LLM judge.
> **Target:** Tambahkan ground truth 5 pertanyaan Skenario E + integrasi RAGAS untuk Faithfulness & Relevance.

### Metrik yang sudah diimplementasikan (token-overlap)

| Metrik | Cara Hitung | Validitas |
|---|---|---|
| **Retrieval Relevance** | Cosine similarity query vs rata-rata top-K chunk embeddings | ✅ Valid |
| **Answer Faithfulness** | F1 token overlap antara jawaban dan retrieved context | ⚠️ Approx (bukan LLM judge) |
| **Answer Completeness** | Keyword coverage pertanyaan dalam jawaban | ⚠️ Approx |
| **ROUGE-L** | LCS-based vs retrieved chunks (bukan gold answer) | ⚠️ Self-referential tanpa ground truth |
| **BLEU-1** | Unigram precision vs retrieved chunks | ⚠️ Self-referential tanpa ground truth |
| **Precision@K** | Proporsi chunk dengan similarity ≥ threshold | ✅ Valid |
| **MRR** | Rank of first chunk above threshold | ✅ Valid |
| **Context Coverage** | Keragaman sumber file di retrieved chunks | ✅ Valid |

### Metrik Komposit (diimplementasikan)

| Metrik | Formula | Interpretasi |
|---|---|---|
| **KTE** | `(Faithfulness + Completeness) / 2` | Efektivitas transfer pengetahuan ke jawaban |
| **MSRS** | `(P@K + Context Coverage) / 2` | Kualitas retrieval multi-source |
| **AQI** | `(Faithfulness + Completeness + ROUGE-L) / 3` | Kualitas linguistik jawaban |

### Gap yang harus diperbaiki sebelum submit

| Gap | Risiko | Fix |
|---|---|---|
| ~~Tidak ada ground truth answers~~ | ~~ROUGE-L & BLEU tidak valid~~ | ✅ **FIXED** — `GROUND_TRUTH_HYBRID` 5 jawaban manual, pass ke `run_batch(ground_truths=...)` |
| Tidak ada LLM-as-judge | Faithfulness approx saja | Integrasi RAGAS (Faithfulness + Answer Relevance) |
| n=5 per skenario | Terlalu kecil untuk klaim statistik | Naikkan ke 10, atau tambahkan catatan limitasi eksplisit |
| Tidak ada std deviation | Hasil point estimate tidak meyakinkan | Tambahkan std dev di tabel ringkasan |

---

## Roadmap Update Notebook (Bertahap)

| Tahap | Task | Status |
|---|---|---|
| **1** | Update `PERENCANAAN_EVALUASI.md` — redesign 5 skenario + 25 pertanyaan | ✅ Selesai (v3.0) |
| **2** | Tambah **Skenario A (Chat only)** di notebook — cell baru sebelum Skenario B | ✅ Selesai |
| **3** | Rename Skenario B→C, C→D, D→E di notebook (cell label + variabel + print) | ✅ Selesai |
| **4** | Fix `SOURCE_FR` Skenario B → `sample_data/pdf/` (bukan `sample_data/`) | ✅ Selesai |
| **5** | Update Ablation Study — tambah konfigurasi Chat-only sebagai baseline | ✅ Selesai |
| **6** | Buat **5 ground truth answers** untuk Skenario E | ✅ Selesai — cell `GROUND_TRUTH_HYBRID` di notebook |
| **7** | Tambahkan evaluasi RAGAS (Faithfulness + Relevance) minimal untuk Skenario E | ⬜ |
| **8** | Tambahkan std deviation di tabel ringkasan | ⬜ |
| **9** | Update arsitektur diagram & tabel skenario di markdown header notebook | ⬜ |
| **10** | Re-run semua skenario A–E + Ablation di Google Colab | ✅ Selesai — hasil di `newresult/` 2 Mei 2026 |

---

## Rencana Script

### `parse_chat_corpus.py` — Sudah dibuat
- Input: `cuplikan`, `cuplikan2`, `cuplikan-personal-message`
- Output: `data/processed_chats/` (3 file TXT bersih)
- Masking: nama individu → pseudonim

### Pipeline Evaluasi
```
Notebook: QA_RAG_AgnosticSource.ipynb
  ↓
load_documents() → 3 layer sumber
  ↓
Untuk setiap skenario (A, B, C, D, E):
  ↓
Jalankan 5 pertanyaan
  ↓
Hitung metrik (token-overlap + RAGAS target)
  ↓
Tabel hasil evaluasi → Bab 4 DRAFT_JURNAL.md
```

---

## Status & Next Steps

| # | Task | Status |
|---|---|---|
| 1 | Introduction baru (urgency + 3 kontribusi) | ✅ Selesai |
| 2 | Abstract (ID + EN) diperbarui | ✅ Selesai |
| 3 | Corpus Layer 1 (FR LPDU) — masking selesai | ✅ Selesai |
| 4 | Corpus Layer 2 (MOFIDS DB SQL) — 8 tabel, masking selesai | ✅ Selesai |
| 5 | Corpus Layer 3 (Chat 908 pesan) — parse + masking selesai | ✅ Selesai |
| 6 | SOURCE_ALL notebook diupdate ke `sample_data/` | ✅ Selesai |
| 7 | Q18 notebook difix: NEWCTP → NEWCORE | ✅ Selesai |
| 8 | Q17 corpus fix: klarifikasi offering digit ditambah ke chat | ✅ Selesai |
| 9 | Q20 corpus fix: status amend IN PROGRESS ditambah ke chat | ✅ Selesai |
| 10 | Redesign 5 skenario + 25 pertanyaan (v3.0) | ✅ Selesai |
| 11 | Tambah Skenario A (Chat only) di notebook | ✅ Selesai |
| 12 | Rename skenario B–E di notebook | ✅ Selesai |
| 13 | Buat 5 ground truth answers untuk Skenario E | ✅ Selesai — `GROUND_TRUTH_HYBRID` di notebook |
| 14 | Re-run Skenario E (reference-based) + Ablation di Colab | ✅ Selesai — 2 Mei 2026 |
| 15 | Re-run semua skenario A–E final di Google Colab | ✅ Selesai — 2 Mei 2026, `newresult/evaluasi_multisumber_20260502_162002.csv` |
| 16 | Analisis hasil → tulis Bab 2–4 DRAFT_JURNAL.md | ✅ Selesai |
| 17 | **Terminologi FR → PDF** di seluruh draft (ID + EN) | ✅ Selesai — 3 Mei 2026 |
| 18 | **Tabel 5 → 5a + 5b** (retrieval + generation terpisah) | ✅ Selesai — 3 Mei 2026 |
| 19 | **DRAFT_JURNAL_EN.md** (versi Inggris lengkap) | ✅ Selesai — 3 Mei 2026 |
| 20 | **Citation style** author-year + DOI link bibliography | ✅ Selesai |
| 21 | **Domain masking** BOND_SYS konsisten | ✅ Selesai |
| 22 | **Gambar 1** — draft Mermaid di `GAMBAR_1_ARSITEKTUR.md` | ⬜ Perlu render PNG ≥300 DPI |
| 23 | Kirim draft ke dosen untuk review | ⬜ Next step |
| 24 | Integrasi RAGAS (Faithfulness + Answer Relevance) | ⬜ Opsional |
| 25 | Tambah std deviation di tabel ringkasan | ⬜ Opsional |
| 26 | Encoding autonumber data untuk lampiran jurnal | ⬜ Sebelum submit |

---

## Diskusi Terbuka (Perlu Konfirmasi)

1. **Ground truth Skenario E:** ✅ Selesai — 5 jawaban referensi di cell `GROUND_TRUTH_HYBRID`, pass ke `run_batch(ground_truths=GROUND_TRUTH_HYBRID)`. Re-run Skenario E di Colab untuk ROUGE-L & BLEU reference-based.
2. **RAGAS integration:** Minimal untuk Faithfulness & Answer Relevance pada Skenario E ⬜
3. **Layer 1 path:** `sample_data/pdf/` — pastikan folder `pdf/` di Google Drive berisi FR file ✅

