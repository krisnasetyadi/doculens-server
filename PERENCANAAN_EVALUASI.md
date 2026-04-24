# Perencanaan Evaluasi RAG - Jurnal SINTA

> **Versi:** Pre-planning v2.0 — April 2026
> **Status:** Draft untuk diskusi lanjutan

---

## Arahan Dosen (Pak Ivan Siregar) — Chat 21-22 April 2026

### 1. Revisi Introduction — Urgency "Mengapa QA diperlukan?"
> *"Coba gali sedikit terkait: AI assistants, and customer support applications. Temukan limitation mereka terkait kebutuhan perusahaan sehingga QA menjadi dianggap perlu. Kita harus munculkan urgency, agar penelitian dianggap significant. Ulas di Introduction untuk menjawab: Mengapa harus QA?"*

**Action:** Tambahkan sub-argumen di Bab 1 tentang limitasi AI assistant umum (hallucination, lack of domain knowledge, no grounding) → sitasi paper terkait → QA dengan RAG sebagai solusi.

### 2. Munculkan 3 Kontribusi Eksplisit (gaya Springer)
> *"Setelah itu, munculkan 3 contributions."*

1. **Kontribusi 1:** Identifikasi faktor yang berpengaruh pada kualitas QA dalam **document-based enterprise service systems** (aspek domain, heterogenitas sumber, pola pertanyaan operasional)
2. **Kontribusi 2:** Rancangan model QA dengan RAG **Adapter Pattern multi-source**: `FolderSourceAdapter` (PDF, TXT) + `PostgreSQLAdapter` (relasional) + FAISS real-time in-memory
3. **Kontribusi 3:** Evaluasi empiris berbasis **data operasional nyata** — 4 skenario, 8 metrik kuantitatif, 3 metrik komposit (KTE, MSRS, AQI)

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
- [ ] Tentukan 4 skenario evaluasi (draft sudah ada, perlu finalisasi + update corpus ke MOFIDS)
- [ ] Siapkan layer ketiga — **pertimbangkan `change_logs` sebagai pengganti chat logs**
- [ ] Buat 20 pertanyaan evaluasi (5 per skenario)
- [ ] Jalankan evaluasi RAG
- [ ] Encode/masking data untuk lampiran jurnal

---

## Corpus 3-Layer

| Layer | Sumber | Status |
|---|---|---|
| **PDF/TXT** | `data/functional-reqiurement` (FR LPDU) | ✅ Ada |
| **Database** | MOFIDS (PostgreSQL) → `data/mofids_sample.sql` | ✅ Ada (20 RFQ, 10 Securities, 10 Firms, 10 Quotations, 10 Trades, 10 Trade Statuses, 11 Firm Default Params, 8 Fractions) |
| **Chat** | Chat logs proyek (dicari) | 🔍 Sedang dicari |

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
│   └── data/functional-reqiurement/        ← FR LPDU (Spesifikasi sistem MOFIDS)
│       └── → FAISS index: data/indices/
│
├── Layer 2 — Database (PostgreSQL → SQL)
│   └── data/mofids_sample.sql              ← 20 RFQ, 10 Securities, 10 Firms
│       └── → PostgreSQLAdapter (real-time query)
│
└── Layer 3 — Chat (Teams copy-paste)
    ├── cuplikan                            ← Grup MOFIDS 2022 (~8.600 baris)
    ├── cuplikan2                           ← Grup MOFIDS 2025 (~457 baris)
    └── cuplikan-personal-message           ← Personal Krisna↔Patresia 2022 (~287 baris)
        └── → parse → thread-chunk → FAISS: data/chat_indices/
```

### Keputusan Desain Corpus Layer 3

| Aspek | Keputusan | Alasan |
|---|---|---|
| **Penggabungan** | Pisah metadata, index bersama | Traceability per sumber dipertahankan |
| **Chunking** | Per thread (window 30 menit) | Konteks percakapan lebih koheren untuk RAG |
| **Folder output** | `data/chat_indices/` (via `source/`) | Konsisten dengan pipeline `load_chat_documents()` |
| **Metadata `source`** | `chat_group_mofids`, `chat_group_mofids2`, `chat_personal_mofids` | Untuk filter per sumber di Skenario D jika diperlukan |
| **Masking nama** | PO: `Patresia→PO_1`, `Mesakh→PO_2`, `Ivena→PO_3` · PM: `Bondan→PM_1` · Dev: `Ardy→Dev_A`, `Krisna→Dev_B`, `Sheldy→Dev_C`, `Ezra→Dev_D`, `Dhifa→Dev_E`, `Julio→Dev_F`, `Leslie→Dev_G` · DevOps: `Sandy→DevOps_1` | Anonimisasi pribadi |
| **Script** | `parse_chat_corpus.py` | Parser Teams format → Documents → FAISS |

---

## 4 Skenario Evaluasi

| Skenario | Sumber Aktif | Fokus |
|---|---|---|
| **A** | PDF/TXT saja | Pemahaman alur bisnis dari dokumen FR |
| **B** | Database saja | Query data struktural/konfigurasi sistem |
| **C** | PDF + Database | Cross-reference dokumen & data aktual |
| **D** | PDF + Database + Chat | Konteks operasional + diskusi tim |

---

## 20 Pertanyaan Evaluasi (Draft v1)

> Format: **[ID]** Pertanyaan → *[Jawaban referensi singkat]*

### Skenario A — PDF Only (Alur Bisnis FR LPDU)

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
| D3 | Berdasarkan log diskusi, apa isu pada NEWCTP-2442 dan bagaimana penyelesaiannya? | Filter status WAITING belum ada di backend; solusi: filter di UI saja (PDS), refresh setiap 1 menit |
| D4 | Bagaimana tim menangani isu RBAC/Keycloak untuk akses firm dalam sistem MOFIDS? | Akun harus di-set langsung di Keycloak (bukan hanya RBAC UI); akses per firm_id |
| D5 | Berdasarkan seluruh sumber, apa status implementasi fitur amend pada modul trade custody? | FR mendefinisikan amend; DB: trade_statuses semua `Success Report` (normal flow); Chat (2025): bug amend create row baru vs update yang lama — belum resolve, tidak tercermin di data trade_statuses yang ada |

---

## Metrik Evaluasi

### 8 Metrik Kuantitatif

| Metrik | Deskripsi | Tools |
|---|---|---|
| **Precision@K** | Proporsi dokumen relevan dari K hasil retrieval | Manual / RAGAS |
| **Recall@K** | Proporsi dokumen relevan yang berhasil di-retrieve | Manual / RAGAS |
| **F1@K** | Harmonic mean Precision & Recall | Dihitung |
| **MRR** | Mean Reciprocal Rank — posisi jawaban benar pertama | Manual |
| **NDCG@K** | Normalized Discounted Cumulative Gain | RAGAS |
| **Faithfulness** | Jawaban LLM sesuai konteks retrieval (tidak hallucinate) | RAGAS |
| **Answer Relevance** | Relevansi jawaban terhadap pertanyaan | RAGAS |
| **Context Precision** | Proporsi konteks yang benar-benar digunakan | RAGAS |

### 3 Metrik Komposit

| Metrik | Formula | Interpretasi |
|---|---|---|
| **KTE** (Knowledge Transfer Effectiveness) | `(Faithfulness + Answer Relevance) / 2` | Seberapa efektif pengetahuan dokumen ditransfer ke jawaban |
| **MSRS** (Multi-Source Retrieval Score) | `(Precision@5 + Recall@5 + NDCG@5) / 3` | Kualitas retrieval lintas sumber |
| **AQI** (Answer Quality Index) | `(KTE × 0.5) + (MSRS × 0.3) + (F1@5 × 0.2)` | Indeks kualitas jawaban keseluruhan |

---

## Rencana Script

### `parse_chat_corpus.py` — Sudah dibuat
- Input: `cuplikan`, `cuplikan2`, `cuplikan-personal-message`
- Output: `data/chat_indices/` (FAISS)
- Thread window: 30 menit
- Masking: nama individu → pseudonim

### Pipeline Evaluasi
```
Notebook: QA_RAG_AgnosticSource.ipynb
  ↓
load_documents() → 3 sumber
  ↓
Untuk setiap skenario (A, B, C, D):
  ↓  
Jalankan 5 pertanyaan
  ↓
Hitung 8 metrik + 3 komposit
  ↓
Tabel hasil evaluasi → Bab 4 DRAFT_JURNAL.md
```

---

## Status & Next Steps

| # | Task | Status |
|---|---|---|
| 1 | Introduction baru (urgency + 3 kontribusi) | ✅ Selesai |
| 2 | Abstract (ID + EN) diperbarui | ✅ Selesai |
| 3 | Rumusan Masalah & Tujuan dihapus | ✅ Selesai |
| 4 | Corpus Layer 1 (FR LPDU PDF) | ✅ Ada di `data/functional-reqiurement` |
| 5 | Corpus Layer 2 (MOFIDS DB SQL) | ✅ Lengkap — 8 tabel, 89 records di `data/mofids_sample.sql` |
| 6 | Corpus Layer 3 — credentials dihapus dari file chat | ✅ Sudah dihapus user |
| 7 | `parse_chat_corpus.py` — script parser Teams format | ✅ Dibuat (perlu dijalankan) |
| 8 | 20 pertanyaan evaluasi (draft v1) | ✅ Draft di atas — perlu diskusi |
| 9 | Mask Bab 2.2 DRAFT_JURNAL.md (BBKP, TINS, equity research) | ⬜ Belum |
| 10 | Jalankan `parse_chat_corpus.py` → build FAISS chat index | ⬜ Belum |
| 11 | Jalankan evaluasi 4 skenario × 5 pertanyaan | ⬜ Belum |
| 12 | Analisis hasil → tulis Bab 3–4 DRAFT_JURNAL.md | ⬜ Belum |
| 13 | Encoding autonumber data untuk lampiran jurnal | ⬜ Belum (akhir) |

---

## Diskusi Terbuka (Perlu Konfirmasi)

1. **20 pertanyaan:** Apakah pertanyaan D1–D5 sudah cukup menguji keunggulan multi-source vs. single-source?
2. **Ground truth:** Apakah jawaban referensi ditetapkan secara manual atau pakai LLM-as-judge?
3. **Evaluator LLM:** Pakai model apa untuk RAGAS scoring? (GPT-4, atau lokal?)
4. **Thread window:** 30 menit sudah tepat, atau lebih pendek (15 menit) untuk chat personal yang singkat?
5. **Skenario D coverage:** Semua 3 file chat dipakai semua, atau hanya `cuplikan` (grup besar)?
