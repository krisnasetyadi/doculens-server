# DocuLens — Document Intelligence Platform

Dokumen ini adalah **penjelasan level-produk** tentang keseluruhan project DocuLens — apa yang dikerjakan, kenapa ada, dan bagaimana potongan-potongannya (frontend, backend, deployment) saling terkait. Untuk instruksi setup/instalasi teknis per komponen, lihat `README.md` di masing-masing folder — dokumen ini tidak menggantikannya, hanya memberi gambaran besar sebelum masuk ke detail.

## Apa itu DocuLens?

DocuLens (*Document Lens*) adalah platform **document intelligence**: sistem tanya-jawab berbasis AI yang mencari jawaban lintas beberapa jenis sumber pengetahuan sekaligus dalam satu percakapan —

- **Dokumen PDF** yang diupload (manual/laporan/kontrak/regulasi, dll)
- **Database SQL** milik user sendiri (PostgreSQL) — ditanya dengan bahasa natural, bukan SQL
- **Log chat** yang diekspor (WhatsApp/Telegram/Teams) sebagai riwayat percakapan yang bisa dicari
- **Public link** (mis. Google Drive) sebagai sumber tambahan tanpa perlu upload manual

Ini bukan cuma "chat dengan PDF" — bagian pembeda utamanya adalah **hybrid search**: satu pertanyaan bisa otomatis dirutekan ke beberapa sumber sekaligus (PDF + DB + chat log), hasilnya digabung, lalu dijawab oleh LLM dengan tautan sumber yang bisa dibuka langsung ke halaman PDF yang relevan.

Selain tanya-jawab, ada satu fitur "Skill" turunan yang lebih spesifik untuk kebutuhan enterprise: **Compliance Gap Check** — membandingkan satu dokumen (misalnya dokumen internal perusahaan) terhadap dokumen acuan/standar (misalnya ISO 27001, SOP, atau regulasi apa pun yang diupload sebagai PDF), lalu menghasilkan laporan gap per item (met / partial / not met) yang bisa diunduh sebagai Markdown atau PDF.

## Kenapa project ini ada

Kebutuhan yang mendasarinya: organisasi biasanya punya pengetahuan yang tersebar di tempat berbeda-beda — dokumen resmi di PDF, data operasional di database, dan riwayat diskusi di chat — dan mencari jawaban berarti harus buka satu-satu secara manual. DocuLens menyatukan pencarian itu ke satu antarmuka chat, dengan role-based access (Member vs Admin) supaya sumber sensitif (koneksi database, log chat internal) hanya bisa dikelola oleh Admin, sementara Member tetap bisa bertanya menggunakan sumber yang sudah diaktifkan.

## Potongan-potongan sistem

Project ini terdiri dari beberapa repo/folder terpisah yang bersama-sama membentuk satu produk:

| Folder | Peran | Status |
|---|---|---|
| `chat-ui/` | Frontend — Next.js 16 + React 19 + shadcn/ui. Tempat user login, kelola sumber (Sources), chat (Ask), lihat riwayat (History), dan jalankan Gap Check. | Aktif dikembangkan |
| `pdf-reader/` | Backend utama — FastAPI + LangChain + FAISS. Semua logika inti: upload & indexing PDF, hybrid search, auth/RBAC, koneksi database eksternal, public link, session history, compliance gap analysis. | Aktif dikembangkan (sumber kebenaran backend) |
| `hf-doculens-api/` | Mirror/deploy target backend ke Hugging Face Spaces (Docker). Router-nya nyaris identik dengan `pdf-reader/`, tapi saat ini **belum termasuk** router `compliance` (Gap Check) — artinya versi yang live di HF Space sedikit tertinggal dari versi dev. | Deployment mirror |
| `doculens-api/`, `rag/` | Eksperimen/prototipe awal (skrip RAG standalone, skeleton HF Space lama). Bukan bagian dari alur produk aktif saat ini. | Legacy / eksperimen |

Alur singkatnya: **chat-ui** (browser) → **pdf-reader** (API, `:8000`) → FAISS index / PostgreSQL / LLM provider → jawaban + sumber dikembalikan ke chat-ui. Versi yang di-deploy publik jalan di atas `hf-doculens-api`.

## Fitur inti (yang benar-benar sudah ada di kode, bukan rencana)

**Sumber pengetahuan (Sources)**
- Upload PDF → di-chunk → di-embed (`paraphrase-multilingual-MiniLM-L12-v2`) → diindeks ke FAISS per collection
- Hubungkan database PostgreSQL milik sendiri → browse tabel & kolom, tanya dengan bahasa natural
- Import chat log (`.txt`) → dicari sama seperti PDF (FAISS)
- Tambahkan public link (Google Drive, dll) sebagai sumber tanpa upload manual
- Setiap sumber punya toggle Active/Inactive independen dari keberadaannya (dihapus vs dinonaktifkan sementara)

**Hybrid Search & Chat**
- Satu pertanyaan dirutekan otomatis ke PDF / Database / Chat log yang relevan (query expansion + keyword routing)
- Jawaban disertai kutipan sumber — klik untuk membuka PDF langsung ke halaman & teks yang relevan
- Riwayat percakapan tersimpan per session, bisa dilanjutkan/dicari/dihapus dari halaman History
- Dukungan multi-provider LLM: HuggingFace (lokal), Ollama (lokal), Gemini (cloud) — dipilih per percakapan

**Compliance Gap Check**
- Pilih collection "reference" (standar/framework) dan collection "target" (dokumen perusahaan)
- Sistem mengekstrak item-item dari reference, mengecek pemenuhannya di target, dan memberi status met/partial/not met + evidence + rekomendasi
- Hasil bisa diekspor ke Markdown atau PDF

**Akses & Keamanan**
- Auth berbasis JWT dengan role Admin/Member (RBAC)
- Sumber Database & Chat log adalah admin-only di UI (dan ditegakkan juga di backend, bukan cuma disembunyikan di UI)
- Password akun bisa diganti sendiri; Admin bisa reset password user lain

## Tech stack

**Frontend (`chat-ui/`)**
Next.js 16 (App Router) · React 19 · TypeScript · Tailwind CSS 4 · shadcn/ui (Radix primitives) · Zustand (state) · dayjs

**Backend (`pdf-reader/`, `hf-doculens-api/`)**
FastAPI · LangChain · FAISS (vector store) · PostgreSQL (via Neon atau instance sendiri) · HuggingFace Transformers / Ollama / Gemini sebagai LLM provider · PyPDF2 untuk ekstraksi teks PDF

## Peta ke dokumen lain

- Setup & instalasi backend → `pdf-reader/README.md`
- Setup & instalasi frontend → `chat-ui/README.md`
- Detail model biaya (token/compute) → `pdf-reader/PRICING_COST_MODEL.md`
- Catatan desain ulang panel Sources → `pdf-reader/SOURCE_PANEL_REDESIGN_DISCUSSION.md`
- Rencana/kerja-dalam-progres backend → `pdf-reader/new-plan.md`
