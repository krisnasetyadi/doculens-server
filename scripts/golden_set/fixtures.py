"""Source files for the golden question set (MS-908, the .pptx deck MS-910).

Every file is generated here, so the text on each page is known exactly and
the expected answers in questions.json can be checked against it. Scanned
pages are rendered to images (no embedded text) with the distortions real
scans have: shadow, rotation, tilt, noise, low contrast, a stamp.

build_fixtures(folder) writes the files and returns {file name: {page: text}}.
"""

import csv
import io
import os
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

KEBIJAKAN_CUTI = {
    1: "Kebijakan Cuti PT Sinar Nusantara. Karyawan tetap berhak atas cuti tahunan sebanyak 14 hari kerja "
       "setiap tahun setelah menyelesaikan masa kerja 12 bulan.",
    2: "Karyawati berhak atas cuti melahirkan selama 3 bulan dengan gaji penuh. Karyawan pria berhak atas "
       "cuti ayah selama 5 hari kerja saat istri melahirkan.",
    3: "Pengajuan cuti dilakukan paling lambat 7 hari sebelum tanggal cuti melalui aplikasi HRIS dan "
       "harus disetujui oleh atasan langsung.",
}
SECURITY_POLICY = {
    1: "Information Security Policy. Every account password must be at least 14 characters long and "
       "must be rotated every 180 days. Shared accounts are not allowed.",
    2: "Security incidents must be reported within 4 hours through the internal ticket queue named "
       "SEC-DESK. The security team reviews every report within one business day.",
}
KONTRAK_SEWA = {
    1: "Perjanjian Sewa Gudang antara PT Sinar Nusantara sebagai penyewa dan CV Mitra Logam sebagai "
       "pemilik untuk gudang di kawasan industri Cikarang.",
    2: "Harga sewa gudang adalah Rp 85.000.000 per tahun yang dibayar di muka paling lambat tanggal 10 "
       "Januari setiap tahun.",
    3: "Jangka waktu sewa adalah 3 tahun terhitung mulai tanggal 1 Januari 2026 dan dapat diperpanjang "
       "atas persetujuan kedua pihak.",
    4: "Keterlambatan pembayaran sewa dikenakan denda sebesar 2 persen per bulan dari jumlah yang "
       "terlambat dibayar.",
}
LAPORAN_AUDIT = {
    1: "Ringkasan Audit Internal Semester 1 Tahun 2026. Tim audit mencatat 12 temuan di seluruh unit "
       "operasional perusahaan.",
    2: "Temuan kritis: terdapat selisih stok bahan baku sebanyak 340 unit di gudang Bekasi yang belum "
       "dapat dijelaskan oleh bagian persediaan.",
    3: "Rekomendasi: perusahaan wajib melakukan stock opname bulanan mulai Agustus 2026 dan melaporkan "
       "hasilnya kepada komite audit.",
}
SCANNED_INVOICE = {
    1: "Invoice INV-2026-0417 issued by Bright Tools Ltd for industrial drill equipment. Total amount due "
       "is USD 12,480 including tax.",
    2: "Payment terms are net 45 days from the invoice date. A late payment fee of 1.5 percent per month "
       "applies to overdue balances.",
}
SOP_PENGADAAN = (
    "Standar Operasional Prosedur Pengadaan Barang. Setiap pembelian dengan nilai di atas Rp 50.000.000 "
    "wajib melalui proses tender dengan minimal 3 vendor pembanding. Dokumen penawaran disimpan selama 5 tahun."
)
SOP_PENGADAAN_TABLE = [
    ("Nilai Pembelian", "Penyetuju"),
    ("Di bawah Rp 10.000.000", "Supervisor"),
    ("Rp 10.000.000 sampai Rp 50.000.000", "Manajer Divisi"),
    ("Di atas Rp 50.000.000", "Direktur Keuangan"),
]
ONBOARDING_GUIDE = (
    "Employee Onboarding Guide. New employees receive their laptop from the IT helpdesk on the first day. "
    "Mandatory security awareness training must be completed within 30 days of the start date. "
    "Each new employee is assigned a buddy from the same team for the first 3 months."
)
ANGGARAN_SHEETS = {
    "Operasional": [("Divisi", "Anggaran"), ("Logistik", "Rp 1.250.000.000"), ("Pemasaran", "Rp 640.000.000"),
                    ("Keuangan", "Rp 310.000.000")],
    "Investasi": [("Proyek", "Nilai"), ("Gudang Cikarang", "Rp 3.400.000.000"), ("Sistem ERP", "Rp 900.000.000")],
}
INVENTARIS = [
    ("Kode", "Barang", "Jumlah", "Lokasi"),
    ("INV-001", "Forklift listrik", "4", "Bekasi"),
    ("INV-002", "Pallet kayu", "1200", "Cikarang"),
    ("INV-003", "Rak besi", "85", "Bekasi"),
    ("INV-004", "Timbangan digital", "12", "Cikarang"),
]
CHAT_LINES = [
    "[2026-10-01 09:00:00] Rina: Selamat pagi tim, kickoff proyek ERP dipindah ke hari Kamis 13 November",
    "[2026-10-01 09:05:12] Budi: Siap, ruang rapat lantai 5 sudah saya pesan",
    "[2026-10-02 14:20:00] Rina: Vendor server yang dipilih adalah Lintas Data setelah evaluasi tiga penawaran",
    "[2026-10-02 14:25:41] Andi: Noted. The server delivery is scheduled for the second week of December",
    "[2026-10-03 10:00:00] Rina: Anggaran proyek sudah disetujui direksi sebesar Rp 275 juta",
    "[2026-10-03 10:02:30] Budi: Mantap, saya update di dokumen rencana kerja",
    "[2026-10-04 16:45:00] Andi: Go-live target is 2 February 2027, user training starts mid January",
    "[2026-10-04 16:50:10] Rina: Oke, Andi jadi penanggung jawab migrasi data dari sistem lama",
]

# MS-910: a training deck. Slide 1 is a title slide (carried into slide 2),
# slide 3's fact is only in the speaker notes and slide 4's only in a picture.
DECK_TITLE = "Pelatihan Keselamatan Kerja Gudang 2026"
DECK_SLIDES = {
    2: "Alat pelindung diri wajib dipakai di area gudang: helm, sepatu safety dan rompi reflektif. "
       "Forklift hanya boleh dikemudikan oleh operator bersertifikat SIO.",
    3: "Prosedur evakuasi: hentikan pekerjaan, matikan mesin, lalu ikuti jalur hijau menuju pintu keluar terdekat.",
    4: "Batas beban rak palet, lihat tabel dari tim fasilitas di bawah ini.",
}
DECK_NOTES = {3: "The emergency assembly point is the east parking lot; team leaders take a roll call within 10 minutes."}
DECK_PICTURE = "Kapasitas maksimum rak palet adalah 1.200 kg per level. Rak yang penyok wajib dilaporkan ke tim fasilitas."

PAGE_PX = (1240, 1754)


def _font(size):
    # Pillow's bundled font, so the files come out the same on any machine.
    return ImageFont.load_default(size=size)


def _page_image(text, size=36, width_chars=52):
    image = Image.new("L", PAGE_PX, 255)
    draw = ImageDraw.Draw(image)
    y = 110
    for line in textwrap.wrap(text, width_chars):
        draw.text((90, y), line, font=_font(size), fill=0)
        y += int(size * 1.6)
    return image


def _shadow(image, strength=0.7):
    width, height = image.size
    gradient = np.tile(np.linspace(1 - strength, 1, width), (height, 1))
    return Image.fromarray((np.asarray(image) * gradient).astype("uint8"))


def _noise(image, quality=25):
    rng = np.random.default_rng(908)
    noisy = np.clip(np.asarray(image).astype("float") + rng.normal(0, 16, image.size[::-1]), 0, 255)
    blurred = Image.fromarray(noisy.astype("uint8")).filter(ImageFilter.GaussianBlur(1.0))
    buffer = io.BytesIO()
    blurred.save(buffer, "JPEG", quality=quality)
    return Image.open(buffer).convert("L")


def _stamp(image):
    color = image.convert("RGB")
    draw = ImageDraw.Draw(color)
    draw.ellipse([700, 380, 1060, 640], outline=(40, 40, 170), width=7)
    draw.text((760, 490), "LUNAS", font=_font(44), fill=(40, 40, 170))
    return color


SCAN_EFFECTS = {
    "clean": lambda image: image,
    "shadow": _shadow,
    "rotated": lambda image: image.rotate(90, expand=True, fillcolor=255),
    "tilt_stamp": lambda image: _stamp(image.rotate(3, expand=True, fillcolor=255)),
    "noise": _noise,
    "low_contrast": lambda image: Image.fromarray((np.asarray(image) * 0.35 + 150).astype("uint8")),
}


def _draw_scan(pdf, image, partial=False):
    """One image-only page. partial=True places a small photo on an empty Letter page."""
    image = image.convert("RGB")
    if partial:
        pdf.setPageSize((612, 792))
        pdf.drawImage(ImageReader(image), 50, 300, width=512, height=440)
    else:
        width, height = image.size
        scale = (595 / width) if width <= height else (842 / width)
        pdf.setPageSize((width * scale, height * scale))
        pdf.drawImage(ImageReader(image), 0, 0, width=width * scale, height=height * scale)
    pdf.showPage()


def _draw_digital(pdf, text):
    pdf.setPageSize((595, 842))
    y = 780
    for line in textwrap.wrap(text, 90):
        pdf.drawString(60, y, line)
        y -= 16
    pdf.showPage()


def _write_pdf(path, pages):
    """pages: list of (text, kind) where kind is "digital" or a SCAN_EFFECTS key or "partial"."""
    pdf = canvas.Canvas(path)
    for text, kind in pages:
        if kind == "digital":
            _draw_digital(pdf, text)
        elif kind == "partial":
            photo = _shadow(_page_image(text, size=34, width_chars=56).crop((0, 0, 1240, 900)), 0.45)
            _draw_scan(pdf, photo.rotate(2, expand=True, fillcolor=235), partial=True)
        else:
            _draw_scan(pdf, SCAN_EFFECTS[kind](_page_image(text)))
    pdf.save()


def _write_docx(path, paragraphs, table=None):
    import docx

    document = docx.Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    if table:
        grid = document.add_table(rows=0, cols=len(table[0]))
        for row in table:
            cells = grid.add_row().cells
            for cell, value in zip(cells, row):
                cell.text = value
    document.save(path)


def _write_xlsx(path, sheets):
    import openpyxl

    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for title, rows in sheets.items():
        sheet = workbook.create_sheet(title)
        for row in rows:
            sheet.append(list(row))
    workbook.save(path)


def _write_pptx(path):
    from pptx import Presentation
    from pptx.util import Inches

    deck = Presentation()
    title = deck.slides.add_slide(deck.slide_layouts[0])
    title.shapes.title.text = DECK_TITLE
    for number, text in DECK_SLIDES.items():
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(9), Inches(1.5)).text_frame.text = text
        if number in DECK_NOTES:
            slide.notes_slide.notes_text_frame.text = DECK_NOTES[number]
        if number == 4:
            slide.shapes.add_picture(_deck_picture(), Inches(0.5), Inches(2.5), width=Inches(9))
    deck.save(path)


def _deck_picture():
    """A screenshot-like picture: light background, a border, two lines of text."""
    picture = Image.new("RGB", (1400, 360), (246, 246, 242))
    draw = ImageDraw.Draw(picture)
    draw.rectangle([4, 4, 1395, 355], outline=(120, 120, 120), width=3)
    for row, line in enumerate(textwrap.wrap(DECK_PICTURE, 56)):
        draw.text((50, 70 + row * 90), line, font=_font(42), fill=(25, 25, 25))
    buffer = io.BytesIO()
    picture.save(buffer, "PNG")
    buffer.seek(0)
    return buffer


def build_fixtures(folder):
    """Write every golden source into `folder`. Returns {file: {page: text}}."""
    os.makedirs(folder, exist_ok=True)
    truth = {}

    def pdf(name, texts, kinds):
        _write_pdf(os.path.join(folder, name), [(texts[page], kind) for page, kind in zip(sorted(texts), kinds)])
        truth[name] = dict(texts)

    pdf("kebijakan-cuti.pdf", KEBIJAKAN_CUTI, ["digital"] * 3)
    pdf("security-policy.pdf", SECURITY_POLICY, ["digital"] * 2)
    pdf("kontrak-sewa-scan.pdf", KONTRAK_SEWA, ["clean", "shadow", "rotated", "tilt_stamp"])
    pdf("laporan-audit-campuran.pdf", LAPORAN_AUDIT, ["digital", "partial", "digital"])
    pdf("scanned-invoice.pdf", SCANNED_INVOICE, ["noise", "low_contrast"])

    _write_docx(os.path.join(folder, "sop-pengadaan.docx"), [SOP_PENGADAAN], SOP_PENGADAAN_TABLE)
    truth["sop-pengadaan.docx"] = {1: SOP_PENGADAAN + " " + " ".join(" | ".join(row) for row in SOP_PENGADAAN_TABLE)}
    _write_docx(os.path.join(folder, "onboarding-guide.docx"), [ONBOARDING_GUIDE])
    truth["onboarding-guide.docx"] = {1: ONBOARDING_GUIDE}

    _write_xlsx(os.path.join(folder, "anggaran-2026.xlsx"), ANGGARAN_SHEETS)
    truth["anggaran-2026.xlsx"] = {
        index: " ".join(" ".join(row) for row in rows) for index, rows in enumerate(ANGGARAN_SHEETS.values(), 1)
    }

    with open(os.path.join(folder, "inventaris.csv"), "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(INVENTARIS)
    header, *rows = INVENTARIS
    truth["inventaris.csv"] = {1: " ".join(", ".join(f"{h}: {v}" for h, v in zip(header, row)) for row in rows)}

    _write_pptx(os.path.join(folder, "pelatihan-k3-gudang.pptx"))
    truth["pelatihan-k3-gudang.pptx"] = {
        2: f"{DECK_TITLE} {DECK_SLIDES[2]}",
        3: f"{DECK_SLIDES[3]} Speaker notes: {DECK_NOTES[3]}",
        4: f"{DECK_SLIDES[4]} {DECK_PICTURE}",
    }

    with open(os.path.join(folder, "chat-tim-proyek.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(CHAT_LINES) + "\n")
    truth["chat-tim-proyek.txt"] = {None: " ".join(line.split("] ", 1)[1] for line in CHAT_LINES)}
    return truth


# Which pages are image-only, i.e. only readable with OCR.
SCANNED_PAGES = {
    "kontrak-sewa-scan.pdf": {1, 2, 3, 4},
    "laporan-audit-campuran.pdf": {2},
    "scanned-invoice.pdf": {1, 2},
}

# Slides whose answer is only inside a picture, i.e. only readable with OCR.
PICTURE_SLIDES = {"pelatihan-k3-gudang.pptx": {4}}
