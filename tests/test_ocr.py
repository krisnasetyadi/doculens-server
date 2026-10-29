"""MS-908: scanned PDF pages are read with OCR instead of being dropped.

The OCR engine itself is faked, so these run without tesseract installed;
they cover which pages go to OCR, the caps, failure handling and progress."""

import io
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "ms908-local-test-secret-not-for-production")

from reportlab.pdfgen import canvas

from config import config
from services.ingest import ocr as ocr_module
from services.ingest.documents import _extract_pdf_with_progress, extract_text_from_pdf
from services.ingest.ocr import (
    PdfOcr, describe_ocr_outcome, drop_colored_ink, even_out_lighting, fix_orientation,
    new_ocr_stats, prepare_page_image, remove_ruling_lines, straighten, trim_blank_margins,
)
from services.ingest.progress import _UploadProgressStore


def make_pdf(path, pages):
    """pages: a string for a digital page, None for an image-only (blank) page."""
    c = canvas.Canvas(path)
    for text in pages:
        if text:
            c.drawString(72, 720, text)
        c.showPage()
    c.save()


class OcrExtractionTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "doc.pdf")
        for name, value in {
            "ocr_enabled": True, "ocr_min_page_chars": 20,
            "ocr_max_pages_per_file": 40, "ocr_time_limit_seconds": 120,
        }.items():
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def extract(self, ocr_text="Teks hasil OCR dari halaman pindaian", **kwargs):
        stats = new_ocr_stats()
        with patch.object(PdfOcr, "_ocr", return_value=ocr_text) as fake:
            pages = extract_text_from_pdf(self.path, ocr_stats=stats, **kwargs)
        return pages, stats, fake

    def test_digital_pdf_is_not_ocrd(self):
        make_pdf(self.path, ["Halaman digital dengan teks yang cukup panjang."] * 2)
        pages, stats, fake = self.extract()
        fake.assert_not_called()
        self.assertEqual([p["page"] for p in pages], [1, 2])
        self.assertEqual(stats, new_ocr_stats())

    def test_scanned_pdf_is_read_with_its_page_numbers(self):
        make_pdf(self.path, [None, None])
        pages, stats, _ = self.extract()
        self.assertEqual([p["page"] for p in pages], [1, 2])
        self.assertEqual(pages[0]["text"], "Teks hasil OCR dari halaman pindaian")
        self.assertEqual(stats["read"], 2)
        self.assertIsNone(describe_ocr_outcome(stats))

    def test_mixed_pdf_keeps_every_page(self):
        make_pdf(self.path, ["Halaman digital dengan teks yang cukup panjang.", None, "Digital lagi, teks cukup panjang di sini."])
        pages, stats, fake = self.extract()
        self.assertEqual([p["page"] for p in pages], [1, 2, 3])
        self.assertEqual(fake.call_count, 1)
        self.assertEqual(pages[0]["text"], "Halaman digital dengan teks yang cukup panjang.")
        self.assertEqual(pages[1]["text"], "Teks hasil OCR dari halaman pindaian")

    def test_page_with_no_ocr_text_is_skipped_and_reported(self):
        make_pdf(self.path, ["Halaman digital dengan teks yang cukup panjang.", None])
        pages, stats, _ = self.extract(ocr_text="")
        self.assertEqual([p["page"] for p in pages], [1])
        self.assertEqual(stats["empty"], 1)
        self.assertIn("1 page(s) had no readable text", describe_ocr_outcome(stats))

    def test_ocr_failure_skips_the_page_not_the_file(self):
        make_pdf(self.path, [None, "Halaman digital dengan teks yang cukup panjang."])
        stats = new_ocr_stats()
        with patch.object(PdfOcr, "_ocr", side_effect=RuntimeError("bad image")):
            pages = extract_text_from_pdf(self.path, ocr_stats=stats)
        self.assertEqual([p["page"] for p in pages], [2])
        self.assertEqual(stats["empty"], 1)

    def test_missing_tesseract_is_reported_once_and_does_not_retry(self):
        class TesseractNotFoundError(Exception):
            pass

        make_pdf(self.path, [None, None, "Halaman digital dengan teks yang cukup panjang."])
        stats = new_ocr_stats()
        with patch.object(PdfOcr, "_ocr", side_effect=TesseractNotFoundError()) as fake:
            pages = extract_text_from_pdf(self.path, ocr_stats=stats)
        self.assertEqual(fake.call_count, 1)
        self.assertEqual(stats["unavailable"], 2)
        self.assertEqual([p["page"] for p in pages], [3])
        self.assertIn("OCR is unavailable", describe_ocr_outcome(stats))

    def test_page_cap_stops_ocr_and_says_how_many_were_read(self):
        make_pdf(self.path, [None] * 5)
        with patch.object(config, "ocr_max_pages_per_file", 2):
            pages, stats, fake = self.extract()
        self.assertEqual(fake.call_count, 2)
        self.assertEqual([p["page"] for p in pages], [1, 2])
        self.assertEqual((stats["read"], stats["over_limit"]), (2, 3))
        self.assertIn("3 scanned page(s) were not read", describe_ocr_outcome(stats))
        self.assertIn("2 scanned page(s) were read", describe_ocr_outcome(stats))

    def test_time_cap_stops_ocr(self):
        make_pdf(self.path, [None] * 3)
        ticks = iter([0.0, 0.0, 999.0, 999.0, 999.0])
        with patch.object(ocr_module, "monotonic", side_effect=lambda: next(ticks)):
            pages, stats, fake = self.extract()
        self.assertEqual(fake.call_count, 1)
        self.assertEqual(stats["over_limit"], 2)

    def test_ocr_can_be_switched_off(self):
        make_pdf(self.path, [None, "Halaman digital dengan teks yang cukup panjang."])
        with patch.object(config, "ocr_enabled", False):
            pages, stats, fake = self.extract()
        fake.assert_not_called()
        self.assertEqual([p["page"] for p in pages], [2])

    def test_shorter_ocr_result_keeps_the_embedded_text(self):
        make_pdf(self.path, ["No. 7"])
        pages, stats, _ = self.extract(ocr_text="N")
        self.assertEqual(pages[0]["text"], "No. 7")


class TrimBlankMarginsTests(unittest.TestCase):
    def test_crops_to_the_content_plus_padding(self):
        from PIL import Image
        page = Image.new("L", (200, 300), 255)
        page.paste(0, (50, 40, 150, 100))
        self.assertEqual(trim_blank_margins(page, pad=5).size, (110, 70))

    def test_blank_page_is_left_alone(self):
        from PIL import Image
        page = Image.new("L", (200, 300), 255)
        self.assertEqual(trim_blank_margins(page).size, (200, 300))

    def test_padding_never_goes_past_the_page(self):
        from PIL import Image
        page = Image.new("L", (100, 100), 255)
        page.paste(0, (0, 0, 30, 30))
        self.assertEqual(trim_blank_margins(page, pad=12).size, (42, 42))


def _text_like(width=400, height=300):
    """White page with ten short dark 'words' per line, like body text."""
    from PIL import Image
    page = Image.new("L", (width, height), 255)
    for y in range(40, height - 40, 24):
        for x in range(30, width - 60, 40):
            page.paste(0, (x, y, x + 28, y + 8))
    return page


class PreparePageTests(unittest.TestCase):
    def test_colored_ink_is_removed_and_black_text_kept(self):
        from PIL import Image
        import numpy as np
        rgb = Image.new("RGB", (60, 20), (255, 255, 255))
        rgb.paste((200, 30, 30), (0, 0, 20, 20))
        rgb.paste((0, 0, 0), (40, 0, 60, 20))
        gray = np.asarray(drop_colored_ink(rgb))
        self.assertEqual(int(gray[10, 10]), 255)
        self.assertEqual(int(gray[10, 50]), 0)

    def test_ruling_lines_go_but_text_stays(self):
        import numpy as np
        page = _text_like()
        page.paste(0, (0, 150, 400, 152))
        page.paste(0, (200, 0, 202, 300))
        cleaned = np.asarray(remove_ruling_lines(page))
        self.assertGreater(int(cleaned[151, 100]), 200)
        self.assertGreater(int(cleaned[100, 201]), 200)
        self.assertLess(int(cleaned[44, 40]), 50)

    def test_letter_strokes_stay_on_a_single_line_crop(self):
        import numpy as np
        from PIL import Image
        line = Image.new("L", (780, 75), 255)
        line.paste(0, (40, 10, 46, 65))  # the stem of a "k", most of the crop's height
        cleaned = np.asarray(remove_ruling_lines(line))
        self.assertLess(int(cleaned[40, 43]), 50)

    def test_small_tilt_is_straightened(self):
        import numpy as np
        page = _text_like()
        tilted = page.rotate(4, expand=True, fillcolor=255)
        fixed = straighten(tilted)
        rows = lambda im: np.var((np.asarray(im) < 128).sum(axis=1).astype(float))
        self.assertGreater(rows(fixed), rows(tilted))

    def test_straight_page_is_not_rotated(self):
        page = _text_like()
        self.assertEqual(straighten(page).size, page.size)

    def test_uneven_light_is_evened_out(self):
        import numpy as np
        from PIL import Image
        page = _text_like(); width, height = page.size
        shade = np.tile(np.linspace(0.4, 1, width), (height, 1))
        dark = Image.fromarray((np.asarray(page) * shade).astype("uint8"))
        fixed = np.asarray(even_out_lighting(dark))
        background = lambda a: (a[:, :40].mean(), a[:, -40:].mean())
        left, right = background(fixed)
        self.assertLess(abs(left - right), 0.5 * abs(background(np.asarray(dark))[0] - background(np.asarray(dark))[1]))

    def test_orientation_follows_osd_when_it_is_sure(self):
        from PIL import Image
        page = Image.new("L", (200, 100), 255)
        with patch("pytesseract.image_to_osd", return_value={"rotate": 90, "orientation_conf": "20.0"}):
            self.assertEqual(fix_orientation(page).size, (100, 200))
        with patch("pytesseract.image_to_osd", return_value={"rotate": 0, "orientation_conf": "20.0"}):
            self.assertEqual(fix_orientation(page).size, (200, 100))

    def test_unsure_osd_rotates_only_when_the_rotated_page_reads_better(self):
        from PIL import Image
        page = Image.new("L", (200, 100), 255)
        unsure = {"rotate": 90, "orientation_conf": "5.9"}
        with patch("pytesseract.image_to_osd", return_value=unsure), \
                patch.object(ocr_module, "_confident_words", side_effect=lambda image: 40 if image.size == (100, 200) else 2):
            self.assertEqual(fix_orientation(page).size, (100, 200))
        with patch("pytesseract.image_to_osd", return_value=unsure), \
                patch.object(ocr_module, "_confident_words", side_effect=lambda image: 2 if image.size == (100, 200) else 40):
            self.assertEqual(fix_orientation(page).size, (200, 100))

    def test_a_failing_step_is_skipped_not_fatal(self):
        from PIL import Image
        rgb = Image.new("RGB", (200, 120), (255, 255, 255))
        with patch("pytesseract.image_to_osd", side_effect=RuntimeError("osd data missing")):
            self.assertIsNotNone(prepare_page_image(rgb))

    def test_missing_tesseract_still_surfaces(self):
        from PIL import Image

        class TesseractNotFoundError(Exception):
            pass

        with patch("pytesseract.image_to_osd", side_effect=TesseractNotFoundError()):
            with self.assertRaises(TesseractNotFoundError):
                prepare_page_image(Image.new("RGB", (200, 120), (255, 255, 255)))


class OcrProgressTests(unittest.TestCase):
    def test_ocr_pages_report_the_ocr_stage_with_page_counts(self):
        import tempfile
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "doc.pdf")
            make_pdf(path, ["Halaman digital dengan teks yang cukup panjang.", None, None])
            with patch.object(PdfOcr, "_ocr", return_value="Teks hasil OCR dari halaman pindaian"):
                _extract_pdf_with_progress(
                    extract_text_from_pdf, path, 0, 1,
                    lambda *args: events.append(args), new_ocr_stats(),
                )
        stages = [(e[0], e[2] if len(e) > 2 else None) for e in events]
        self.assertEqual(stages[0], ("reading", None))
        self.assertIn(("ocr", {"page": 2, "total": 3}), stages)
        self.assertIn(("ocr", {"page": 3, "total": 3}), stages)
        progress = [e[1] for e in events]
        self.assertEqual(progress, sorted(progress))

    def test_snapshot_carries_pages_only_while_ocr_is_running(self):
        store = _UploadProgressStore()
        store.start("u1", owner_id="o")
        store.update("u1", {"type": "progress", "stage": "ocr", "progress": 20, "pages": {"page": 3, "total": 12}})
        self.assertEqual(store.get("u1", "o")["pages"], {"page": 3, "total": 12})
        store.update("u1", {"type": "progress", "stage": "preparing", "progress": 30})
        self.assertNotIn("pages", store.get("u1", "o"))


if __name__ == "__main__":
    unittest.main()
