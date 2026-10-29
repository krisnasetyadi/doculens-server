"""OCR for PDF pages that have no embedded text (MS-908).

Scanned and photographed pages are only images, so pdfplumber/pypdf return
nothing for them. PdfOcr renders such a page with pypdfium2 and reads it with
tesseract. Work per file is bounded by config (page count and wall-clock
time), and everything that was not read is counted so the upload can tell
the user instead of silently dropping pages.
"""

import io
import logging
import re
from time import monotonic
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from config import config

logger = logging.getLogger(__name__)


def trim_blank_margins(image, white: int = 245, pad: int = 12):
    """Crop a grayscale page image to its non-white content.

    A phone photo placed on a mostly empty page makes tesseract's layout
    analysis lose the text (a measured book-page photo read 4% of its words
    as a full page, 95% once the white was cropped away)."""
    bbox = image.point(lambda p: 255 if p < white else 0).getbbox()
    if not bbox:
        return image
    left, top, right, bottom = bbox
    width, height = image.size
    return image.crop((max(0, left - pad), max(0, top - pad),
                       min(width, right + pad), min(height, bottom + pad)))


def drop_colored_ink(rgb):
    """Gray page with saturated ink (stamps, colored signatures) painted white.

    Body text is black or gray, so only saturated pixels are removed."""
    hsv = np.asarray(rgb.convert("HSV")).astype("float") / 255
    gray = np.asarray(rgb.convert("L")).copy()
    gray[(hsv[..., 1] > 0.35) & (hsv[..., 2] > 0.25)] = 255
    return Image.fromarray(gray)


def _confident_words(gray) -> int:
    """How many words tesseract reads with high confidence (a quick half-size pass)."""
    import pytesseract

    small = gray.resize((max(1, gray.width // 2), max(1, gray.height // 2)))
    data = pytesseract.image_to_data(small, lang=config.ocr_languages, output_type=pytesseract.Output.DICT,
                                     timeout=config.ocr_page_timeout_seconds)
    return sum(1 for word, conf in zip(data["text"], data["conf"])
               if float(conf) >= 70 and len(word.strip()) >= 2)


def fix_orientation(gray, sure_confidence: float = 10.0):
    """Rotate a sideways or upside-down page upright using tesseract's OSD.

    OSD is trusted outright when it is sure. When it suggests a rotation but
    is unsure (a measured sideways page scored 5.9 while upright pages scored
    4 to 28), both orientations are read and the one with more confident
    words wins, so an upright page is never turned by a weak guess."""
    import pytesseract

    info = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
    if not info["rotate"]:
        return gray
    rotated = gray.rotate(-info["rotate"], expand=True, fillcolor=255)
    if float(info["orientation_conf"]) >= sure_confidence:
        return rotated
    return rotated if _confident_words(rotated) > _confident_words(gray) else gray


def even_out_lighting(gray):
    """Divide out the page background so a shadow or a dark side stops
    hiding the text (a shaded photo read 39% of its words, 100% after)."""
    width, height = gray.size
    small = gray.resize((max(1, width // 4), max(1, height // 4)))
    background = small.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(8))
    background = np.maximum(np.asarray(background.resize((width, height), Image.BILINEAR)).astype("float"), 1)
    return Image.fromarray(np.clip(np.asarray(gray).astype("float") / background * 255, 0, 255).astype("uint8"))


def _longest_runs(dark):
    runs = np.zeros(dark.shape[0], dtype=int)
    for index, row in enumerate(dark):
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
        if len(edges):
            runs[index] = (edges[1::2] - edges[0::2]).max()
    return runs


def remove_ruling_lines(gray, fraction: float = 0.3):
    """Whiten table borders and rules: a dark run across more than
    `fraction` of the page, which text lines never form."""
    pixels = np.asarray(gray).copy()
    dark = pixels < 150
    height, width = pixels.shape
    # Measured against the long side: a page trimmed to one line of text is
    # barely taller than its letters, and their strokes are not rules.
    limit = fraction * max(height, width)
    for row in np.flatnonzero(_longest_runs(dark) > limit):
        band = slice(max(0, row - 1), row + 2)
        pixels[band, :][dark[band, :]] = 255
    for col in np.flatnonzero(_longest_runs(dark.T) > limit):
        band = slice(max(0, col - 1), col + 2)
        pixels[:, band][dark[:, band]] = 255
    return Image.fromarray(pixels)


def straighten(gray, max_angle: float = 8.0, step: float = 0.5):
    """Undo a small tilt: pick the angle whose text rows are sharpest."""
    small = np.asarray(gray.resize((max(1, gray.width // 3), max(1, gray.height // 3)))).astype("float")
    threshold = (np.percentile(small, 2) + np.percentile(small, 50)) / 2
    ink = Image.fromarray(((small < threshold) * 255).astype("uint8"))
    sharpness = {
        round(float(angle), 2): np.var(np.asarray(ink.rotate(angle, resample=Image.BILINEAR)).astype("float").sum(axis=1))
        for angle in np.arange(-max_angle, max_angle + step, step)
    }
    best = max(sharpness, key=sharpness.get)
    if abs(best) < 0.5 or sharpness[best] < 1.05 * sharpness[0.0]:
        return gray
    return gray.rotate(best, expand=True, fillcolor=255, resample=Image.BICUBIC)


def prepare_page_image(rgb):
    """Everything between a rendered PDF page and tesseract.

    Each step is measured to help on its own kind of page (see tests/test_ocr.py
    and scripts/golden_set) and is skipped, not fatal, when it fails."""
    gray = drop_colored_ink(rgb)
    for step in (fix_orientation, trim_blank_margins, even_out_lighting, remove_ruling_lines, straighten, trim_blank_margins):
        try:
            gray = step(gray)
        except Exception as exc:
            if type(exc).__name__ == "TesseractNotFoundError":
                raise
            logger.warning("OCR preprocessing step %s skipped: %s", step.__name__, exc)
    return gray


def new_ocr_stats() -> dict:
    return {"read": 0, "empty": 0, "over_limit": 0, "unavailable": 0}


class PdfOcr:
    """OCR for the pages of one PDF, under one page and time budget."""

    def __init__(self, pdf_path: str, stats: Optional[dict] = None):
        self.pdf_path = pdf_path
        self.stats = stats if stats is not None else new_ocr_stats()
        self._deadline = monotonic() + config.ocr_time_limit_seconds
        self._attempted = 0
        self._pdf = None
        self._broken = False

    def needs_ocr(self, embedded_text: str) -> bool:
        return config.ocr_enabled and len((embedded_text or "").strip()) < config.ocr_min_page_chars

    def read_page(self, page_index: int, embedded_text: str = "") -> str:
        """Text for a 0-based page whose embedded text is missing or tiny.

        Returns the better of the embedded and the OCR'd text, or "" when
        nothing could be read. Never raises: a failed page is counted and the
        rest of the file carries on.
        """
        embedded_text = (embedded_text or "").strip()
        if self._broken:
            self.stats["unavailable"] += 1
            return embedded_text
        if (self._attempted >= config.ocr_max_pages_per_file
                or monotonic() >= self._deadline):
            self.stats["over_limit"] += 1
            return embedded_text
        self._attempted += 1
        try:
            text = self._ocr(page_index)
        except Exception as exc:
            if type(exc).__name__ == "TesseractNotFoundError":
                logger.warning("OCR skipped: tesseract is not installed")
                self._broken = True
                self.stats["unavailable"] += 1
            else:
                logger.warning("OCR failed on page %d of %s: %s", page_index + 1, self.pdf_path, exc)
                self.stats["empty"] += 1
            return embedded_text
        if len(text) > len(embedded_text):
            self.stats["read"] += 1
            return text
        if not embedded_text:
            self.stats["empty"] += 1
        return embedded_text

    def _ocr(self, page_index: int) -> str:
        import pypdfium2 as pdfium

        if self._pdf is None:
            self._pdf = pdfium.PdfDocument(self.pdf_path)
        return read_image_text(
            self._pdf[page_index].render(scale=config.ocr_dpi / 72).to_pil().convert("RGB")
        )

    def close(self) -> None:
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None


class SlideOcr(PdfOcr):
    """OCR for the pictures in one PowerPoint deck (MS-910), under the same
    per-file page and time budget as a PDF, one picture counting as a page."""

    def __init__(self, pptx_path: str, stats: Optional[dict] = None):
        super().__init__(pptx_path, stats)
        self._picture = None

    def read_picture(self, slide_number: int, blob: bytes) -> str:
        """Text in one slide picture, or "" when it holds none. Photos and
        logos without text are normal on slides, so they are not counted as
        pages with no readable text."""
        if not config.ocr_enabled:
            return ""
        self._picture = blob
        empty = self.stats["empty"]
        try:
            return self.read_page(slide_number - 1)
        finally:
            self.stats["empty"] = empty
            self._picture = None

    def _ocr(self, page_index: int) -> str:
        with Image.open(io.BytesIO(self._picture)) as picture:
            return read_image_text(picture.convert("RGB"))


def read_image_text(rgb) -> str:
    """Tesseract's text for one image, cleaned up by prepare_page_image."""
    import pytesseract

    image = prepare_page_image(rgb)
    try:
        raw = pytesseract.image_to_string(
            image, lang=config.ocr_languages, timeout=config.ocr_page_timeout_seconds,
        )
    finally:
        image.close()
    return re.sub(r"\s+", " ", raw).strip()


def describe_ocr_outcome(stats: dict) -> Optional[str]:
    """One sentence for the user when scanned pages were skipped, else None."""
    if not stats:
        return None
    parts = []
    if stats.get("over_limit"):
        parts.append(
            f"{stats['over_limit']} scanned page(s) were not read because the file "
            f"exceeds the OCR limit ({stats.get('read', 0)} scanned page(s) were read)"
        )
    if stats.get("unavailable"):
        parts.append(f"{stats['unavailable']} scanned page(s) were skipped because OCR is unavailable")
    if stats.get("empty"):
        parts.append(f"{stats['empty']} page(s) had no readable text")
    if not parts:
        return None
    return "Some pages were not indexed: " + "; ".join(parts) + "."
