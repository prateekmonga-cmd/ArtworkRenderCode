"""
Artwork Compliance Extraction FastAPI Application v2.0
======================================================
Enhanced: zone detection, rotation, ligature repair, annotation filtering.
"""

import fitz  # PyMuPDF
import re
import math
import io
import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
from typing import Any, Optional
from playwright.async_api import async_playwright

logger = logging.getLogger("artwork-extractor")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# ─── Optional OCR support (graceful degrade if tesseract missing) ───
try:
    import pytesseract
    from PIL import Image
    try:
        pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
    except Exception:
        TESSERACT_AVAILABLE = False
        logger.warning("pytesseract installed but tesseract binary not found — OCR fallback disabled")
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract/Pillow not installed — OCR fallback disabled")

_ocr_lang = None

def get_ocr_lang() -> str:
    """Prefer Spanish+English if the spa language pack is installed."""
    global _ocr_lang
    if _ocr_lang is None:
        _ocr_lang = "eng"
        try:
            if "spa" in pytesseract.get_languages(config=""):
                _ocr_lang = "spa+eng"
        except Exception:
            pass
    return _ocr_lang

# Global browser instance (reused across requests)
_playwright = None
_browser = None


app = FastAPI(
    title="Artwork Compliance Extractor",
    description="Enhanced extraction with zone detection, rotation, and text repair",
    version="2.0.0"
)

PT_TO_MM = 0.3528

PAGE_CONFIG = {
    0: {"key": "page1", "name": "outer carton"},
    1: {"key": "page2", "name": "sticker label"},
    2: {"key": "page3", "name": "insert front"},
    3: {"key": "page4", "name": "insert back"},
}
SKIP_PAGES = []  # Removed hardcoded [4] to allow processing all pages

# ─── OCR performance bounds ──────────────────────────────────
# OCR is a targeted fallback, not a bulk pass. A page riddled with corrupt
# spans is handled by the downstream vision cross-check instead — OCR-ing
# hundreds of spans serially is what caused the 15-min /extract hang.
OCR_DPI = 200                     # 200 DPI: ~20px glyphs on 6-7pt pharma print (150 was borderline)
OCR_SPAN_CAP = 15                 # >this many guessed spans on a page → skip OCR, flag page
OCR_MAX_WORKERS = 4               # bounded parallel Tesseract calls (Render CPU is small)
EXTRACTION_TIME_BUDGET_S = 120    # wall-clock guard: degrade gracefully, never 502

# ─── Annotation detection ────────────────────────────────────
ANNOTATION_COLORS = {"#0000ff", "#0000cd", "#0054a6", "#0091d2", "#1a73e8", "#196ea6"}
DIMENSION_PATTERN = re.compile(r'^\s*[\d.]+\s*mm\s*$', re.IGNORECASE)
ARROW_CHARS = {'◄', '►', '▲', '▼', '←', '→', '↑', '↓', '◀', '▶'}

# ─── Ligature repair ─────────────────────────────────────────
KNOWN_CORRECTIONS = {
    'con\ufffdene': 'contiene', 'Con\ufffdene': 'Contiene',
    'pharmaceu\ufffdcals': 'pharmaceuticals', 'Pharmaceu\ufffdcals': 'Pharmaceuticals',
    'úl\ufffdmo': 'último', 'e\ufffdqueta': 'etiqueta',
    'me\ufffdlo': 'metilo', 'ac\ufffdvo': 'activo', 'ac\ufffdva': 'activa',
    'an\ufffdhelmín\ufffdco': 'antihelmíntico', 'Vida ú\ufffdl': 'Vida útil',
    'sus\ufffdtución': 'sustitución', 'garan\ufffdza': 'garantiza',
    'iden\ufffdficar': 'identificar', 'inves\ufffdgar': 'investigar',
    'efe\ufffdvos': 'efectivos', 'Ges\ufffdón': 'Gestión',
    'can\ufffddad': 'cantidad', 'repe\ufffdr': 'repetir',
    'par\ufffdcular': 'particular', 'compa\ufffdble': 'compatible',
    'mul\ufffdplicar': 'multiplicar', 'alterna\ufffdva': 'alternativa',
    'Alterna\ufffdva': 'Alternativa', 'obje\ufffdvo': 'objetivo',
    'sen\ufffddo': 'sentido', 'intes\ufffdnal': 'intestinal',
    'Intes\ufffdnal': 'Intestinal', 'adver\ufffdda': 'advertida',
    'sor\ufffdtol': 'sorbitol', 'Sor\ufffdtol': 'Sorbitol',
}


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def pt_to_mm(value: float) -> float:
    return round(value * PT_TO_MM, 2)

def bbox_to_mm(bbox) -> list:
    return [pt_to_mm(v) for v in bbox]

def rgb_to_hex(color: Any) -> str:
    if color is None:
        return "#000000"
    if isinstance(color, (int, float)):
        val = int(color * 255)
        return f"#{val:02x}{val:02x}{val:02x}"
    if len(color) == 1:
        val = int(color[0] * 255)
        return f"#{val:02x}{val:02x}{val:02x}"
    if len(color) == 3:
        r, g, b = [int(c * 255) for c in color]
        return f"#{r:02x}{g:02x}{b:02x}"
    if len(color) == 4:
        c_val, m, y, k = color
        r = int(255 * (1 - c_val) * (1 - k))
        g = int(255 * (1 - m) * (1 - k))
        b = int(255 * (1 - y) * (1 - k))
        return f"#{r:02x}{g:02x}{b:02x}"
    return "#000000"

def int_color_to_hex(color: int) -> str:
    r = (color >> 16) & 0xFF
    g = (color >> 8) & 0xFF
    b = color & 0xFF
    return f"#{r:02x}{g:02x}{b:02x}"

def is_bold(font_name: str, flags: int) -> bool:
    if flags & (1 << 18):
        return True
    bold_patterns = ["bold", "black", "heavy", "demi", "semibold", "extrabold"]
    return any(p in font_name.lower() for p in bold_patterns)

def is_italic(font_name: str, flags: int) -> bool:
    if flags & (1 << 6):
        return True
    return any(p in font_name.lower() for p in ["italic", "oblique", "slant"])

def get_base_font_name(font_name: str) -> str:
    name = font_name
    if "+" in name:
        name = name.split("+")[-1]
    for suffix in ["-Bold", "-Italic", "-BoldItalic", "-Regular",
                   "Bold", "Italic", "Regular", "Light", "Medium",
                   "-MT", "MT", "-PS", "PS"]:
        name = name.replace(suffix, "")
    return name.strip()

def snap_size(size: float, precision: float = 0.5) -> float:
    return round(size / precision) * precision

def snap_coord(coord: float) -> float:
    return round(coord)


# ═══════════════════════════════════════════════════════════════
# TEXT REPAIR
# ═══════════════════════════════════════════════════════════════

# Characters that appear when UTF-8 bytes are decoded as CP936/GBK: one CJK glyph
# in place of each accented letter. Spanish and English artwork can never
# legitimately contain these, so a run of them is always extractor damage.
MOJIBAKE_RUN = re.compile(
    r"[　-〿一-鿿豈-﫿＀-￯]+"
)


def is_plausible_artwork_char(ch: str) -> bool:
    """True for characters that can legitimately appear in this artwork.

    Latin letters and accents, punctuation, symbols such as the degree sign and
    superscripts. Anything else (Cyrillic, Greek, CJK) means a mojibake candidate
    decoded into noise rather than the Latin text that was lost.
    """
    cp = ord(ch)
    return (
        cp < 0x0250                    # ASCII, Latin-1, Latin Extended-A/B
        or 0x2000 <= cp <= 0x20CF      # punctuation, superscripts, currency
        or 0x2122 == cp                # trademark
    )


def repair_mojibake(text: str) -> tuple:
    """Reverse UTF-8-decoded-as-GBK corruption.

    "Composici<CJK>n" becomes "Composicion" with the accent restored, because the
    CJK glyph carries the original UTF-8 bytes of the accented letter. Each run of
    CJK characters is re-encoded to those bytes and decoded as UTF-8.

    Runs that do not round-trip cleanly are left exactly as they were, so genuine
    CJK text and unrecoverable damage (where the extractor already collapsed bytes
    to U+FFFD) are never mangled further.

    Returns (repaired, repairs).
    """
    if not MOJIBAKE_RUN.search(text):
        return text, []

    repairs = []

    def fix(match):
        run = match.group(0)
        try:
            candidate = run.encode("gbk").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return run
        if not candidate or "�" in candidate:
            return run
        # A run can round-trip into something that is merely *valid* UTF-8 rather
        # than the Latin text we lost (a lone CJK char decodes to Cyrillic, for
        # instance). Only accept a candidate that is plausible artwork text.
        if not all(is_plausible_artwork_char(ch) for ch in candidate):
            return run
        repairs.append({"from": run, "to": candidate, "type": "mojibake_gbk",
                        "confidence": "high"})
        return candidate

    return MOJIBAKE_RUN.sub(fix, text), repairs


def repair_text(text: str) -> tuple:
    """Repair mojibake and ligature corruption.

    Returns (repaired, was_repaired, repairs, confidence) where confidence is:
      - None      -> text was untouched
      - "high"    -> only known/validated corrections applied (KNOWN_CORRECTIONS,
                     standard fi/fl ligatures, soft-hyphen removal)
      - "guessed" -> the blanket U+FFFD->"ti" or soft-hyphen->"ti" fallback fired;
                     downstream should treat this text as unverified
    """
    has_ligature_damage = ('\ufffd' in text or '\xad' in text
                           or '\ufb01' in text or '\ufb02' in text)
    has_mojibake = bool(MOJIBAKE_RUN.search(text))
    if not has_ligature_damage and not has_mojibake:
        return text, False, [], None

    repairs = []
    repaired = text
    guessed = False

    # 0. Mojibake first \u2014 the later steps must see real accented letters, and
    #    KNOWN_CORRECTIONS keys are written in their accented form.
    if has_mojibake:
        repaired, mojibake_repairs = repair_mojibake(repaired)
        repairs.extend(mojibake_repairs)

    # 1. Known word corrections (case-sensitive)
    for broken, fixed in KNOWN_CORRECTIONS.items():
        if broken in repaired:
            repaired = repaired.replace(broken, fixed)
            repairs.append({"from": broken, "to": fixed, "type": "known_word",
                            "confidence": "high"})

    # 2. Remaining U+FFFD → assume "ti" ligature (dominant in pharma fonts)
    if '\ufffd' in repaired:
        repaired = repaired.replace('\ufffd', 'ti')
        repairs.append({"from": "U+FFFD", "to": "ti", "type": "ligature_ti",
                        "confidence": "guessed"})
        guessed = True

    # 3. Soft hyphens — only replace with 'ti' when between letters (ligature artifact)
    if '\xad' in repaired:
        # Only replace \xad when it sits between two word characters (ligature gap)
        new_text = re.sub(r'(?<=\w)\xad(?=\w)', 'ti', repaired)
        if new_text != repaired:
            repairs.append({"from": "U+00AD", "to": "ti", "type": "soft_hyphen_ligature",
                            "confidence": "guessed"})
            repaired = new_text
            guessed = True
        else:
            # Standalone soft hyphen — just remove it
            repaired = repaired.replace('\xad', '')
            repairs.append({"from": "U+00AD", "to": "", "type": "soft_hyphen_removed",
                            "confidence": "high"})

    # 4. Standard ligature chars
    for lig, rep in {'\ufb01': 'fi', '\ufb02': 'fl'}.items():
        if lig in repaired:
            repaired = repaired.replace(lig, rep)
            repairs.append({"from": f"U+{ord(lig):04X}", "to": rep, "type": "ligature",
                            "confidence": "high"})

    confidence = "guessed" if guessed else ("high" if repairs else None)
    return repaired, len(repairs) > 0, repairs, confidence


# ═══════════════════════════════════════════════════════════════
# OCR CROP FALLBACK (for guessed corruption repairs)
# ═══════════════════════════════════════════════════════════════

def ocr_guessed_spans(page: fitz.Page, sections: dict,
                      deadline: Optional[float] = None) -> dict:
    """Post-extraction OCR pass for spans flagged corruption_confidence=='guessed'.

    Bounded by design (this used to run per-span, serially, re-rasterizing the
    page region each time — 15-minute hangs on corrupt pharma PDFs):
      - skips entirely if more than OCR_SPAN_CAP spans are flagged (the page is
        badly broken; the downstream vision cross-check reads the page image),
      - renders the page bitmap ONCE at OCR_DPI and crops spans in memory,
      - runs Tesseract calls in a small thread pool,
      - respects the extraction wall-clock deadline.

    Mutates the flagged spans in place (sets ocr_text / corruption_confidence)
    and returns a status dict for extraction_meta.
    """
    guessed = [s for s in sections["header_table"] + sections["body"]
               if s.get("corruption_confidence") == "guessed"]
    if not guessed:
        return {"ocr_spans": 0, "ocr_skipped_reason": None}
    if not TESSERACT_AVAILABLE:
        return {"ocr_spans": 0, "ocr_skipped_reason": "ocr_unavailable"}
    if len(guessed) > OCR_SPAN_CAP:
        logger.warning(
            "Skipping OCR: %d guessed spans exceeds cap of %d — page flagged for review",
            len(guessed), OCR_SPAN_CAP,
        )
        return {"ocr_spans": 0,
                "ocr_skipped_reason": "ocr_skipped_too_many_corrupt_spans"}
    if deadline is not None and time.monotonic() > deadline:
        return {"ocr_spans": 0, "ocr_skipped_reason": "extraction_timeout"}

    try:
        # One rasterization per page, cropped in memory per span
        scale = OCR_DPI / 72
        pix = page.get_pixmap(dpi=OCR_DPI)
        page_img = Image.open(io.BytesIO(pix.tobytes("png")))
    except Exception as e:
        logger.warning(f"OCR page render failed: {e}")
        return {"ocr_spans": 0, "ocr_skipped_reason": "ocr_render_failed"}

    pad = round(2 * scale)  # ~2pt padding so glyph edges aren't clipped

    def do_ocr(span):
        if deadline is not None and time.monotonic() > deadline:
            return None
        try:
            x0, y0, x1, y1 = span["bbox"]
            crop = page_img.crop((
                max(0, int(x0 * scale) - pad),
                max(0, int(y0 * scale) - pad),
                min(page_img.width, int(x1 * scale) + pad),
                min(page_img.height, int(y1 * scale) + pad),
            ))
            if crop.width < 2 or crop.height < 2:
                return None
            # psm 7 = treat image as a single text line (spans are single-line)
            return pytesseract.image_to_string(
                crop, lang=get_ocr_lang(), config="--psm 7"
            ).strip() or None
        except Exception as e:
            logger.warning(f"OCR crop failed for bbox {span.get('bbox')}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=min(OCR_MAX_WORKERS, len(guessed))) as pool:
        results = list(pool.map(do_ocr, guessed))

    disagreements = 0
    for span, ocr_text in zip(guessed, results):
        span["ocr_text"] = ocr_text
        if ocr_text is not None:
            if (_normalize_for_ocr_compare(ocr_text)
                    != _normalize_for_ocr_compare(span["text"])):
                span["corruption_confidence"] = "ocr_disagreement"
                disagreements += 1
            else:
                # OCR independently confirms the guess — treat as resolved
                span["corruption_confidence"] = "ocr_confirmed"
        logger.info(
            "Guessed ligature repair: font=%r guessed=%r ocr=%r confidence=%s",
            span.get("font_name_full"), span["text"], ocr_text,
            span["corruption_confidence"],
        )

    timed_out = deadline is not None and time.monotonic() > deadline
    return {"ocr_spans": len(guessed),
            "ocr_disagreements": disagreements,
            "ocr_skipped_reason": "extraction_timeout" if timed_out else None}


def _normalize_for_ocr_compare(text: str) -> str:
    """Normalize text for OCR-vs-repair comparison (whitespace/case tolerant)."""
    return re.sub(r'\s+', ' ', text).strip().lower()


# ═══════════════════════════════════════════════════════════════
# ANNOTATION DETECTION
# ═══════════════════════════════════════════════════════════════

def is_annotation(text: str, color_hex: str) -> bool:
    """Detect dimension marker annotations."""
    color_match = color_hex.lower() in ANNOTATION_COLORS
    pattern_match = bool(DIMENSION_PATTERN.match(text.strip()))
    arrow_match = text.strip() in ARROW_CHARS
    return (color_match and pattern_match) or arrow_match


# ═══════════════════════════════════════════════════════════════
# ROTATION DETECTION
# ═══════════════════════════════════════════════════════════════

def direction_to_rotation(dir_x: float, dir_y: float) -> int:
    """Convert direction vector to rotation degrees (0, 90, 180, 270)."""
    angle = math.degrees(math.atan2(dir_y, dir_x))
    angle = round(angle)
    if angle < 0:
        angle += 360
    return (round(angle / 90) * 90) % 360


# ═══════════════════════════════════════════════════════════════
# ZONE DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_zones(paths: list, page_width_pt: float, page_height_pt: float,
                 header_threshold_pt: float) -> list:
    """Detect panel zones from path rectangles (fold lines)."""
    vertical_lines = []

    for path in paths:
        bbox = path.get("bbox_pt", [0, 0, 0, 0])
        x0, y0, x1, y1 = bbox
        width = abs(x1 - x0)
        height = abs(y1 - y0)

        # Vertical fold line: narrow + tall
        if width < 3 and height > page_height_pt * 0.3:
            vertical_lines.append(round(x0))

    vertical_lines = sorted(set(vertical_lines))

    zones = []
    if len(vertical_lines) >= 2:
        all_x = sorted(set([0] + vertical_lines + [round(page_width_pt)]))
        for i in range(len(all_x) - 1):
            x_start, x_end = all_x[i], all_x[i + 1]
            if (x_end - x_start) < 20:
                continue
            zones.append({
                "zone_id": f"panel_{i}",
                "bbox_pt": [x_start, header_threshold_pt, x_end, page_height_pt],
                "bbox_mm": bbox_to_mm((x_start, header_threshold_pt, x_end, page_height_pt)),
                "width_mm": pt_to_mm(x_end - x_start),
            })

    return zones


def assign_span_to_zone(span_bbox: list, zones: list) -> Optional[str]:
    """Assign a span to the zone containing its center point."""
    if not zones:
        return None
    cx = (span_bbox[0] + span_bbox[2]) / 2
    cy = (span_bbox[1] + span_bbox[3]) / 2
    for zone in zones:
        zb = zone["bbox_pt"]
        if zb[0] <= cx <= zb[2] and zb[1] <= cy <= zb[3]:
            return zone["zone_id"]
    return "outside"


# ═══════════════════════════════════════════════════════════════
# EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def extract_text_spans_enhanced(page: fitz.Page, header_threshold: float) -> dict:
    """Extract text with rotation, repair, and annotation filtering."""
    header_table = []
    body = []
    annotations = []
    annotation_near_misses = []

    text_dict = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    body_spans_with_pos = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            line_dir = line.get("dir", (1.0, 0.0))

            for span in line.get("spans", []):
                # Build text from chars if available
                chars = span.get("chars", [])
                if chars:
                    raw_text = "".join(ch.get("c", "") for ch in chars)
                else:
                    raw_text = span.get("text", "")

                if not raw_text.strip():
                    continue

                bbox = span.get("bbox", (0, 0, 0, 0))
                font_name = span.get("font", "Unknown")
                font_size = snap_size(span.get("size", 0))
                color = span.get("color", 0)
                flags = span.get("flags", 0)

                rotation_deg = direction_to_rotation(line_dir[0], line_dir[1])

                if isinstance(color, int):
                    color_hex = int_color_to_hex(color)
                else:
                    color_hex = rgb_to_hex(color)

                repaired_text, was_repaired, repair_log, corruption_confidence = repair_text(raw_text)

                # NOTE: OCR for guessed repairs happens in ocr_guessed_spans()
                # as a bounded post-pass — never per-span in this hot loop.

                # Annotation check
                if is_annotation(repaired_text, color_hex):
                    annotations.append({
                        "text": repaired_text,
                        "color_hex": color_hex,
                        "type": "dimension_marker",
                        "bbox": [snap_coord(b) for b in bbox],
                        "bbox_mm": bbox_to_mm(bbox),
                    })
                    continue

                # Near-miss: looks like a dimension marker but color not in
                # ANNOTATION_COLORS — either a new annotation color we haven't
                # added yet, or a real body-text dimension. Log + flag it.
                if DIMENSION_PATTERN.match(repaired_text.strip()) and color_hex.lower() not in ANNOTATION_COLORS:
                    annotation_near_misses.append({
                        "text": repaired_text,
                        "color_hex": color_hex,
                        "bbox_mm": bbox_to_mm(bbox),
                    })
                    logger.info(
                        "Annotation near-miss: dimension-like text %r with color %s not in ANNOTATION_COLORS",
                        repaired_text, color_hex,
                    )

                span_data = {
                    "text": repaired_text,
                    "text_raw": raw_text if was_repaired else None,
                    "repaired": was_repaired,
                    "repair_log": repair_log if was_repaired else None,
                    "corruption_confidence": corruption_confidence,
                    "ocr_text": None,  # filled by ocr_guessed_spans() post-pass
                    "font_name": get_base_font_name(font_name),
                    "font_name_full": font_name,
                    "font_size_pt": font_size,
                    "is_bold": is_bold(font_name, flags),
                    "is_italic": is_italic(font_name, flags),
                    "color_hex": color_hex,
                    "rotation_deg": rotation_deg,
                    "bbox": [snap_coord(b) for b in bbox],
                    "bbox_mm": bbox_to_mm(bbox),
                }

                if bbox[1] < header_threshold:
                    header_table.append(span_data)
                else:
                    body_spans_with_pos.append({
                        "data": span_data,
                        "y0": bbox[1], "y1": bbox[3],
                        "font_size": font_size,
                    })

    # Sort body by position
    body_spans_with_pos.sort(key=lambda x: (x["y0"], x["data"]["bbox"][0]))

    # Line spacing: always computed here (cheap O(n)); extract_page_data strips
    # the key for non-insert pages. This avoids re-parsing the entire page a
    # second time just to add spacing when a page turns out to be an insert.
    for i, info in enumerate(body_spans_with_pos):
        sd = info["data"]
        if i < len(body_spans_with_pos) - 1:
            gap = body_spans_with_pos[i + 1]["y0"] - info["y1"]
            fs = info["font_size"]
            sd["line_spacing"] = round((gap + fs) / fs, 2) if fs > 0 else None
        else:
            sd["line_spacing"] = None
        body.append(sd)

    return {"header_table": header_table, "body": body, "annotations": annotations,
            "annotation_near_misses": annotation_near_misses}


def extract_paths_enhanced(page: fitz.Page) -> list:
    """Extract paths with raw pt bbox for zone detection."""
    paths = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return paths

    for drawing in drawings:
        stroke_color = drawing.get("color")
        fill_color = drawing.get("fill")
        stroke_width = drawing.get("width", 0)
        rect = drawing.get("rect")
        if rect is None:
            continue

        paths.append({
            "stroke_width_pt": round(stroke_width, 2) if stroke_width else 0,
            "stroke_color_hex": rgb_to_hex(stroke_color) if stroke_color else None,
            "fill_color_hex": rgb_to_hex(fill_color) if fill_color else None,
            "color_hex": rgb_to_hex(stroke_color) if stroke_color else rgb_to_hex(fill_color),
            "bbox": [snap_coord(r) for r in rect],
            "bbox_pt": list(rect),
            "bbox_mm": bbox_to_mm(rect),
        })
    return paths


def extract_images(page: fitz.Page) -> list:
    """Extract all embedded images with metadata."""
    images = []
    try:
        image_list = page.get_images(full=True)
    except Exception:
        return images

    for img_info in image_list:
        xref = img_info[0]
        # Pixel dimensions come straight from the get_images() tuple —
        # doc.extract_image() would decode the full image binary just to
        # read the same width/height, which is wasteful on artwork PDFs.
        width_px = img_info[2] if len(img_info) > 2 else 0
        height_px = img_info[3] if len(img_info) > 3 else 0
        try:
            img_rects = page.get_image_rects(xref)
            for rect in img_rects:
                image_data = {
                    "bbox": [snap_coord(r) for r in rect],
                    "bbox_mm": bbox_to_mm(rect),
                    "width_px": width_px, "height_px": height_px,
                    "width_mm": pt_to_mm(rect.width),
                    "height_mm": pt_to_mm(rect.height),
                }
                if height_px > 0:
                    image_data["aspect_ratio"] = round(width_px / height_px, 2)
                images.append(image_data)
        except Exception:
            continue
    return images


# ═══════════════════════════════════════════════════════════════
# PAGE-LEVEL EXTRACTION
# ═══════════════════════════════════════════════════════════════

# Fuzzy "component type" matching — tolerate one corrupted/missing/extra char
# per word so a corrupted span doesn't silently fall back to the generic
# default page name.

def _within_one_edit(a: str, b: str) -> bool:
    """True if a and b are within Levenshtein distance 1."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    # la <= lb, differ by 0 or 1
    i = j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            if la == lb:
                i += 1  # substitution
            j += 1      # deletion in b (or skip)
    edits += (lb - j) + (la - i)
    return edits <= 1


def _find_component_type_end(text: str) -> Optional[int]:
    """Find 'component type' in text (exact, then fuzzy with 1-char tolerance
    per word). Returns the index just past the match, or None."""
    m = re.search(r'(?i)component\s+type', text)
    if m:
        return m.end()
    # Fuzzy: scan consecutive word pairs
    for wm in re.finditer(r'\S+', text):
        w1 = wm.group()
        rest = text[wm.end():]
        wm2 = re.match(r'\s+(\S+)', rest)
        if not wm2:
            continue
        w2 = wm2.group(1)
        # Strip trailing punctuation like ':' from w2 for comparison
        w2_clean = w2.rstrip(':-')
        if (_within_one_edit(w1.lower(), 'component')
                and _within_one_edit(w2_clean.lower(), 'type')):
            return wm.end() + wm2.end() - (len(w2) - len(w2_clean))
    return None


def _detect_page_name_from_content(sections: dict, default_name: str) -> tuple:
    """Dynamically detect page name from 'Component Type' text spans.

    Returns (name, detected) — detected=False means we fell back to the
    generic default_name and page-name detection effectively failed.
    """
    spans = sections.get("header_table", []) + sections.get("body", [])

    for span in spans:
        text = span.get("text", "").strip()
        match_end = _find_component_type_end(text)
        if match_end is not None:
            inline_val = re.sub(r'^\s*[:\-]?\s*', '', text[match_end:]).strip()
            if inline_val:
                return inline_val, True

            my_bbox = span["bbox"]
            my_cy = (my_bbox[1] + my_bbox[3]) / 2

            candidates = []
            for other in spans:
                if other == span:
                    continue
                ob = other["bbox"]
                ocy = (ob[1] + ob[3]) / 2
                if abs(ocy - my_cy) < 15 and ob[0] > my_bbox[0]:
                    candidates.append(other)

            if candidates:
                candidates.sort(key=lambda x: x["bbox"][0])
                val = candidates[0].get("text", "").strip()
                if val:
                    return val, True

    return default_name, False


def extract_page_data(page: fitz.Page, page_index: int,
                      deadline: Optional[float] = None) -> dict:
    """Extract all data from a single page with enhanced features."""
    config = PAGE_CONFIG.get(page_index, {"key": f"page{page_index+1}", "name": f"page {page_index+1}"})

    rect = page.rect
    width_pt, height_pt = rect.width, rect.height
    header_threshold = height_pt * 0.20

    # 1. Paths first (zone detection needs them)
    paths = extract_paths_enhanced(page)

    # 2. Zone detection
    zones = detect_zones(paths, width_pt, height_pt, header_threshold)

    # 3. Text extraction — single pass; line_spacing is always computed and
    # stripped below for non-insert pages (no second full parse needed)
    sections = extract_text_spans_enhanced(page, header_threshold)

    # 4. Detect dynamic page name from extracted text
    detected_name, page_name_detected = _detect_page_name_from_content(sections, config["name"])

    # 5. Check if it's actually an insert based on REAL content
    is_insert = "insert" in detected_name.lower()

    # 6. Non-inserts don't expose line_spacing (preserves original output shape)
    if not is_insert:
        for span in sections["body"]:
            span.pop("line_spacing", None)

    # 6b. Bounded OCR post-pass for guessed repairs — runs ONCE, after the
    # final sections are settled (never inside the span hot loop).
    ocr_info = ocr_guessed_spans(page, sections, deadline)

    # 7. Assign spans to zones
    for span in sections["body"]:
        span["zone"] = assign_span_to_zone(span["bbox"], zones)

    # 4b. Column-aware re-sort: zone → X-column bucket (40pt ≈ 14mm) → Y → X
    # Uses left edge (bbox[0]) NOT center X.
    # Reason: narrow spans like "supra" (x=246–259, cx=252.5) would bucket
    # differently from wider siblings also starting at x=246 (cx=261+) if we
    # used center X. Left edge ensures all items starting at the same column
    # margin land in the same bucket regardless of width.
    def _col_sort_key(s, col_w_pt=40):
        b = s["bbox"]
        left_x = b[0]   # left edge — consistent column alignment
        zone_id = s.get("zone") or "z_outside"
        col_bucket = round(left_x / col_w_pt)
        return (zone_id, col_bucket, b[1], b[0])

    sections["body"].sort(key=_col_sort_key)

    # 5. Images
    images = extract_images(page)

    # 6. Build convenience strings
    body_normal = [s["text"] for s in sections["body"] if s.get("rotation_deg", 0) == 0]
    body_text = " ".join(body_normal)

    rotated_elements = [
        {"text": s["text"], "rotation_deg": s["rotation_deg"],
         "bbox_mm": s["bbox_mm"], "zone": s.get("zone")}
        for s in sections["body"] if s.get("rotation_deg", 0) != 0
    ]

    # Count repairs + unresolved (guessed / OCR-disagreement) corruption
    all_spans = sections["body"] + sections["header_table"]
    repair_count = sum(1 for s in all_spans if s.get("repaired"))
    unresolved_corruption_count = sum(
        1 for s in all_spans
        if s.get("corruption_confidence") in ("guessed", "ocr_disagreement")
    )
    guessed_fonts = sorted({
        s.get("font_name_full") for s in all_spans
        if s.get("corruption_confidence") in ("guessed", "ocr_disagreement", "ocr_confirmed")
    })

    # Clean up paths for output
    clean_paths = []
    for p in paths:
        cp = {k: v for k, v in p.items() if k != "bbox_pt"}
        clean_paths.append(cp)

    # Clean zone output (remove internal bbox_pt)
    clean_zones = []
    for z in zones:
        clean_zones.append({k: v for k, v in z.items() if k != "bbox_pt"})

    return {
        "name": detected_name,
        "sections": sections,
        "body_text": body_text,
        "rotated_elements": rotated_elements,
        "zones": clean_zones,
        "page_dimensions": {
            "width_pt": round(width_pt, 2),
            "height_pt": round(height_pt, 2),
            "width_mm": pt_to_mm(width_pt),
            "height_mm": pt_to_mm(height_pt),
        },
        "paths": clean_paths,
        "images": images,
        "extraction_meta": {
            "text_repairs_applied": repair_count,
            "unresolved_corruption_count": unresolved_corruption_count,
            "guessed_repair_fonts": guessed_fonts,
            "annotations_filtered": len(sections["annotations"]),
            "annotation_near_misses": len(sections.get("annotation_near_misses", [])),
            "zones_detected": len(zones),
            "rotated_elements_count": len(rotated_elements),
            "page_name_detected": page_name_detected,
            "ocr_spans": ocr_info.get("ocr_spans", 0),
            "ocr_skipped_reason": ocr_info.get("ocr_skipped_reason"),
        },
        # One clean object the workflow can branch on
        "extraction_confidence": {
            "repairs_applied": repair_count,
            "unresolved_corruption_count": unresolved_corruption_count,
            "annotations_filtered": len(sections["annotations"]),
            "annotation_near_misses": len(sections.get("annotation_near_misses", [])),
            "zones_detected": len(zones),
            "page_name_detected": page_name_detected,
            "ocr_available": TESSERACT_AVAILABLE,
            "ocr_skipped_reason": ocr_info.get("ocr_skipped_reason"),
            "needs_review": (
                unresolved_corruption_count > 0
                or not page_name_detected
                or len(sections.get("annotation_near_misses", [])) > 0
                or ocr_info.get("ocr_skipped_reason") is not None
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════
# DOCUMENT-LEVEL EXTRACTION
# ═══════════════════════════════════════════════════════════════

def render_page_png(page: fitz.Page, dpi: int = 200) -> dict:
    """Render a page to base64 PNG with the scale info needed to map
    pixel coordinates back to the pt/mm bboxes returned by /extract."""
    pix = page.get_pixmap(dpi=dpi)
    return {
        "png_base64": base64.b64encode(pix.tobytes("png")).decode("ascii"),
        "dpi": dpi,
        "width_px": pix.width,
        "height_px": pix.height,
        # px = pt * (dpi / 72); mm = px / px_per_mm
        "scale_px_per_pt": round(dpi / 72, 6),
        "scale_px_per_mm": round(dpi / 72 / PT_TO_MM, 6),
    }


def extract_artwork(pdf_bytes: bytes, filename: str = "", render: bool = False,
                    render_dpi: int = 200) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    result = {}
    failed_pages = []
    # Wall-clock guard: heavy optional work (OCR) degrades gracefully instead
    # of blowing past the platform proxy timeout and 502-ing the whole request.
    start = time.monotonic()
    deadline = start + EXTRACTION_TIME_BUDGET_S

    # Iterate dynamically over all pages instead of capping at 5
    for page_index in range(len(doc)):
        if page_index in SKIP_PAGES:
            continue
        page = doc[page_index]
        # Provide fallback config for pages not explicitly defined in PAGE_CONFIG
        config = PAGE_CONFIG.get(page_index, {"key": f"page{page_index+1}", "name": f"page {page_index+1}"})
        # Per-page isolation: one bad page must not lose the whole document
        try:
            page_data = extract_page_data(page, page_index, deadline)
        except Exception as e:
            logger.exception(f"Page {page_index + 1} extraction failed ({filename})")
            failed_pages.append(page_index + 1)
            page_data = {
                "name": config["name"],
                "error": f"Page extraction failed: {str(e)}",
                "extraction_confidence": {"needs_review": True, "extraction_failed": True},
            }
        if render:
            try:
                page_data["render"] = render_page_png(page, dpi=render_dpi)
            except Exception as e:
                logger.exception(f"Page {page_index + 1} render failed ({filename})")
                page_data["render"] = {"error": str(e)}
        result[config["key"]] = page_data

    doc.close()

    # Document-level review flag — n8n should key off this to route pages
    # to human review / vision cross-check instead of trusting them blind.
    needs_review = bool(failed_pages) or any(
        p.get("extraction_confidence", {}).get("needs_review")
        for p in result.values()
    )
    total_repairs = sum(
        p.get("extraction_meta", {}).get("text_repairs_applied", 0)
        for p in result.values()
    )
    total_unresolved = sum(
        p.get("extraction_meta", {}).get("unresolved_corruption_count", 0)
        for p in result.values()
    )
    elapsed = round(time.monotonic() - start, 2)
    ocr_skip_reasons = sorted({
        r for p in result.values()
        for r in [p.get("extraction_confidence", {}).get("ocr_skipped_reason")]
        if r
    })
    result["document_meta"] = {
        "filename": filename,
        "page_count": len([k for k in result if k.startswith("page")]),
        "failed_pages": failed_pages,
        "total_repairs_applied": total_repairs,
        "total_unresolved_corruption": total_unresolved,
        "ocr_available": TESSERACT_AVAILABLE,
        "ocr_skip_reasons": ocr_skip_reasons,
        "extraction_seconds": elapsed,
        "needs_review": needs_review,
    }
    logger.info(
        "Extracted %s: pages=%d failed=%s repairs=%d unresolved_corruption=%d "
        "ocr_skips=%s needs_review=%s elapsed=%.2fs",
        filename or "<upload>", result["document_meta"]["page_count"],
        failed_pages, total_repairs, total_unresolved,
        ocr_skip_reasons, needs_review, elapsed,
    )
    return result


# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/extract")
async def extract_pdf(file: UploadFile = File(...), render: bool = False,
                      render_dpi: int = 200):
    """Extract artwork compliance data from a PDF file.

    Query params:
      - render=true      → include a base64 PNG render of each page (page_data["render"])
      - render_dpi=200   → DPI for the render (used with render=true)
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if render and not 30 <= render_dpi <= 600:
        raise HTTPException(status_code=400, detail="render_dpi must be between 30 and 600")
    try:
        pdf_bytes = await file.read()
        # Run in threadpool to avoid blocking the async event loop for large PDFs
        result = await run_in_threadpool(
            extract_artwork, pdf_bytes, file.filename, render, render_dpi
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"Extraction failed for {file.filename}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


def _render_all_pages(pdf_bytes: bytes, dpi: int) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_index in range(len(doc)):
        page_render = render_page_png(doc[page_index], dpi=dpi)
        page_render["page_number"] = page_index + 1
        pages.append(page_render)
    doc.close()
    return {"page_count": len(pages), "dpi": dpi, "pages": pages}


@app.post("/render-page")
async def render_pages(file: UploadFile = File(...), dpi: int = 200):
    """Render each PDF page as a base64 PNG (for vision cross-check).

    Response includes the DPI and px-per-pt / px-per-mm scale factors so pixel
    coordinates can be mapped back to /extract's bbox / bbox_mm values.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if not 30 <= dpi <= 600:
        raise HTTPException(status_code=400, detail="dpi must be between 30 and 600")
    try:
        pdf_bytes = await file.read()
        result = await run_in_threadpool(_render_all_pages, pdf_bytes, dpi)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Render failed for {file.filename}")
        raise HTTPException(status_code=500, detail=f"Error rendering PDF: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# GPMI TEXT EXTRACTION — matches DOCX extractor output format
# Preserves superscript/subscript via font-size + position heuristic
# ═══════════════════════════════════════════════════════════════

def _extract_gpmi_page(page: fitz.Page) -> list:
    """Extract paragraphs from a single PDF page with <sup>/<sub> tags."""
    text_dict = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    paragraphs = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # text blocks only
            continue

        block_lines = []
        for line in block.get("lines", []):
            line_spans = []
            for span in line.get("spans", []):
                chars = span.get("chars", [])
                if chars:
                    raw_text = "".join(ch.get("c", "") for ch in chars)
                else:
                    raw_text = span.get("text", "")

                if not raw_text:
                    continue

                repaired_text, _, _, _ = repair_text(raw_text)

                line_spans.append({
                    "text": repaired_text,
                    "font_size": snap_size(span.get("size", 0)),
                    "top_y": span.get("bbox", (0, 0, 0, 0))[1],
                    "bot_y": span.get("bbox", (0, 0, 0, 0))[3],
                })

            if not line_spans:
                continue

            # Find dominant (largest) font size in this line
            dominant_size = max(s["font_size"] for s in line_spans)

            # Build line text with super/sub detection
            line_text = ""
            for span in line_spans:
                text = span["text"]

                if dominant_size > 0 and span["font_size"] < dominant_size * 0.82:
                    # Significantly smaller font → check vertical position
                    dom_spans = [s for s in line_spans
                                 if s["font_size"] >= dominant_size * 0.82]
                    if dom_spans:
                        dom_mid = sum((s["top_y"] + s["bot_y"]) / 2
                                      for s in dom_spans) / len(dom_spans)
                        span_mid = (span["top_y"] + span["bot_y"]) / 2

                        if span_mid < dom_mid - 0.3:
                            text = f"<sup>{text}</sup>"
                        elif span_mid > dom_mid + 0.3:
                            text = f"<sub>{text}</sub>"

                line_text += text

            stripped = line_text.strip()
            if stripped:
                block_lines.append(stripped)

        # Join lines in same block as one paragraph
        para = " ".join(block_lines)
        if para.strip() and len(para.strip()) > 1:
            paragraphs.append(para.strip())

    return paragraphs


@app.post("/extract-gpmi")
async def extract_gpmi(file: UploadFile = File(...)):
    """
    Extract PDF text in GPMI format — paragraph-indexed with <sup>/<sub> preserved.
    Output matches the DOCX extractor: { paragraphCount, text, fileName }
    where text = "[0] first para\\n[1] second para\\n..."
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    try:
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        all_paragraphs = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            all_paragraphs.extend(_extract_gpmi_page(page))

        doc.close()

        indexed_text = "\n".join(f"[{i}] {p}" for i, p in enumerate(all_paragraphs))
        clean_name = file.filename.rsplit(".", 1)[0] if "." in file.filename else file.filename

        return JSONResponse(content={
            "paragraphCount": len(all_paragraphs),
            "text": indexed_text,
            "fileName": clean_name,
            "fileType": "pdf",
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPMI extraction failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "artwork-extractor",
        "version": "2.1.0",
        "ocr_available": TESSERACT_AVAILABLE,
    }


@app.get("/")
async def root():
    return {
        "service": "Artwork Compliance Extractor",
        "version": "2.1.0",
        "features": [
            "Zone detection (panel boundaries from fold lines)",
            "Rotation detection (flap text, rotated elements)",
            "Ligature repair (ti, fi, fl auto-fix with audit trail + confidence tiers)",
            "OCR crop fallback for guessed ligature repairs (needs tesseract)",
            "Annotation filtering (dimension markers separated, near-misses flagged)",
            "Page rendering to base64 PNG (?render=true or POST /render-page)",
            "Per-page error isolation + needs_review flags",
        ],
        "endpoints": {
            "POST /extract": "Upload PDF and extract artwork data (?render=true&render_dpi=200 to include page images)",
            "POST /extract-gpmi": "Upload PDF and extract GPMI-format text with sup/sub preserved",
            "POST /render-page": "Render each PDF page as base64 PNG (?dpi=200)",
            "POST /html-to-pdf": "Convert HTML string to PDF binary",
            "GET /health": "Health check",
        },
    }


@app.on_event("startup")
async def startup_browser():
    global _playwright, _browser
    _playwright = await async_playwright().start()
    _browser = await _playwright.chromium.launch(args=["--no-sandbox", "--disable-gpu"])


@app.on_event("shutdown")
async def shutdown_browser():
    global _playwright, _browser
    if _browser:
        await _browser.close()
    if _playwright:
        await _playwright.stop()


@app.post("/html-to-pdf")
async def html_to_pdf(request: Request):
    """Convert HTML body to PDF using headless Chromium."""
    try:
        data = await request.json()
        html_string = data.get("html", "")
        if not html_string:
            raise HTTPException(status_code=400, detail="Missing 'html' field in request body")
        page = await _browser.new_page()
        await page.set_content(html_string, wait_until="networkidle")
        pdf_bytes = await page.pdf(
            format="A4",
            print_background=True,
            margin={"top": "15mm", "bottom": "15mm", "left": "10mm", "right": "10mm"},
        )
        await page.close()
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=compliance_report.pdf"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")