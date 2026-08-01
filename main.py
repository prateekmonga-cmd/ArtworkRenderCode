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

from textrepair import (
    PT_TO_MM, pt_to_mm, bbox_to_mm, rgb_to_hex, int_color_to_hex,
    is_bold, is_italic, get_base_font_name, snap_size, snap_coord,
    ANNOTATION_COLORS, ANNOTATION_RGB, ANNOTATION_COLOR_TOLERANCE,
    is_annotation_color, DIMENSION_PATTERN, ARROW_CHARS,
    PRODUCTION_MARK_TERMS, PRODUCTION_MARK_SHAPE, PRODUCTION_MARK_SUFFIX_MAX,
    KNOWN_CORRECTIONS, MOJIBAKE_RUN, is_plausible_artwork_char,
    repair_mojibake, repair_text, is_annotation, is_production_mark,
    looks_like_production_mark, _normalize_for_ocr_compare,
    direction_to_rotation, ROTATION_SOP,
    _within_one_edit, _find_component_type_end, _label_key,
)

logger = logging.getLogger("artwork-extractor")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# ─── Optional OCR support (graceful degrade if tesseract missing) ───
# OCR is what confirms a guessed ligature repair. Without it every guessed span
# stays unresolved and both pages come back needs_review=True on every run — a
# deployment difference, not a data difference. So say so loudly at startup
# rather than leaving it to be inferred from the output (B-09).
_OCR_BANNER = (
    "OCR UNAVAILABLE (%s). Ligature repairs cannot be confirmed; pages with "
    "guessed corrections will report needs_review=True. Install tesseract-ocr "
    "and the tesseract-ocr-spa language pack in the deployment image."
)

try:
    import pytesseract
    from PIL import Image
    try:
        pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
    except Exception as _e:
        TESSERACT_AVAILABLE = False
        logger.error(_OCR_BANNER, f"tesseract binary not found: {_e}")
except ImportError as _e:
    TESSERACT_AVAILABLE = False
    logger.error(_OCR_BANNER, f"pytesseract/Pillow not installed: {_e}")

_ocr_lang = None

def get_ocr_lang() -> str:
    """Prefer Spanish+English if the spa language pack is installed."""
    global _ocr_lang
    if _ocr_lang is None:
        _ocr_lang = "eng"
        try:
            if "spa" in pytesseract.get_languages(config=""):
                _ocr_lang = "spa+eng"
            else:
                logger.error(
                    "OCR language pack 'spa' is MISSING — Spanish artwork will be "
                    "read with the English model, so accented characters will be "
                    "misread and repairs may be wrongly marked ocr_disagreement. "
                    "Install tesseract-ocr-spa."
                )
        except Exception as e:
            logger.warning("Could not list tesseract languages (%s); defaulting to eng", e)
    return _ocr_lang


def log_ocr_status() -> None:
    """Report the OCR posture once, at app startup, at a level that shows up."""
    if TESSERACT_AVAILABLE:
        logger.info("OCR available: tesseract %s, languages=%s",
                    pytesseract.get_tesseract_version(), get_ocr_lang())
    else:
        logger.error(_OCR_BANNER, "see startup import errors above")

# Global browser instance (reused across requests)
_playwright = None
_browser = None


app = FastAPI(
    title="Artwork Compliance Extractor",
    description="Enhanced extraction with zone detection, rotation, and text repair",
    version="2.0.0"
)

from textrepair import (
    PT_TO_MM, pt_to_mm, bbox_to_mm, rgb_to_hex, int_color_to_hex,
    is_bold, is_italic, get_base_font_name, snap_size, snap_coord,
    ANNOTATION_COLORS, ANNOTATION_RGB, ANNOTATION_COLOR_TOLERANCE,
    is_annotation_color, DIMENSION_PATTERN, ARROW_CHARS,
    PRODUCTION_MARK_TERMS, PRODUCTION_MARK_SHAPE, PRODUCTION_MARK_SUFFIX_MAX,
    KNOWN_CORRECTIONS, MOJIBAKE_RUN, is_plausible_artwork_char,
    repair_mojibake, repair_text, is_annotation, is_production_mark,
    looks_like_production_mark, _normalize_for_ocr_compare,
    direction_to_rotation, ROTATION_SOP,
    _within_one_edit, _find_component_type_end, _label_key,
)

# Page slots are positional only. There is deliberately no name guess here:
# the old table called page 2 a "sticker label" when this artwork's page 2 is a
# foil, and that guess flowed into is_insert, which decides whether
# line_spacing survives. Content detection supplies the real name; when it
# fails, "page N" says so honestly (B-11).
def page_slot(page_index: int) -> dict:
    n = page_index + 1
    return {"key": f"page{n}", "name": f"page {n}"}

# ─── OCR performance bounds ──────────────────────────────────
# OCR is a targeted fallback, not a bulk pass. A page riddled with corrupt
# spans is handled by the downstream vision cross-check instead — OCR-ing
# hundreds of spans serially is what caused the 15-min /extract hang.
OCR_DPI = 200                     # 200 DPI: ~20px glyphs on 6-7pt pharma print (150 was borderline)
OCR_SPAN_CAP = 15                 # >this many guessed spans on a page → skip OCR, flag page
OCR_MAX_WORKERS = 4               # bounded parallel Tesseract calls (Render CPU is small)
EXTRACTION_TIME_BUDGET_S = 120    # wall-clock guard: degrade gracefully, never 502


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



# ═══════════════════════════════════════════════════════════════
# ZONE DETECTION
# ═══════════════════════════════════════════════════════════════

ZONE_MIN_FOLD_RATIO = 0.30   # a fold line must span 30% of the ARTWORK's height
ZONE_MIN_PANEL_PT = 20       # narrower than this is a gap, not a panel
ZONE_Y_TOLERANCE_PT = 12     # spans may sit just outside the drawn artwork box


def detect_zones(paths: list, header_threshold_pt: float) -> list:
    """Detect panel zones from the artwork's vertical fold lines.

    The fold-line height test is measured against the artwork's own bounding
    box, not the sheet. A carton laid out on A4 has fold lines about 32 mm tall
    against an 841 pt page: a 30%-of-page threshold is ~89 mm, so no small
    component could ever satisfy it and every span came back zone=None,
    which left no face model for any placement rule to run against (B-07).
    """
    body_paths = [p for p in paths
                  if p.get("bbox_pt", [0, 0, 0, 0])[1] >= header_threshold_pt]
    if not body_paths:
        return []

    art_x0 = min(p["bbox_pt"][0] for p in body_paths)
    art_x1 = max(p["bbox_pt"][2] for p in body_paths)
    art_y0 = min(p["bbox_pt"][1] for p in body_paths)
    art_y1 = max(p["bbox_pt"][3] for p in body_paths)
    art_h = max(1.0, art_y1 - art_y0)
    min_fold_h = art_h * ZONE_MIN_FOLD_RATIO

    vertical_lines = []
    for path in body_paths:
        x0, y0, x1, y1 = path.get("bbox_pt", [0, 0, 0, 0])
        width = abs(x1 - x0)
        height = abs(y1 - y0)

        # Vertical fold line: narrow + tall relative to the artwork
        if width < 3 and height > min_fold_h:
            vertical_lines.append(round(x0))

    vertical_lines = sorted(set(vertical_lines))

    zones = []
    if len(vertical_lines) >= 2:
        # Bound the panels by the artwork, not the sheet, so the reported
        # zone widths are the real panel widths a placement rule can check.
        all_x = sorted(set([round(art_x0)] + vertical_lines + [round(art_x1)]))
        for i in range(len(all_x) - 1):
            x_start, x_end = all_x[i], all_x[i + 1]
            if (x_end - x_start) < ZONE_MIN_PANEL_PT:
                continue
            bbox = (x_start, art_y0, x_end, art_y1)
            zones.append({
                "zone_id": f"panel_{len(zones)}",
                "bbox_pt": list(bbox),
                "bbox_mm": bbox_to_mm(bbox),
                "width_mm": pt_to_mm(x_end - x_start),
                "height_mm": pt_to_mm(art_y1 - art_y0),
            })

    return zones


def assign_span_to_zone(span_bbox: list, zones: list) -> Optional[str]:
    """Assign a span to the panel whose X range contains its center.

    Panels are vertical strips, so X is the discriminating axis; Y only has to
    fall inside the artwork band, with tolerance for text that overhangs the
    drawn fold box slightly.
    """
    if not zones:
        return None
    cx = (span_bbox[0] + span_bbox[2]) / 2
    cy = (span_bbox[1] + span_bbox[3]) / 2
    for zone in zones:
        zx0, zy0, zx1, zy1 = zone["bbox_pt"]
        if (zx0 <= cx <= zx1
                and (zy0 - ZONE_Y_TOLERANCE_PT) <= cy <= (zy1 + ZONE_Y_TOLERANCE_PT)):
            return zone["zone_id"]
    return "outside"


# ═══════════════════════════════════════════════════════════════
# EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

COLUMN_WIDTH_PT = 40          # ≈14 mm; column bucket width for sorting/spacing
SPACING_SIZE_TOLERANCE = 0.25  # ±25% font size — beyond this it's a new block


def compute_line_spacing(body_spans: list) -> None:
    """Attach line_spacing to each body span, in place.

    Only measured between consecutive spans that share a zone and a column
    bucket and are set at a similar size — i.e. spans that are plausibly two
    lines of the same paragraph. Measuring against whatever span happened to
    sort next produced gaps that crossed columns and faces, which made the
    insert 1.2-spacing requirement unverifiable (B-08).

    Expects the list already sorted by (zone, column, y, x), and each span to
    still carry the private _y0/_y1/_size geometry.
    """
    def bucket(s):
        return (s.get("zone") or "z_outside",
                round(s["bbox"][0] / COLUMN_WIDTH_PT),
                s.get("rotation_deg", 0))

    for i, s in enumerate(body_spans):
        s["line_spacing"] = None
        if i + 1 >= len(body_spans):
            continue
        nxt = body_spans[i + 1]
        if bucket(s) != bucket(nxt):
            continue

        size = s.get("_size") or 0.0
        next_size = nxt.get("_size") or 0.0
        if size <= 0 or next_size <= 0:
            continue
        if abs(next_size - size) > size * SPACING_SIZE_TOLERANCE:
            continue

        # Baseline-to-baseline over font size is the standard leading ratio.
        # Using bottom edges keeps it independent of ascender height.
        delta = nxt.get("_y1", 0) - s.get("_y1", 0)
        if delta <= 0:                       # same line, or out of order
            continue
        if delta > size * 4:                 # paragraph break, not line spacing
            continue
        s["line_spacing"] = round(delta / size, 2)


def extract_text_spans_enhanced(page: fitz.Page, header_threshold: float) -> dict:
    """Extract text with rotation, repair, and annotation filtering."""
    header_table = []
    annotations = []
    annotation_near_misses = []
    production_marks = []
    production_mark_near_misses = []

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
                # Keep the true size alongside the snapped one. Snapping is for
                # grouping and display; a 2.75 pt span snaps to 3.0 and would
                # silently pass a 3 pt minimum if that were the only value (B-05).
                raw_size = span.get("size", 0) or 0.0
                font_size = snap_size(raw_size)
                color = span.get("color", 0)
                flags = span.get("flags", 0)
                in_header = bbox[1] < header_threshold

                rotation_deg = direction_to_rotation(line_dir[0], line_dir[1])

                if isinstance(color, int):
                    color_hex = int_color_to_hex(color)
                else:
                    color_hex = rgb_to_hex(color)

                repaired_text, was_repaired, repair_log, corruption_confidence = repair_text(raw_text)

                # NOTE: OCR for guessed repairs happens in ocr_guessed_spans()
                # as a bounded post-pass — never per-span in this hot loop.

                # Annotation check. Header spans are exempt: a dimension in the
                # tabular header is a declared field value ("60 mm" as Artwork
                # size), not a callout, and must survive (B-01).
                if is_annotation(repaired_text, color_hex, in_header=in_header):
                    annotations.append({
                        "text": repaired_text,
                        "color_hex": color_hex,
                        "type": "dimension_marker",
                        "bbox": [snap_coord(b) for b in bbox],
                        "bbox_mm": bbox_to_mm(bbox),
                    })
                    continue

                # Print-production marks are on the artwork file but not on the
                # finished pack. Pull them out before any compliance analysis so
                # neither the font/size geometry nor the Spanish body text is
                # judged against a printer instruction. They stay in the payload
                # under production_marks so the review agents can say what was
                # excluded and why.
                if is_production_mark(repaired_text):
                    production_marks.append({
                        "text": repaired_text,
                        "reason": "print-production mark",
                        "font_name": get_base_font_name(font_name),
                        "font_size_pt": font_size,
                        "color_hex": color_hex,
                        "bbox_mm": bbox_to_mm(bbox),
                    })
                    continue

                if looks_like_production_mark(repaired_text, color_hex):
                    production_mark_near_misses.append({
                        "text": repaired_text,
                        "color_hex": color_hex,
                        "bbox_mm": bbox_to_mm(bbox),
                    })
                    logger.info(
                        "Production-mark near-miss: %r in annotation color %s "
                        "is not in PRODUCTION_MARK_TERMS",
                        repaired_text, color_hex,
                    )

                # Near-miss: looks like a dimension marker but color not in
                # ANNOTATION_COLORS — either a new annotation color we haven't
                # added yet, or a real body-text dimension. Log + flag it.
                if DIMENSION_PATTERN.match(repaired_text.strip()) and not is_annotation_color(color_hex):
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
                    "font_size_pt": font_size,          # snapped: display/grouping
                    "font_size_pt_raw": round(raw_size, 3),  # authoritative for minimums
                    "is_bold": is_bold(font_name, flags),
                    "is_italic": is_italic(font_name, flags),
                    "color_hex": color_hex,
                    "rotation_deg": rotation_deg,       # PDF text direction
                    "rotation_sop": ROTATION_SOP.get(rotation_deg, str(rotation_deg)),
                    "bbox": [snap_coord(b) for b in bbox],
                    "bbox_mm": bbox_to_mm(bbox),
                    # Unrounded geometry, kept only until line spacing is
                    # computed in extract_page_data; stripped before output.
                    # The public bbox is rounded to whole points, which is up to
                    # 1 pt of error on an 8 pt line — too coarse for a 1.2
                    # spacing check.
                    "_y0": bbox[1], "_y1": bbox[3], "_size": raw_size,
                }

                if in_header:
                    header_table.append(span_data)
                else:
                    body_spans_with_pos.append({
                        "data": span_data,
                        "y0": bbox[1], "y1": bbox[3],
                        "font_size": font_size,
                    })

    # Sort body by position. line_spacing is NOT computed here: at this point
    # the neighbouring span in reading order is frequently in another column,
    # which made the gap meaningless. It is computed in extract_page_data once
    # zones and column buckets are known (B-08).
    body_spans_with_pos.sort(key=lambda x: (x["y0"], x["data"]["bbox"][0]))
    body = [info["data"] for info in body_spans_with_pos]

    return {"header_table": header_table, "body": body, "annotations": annotations,
            "annotation_near_misses": annotation_near_misses,
            "production_marks": production_marks,
            "production_mark_near_misses": production_mark_near_misses}


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


# ─── Header detection (B-10) ─────────────────────────────────────
# The header used to be defined as "anything in the top 20% of the page", which
# means the extractor asserted the very thing the SOP rule "header must be
# positioned in the upper centre" needs to verify — the data could never
# contradict the assumption. Find it by its field labels instead, and report
# where it actually turned out to be.

# Alias -> canonical field name. Aliases are canonicalised because a single
# "AC Reference" line matches both "ac ref" and "ac reference", which would let
# one line satisfy HEADER_MIN_LABELS on its own.
HEADER_FIELD_LABELS = {
    "product name": "product name",
    "ac reference": "ac reference",
    "ac ref": "ac reference",
    "component type": "component type",
    "substrate": "substrate",
    "pantone no": "pantone no",
    "artwork size": "artwork size",
    "flat size": "flat size",
    "print side": "print side",
    "colour code": "colour code",
    "color code": "colour code",
    "market": "market",
    "version": "version",
    "dimension": "dimension",
    "size in mm": "dimension",
}
HEADER_MIN_LABELS = 2          # one stray phrase is not a header
HEADER_FALLBACK_RATIO = 0.20   # top-of-page guess when content detection fails
HEADER_PAD_PT = 6              # tolerance below the lowest matched label line




def detect_header_region(page: fitz.Page, height_pt: float) -> dict:
    """Locate the tabular header by its field labels.

    Returns the split threshold plus the measured geometry, so a placement rule
    has a real bounding box to check against Artwork Headers V2 §3 (Carton
    180x48 mm, Foil 180x36 mm, and so on).
    """
    fallback = height_pt * HEADER_FALLBACK_RATIO
    try:
        words = page.get_text("words")
    except Exception:
        words = []

    lines = {}
    for w in words:
        if len(w) < 8:
            continue
        x0, y0, x1, y1, word, block_no, line_no, _ = w[:8]
        entry = lines.setdefault((block_no, line_no), {"words": [], "bbox": [x0, y0, x1, y1]})
        entry["words"].append(word)
        b = entry["bbox"]
        b[0], b[1] = min(b[0], x0), min(b[1], y0)
        b[2], b[3] = max(b[2], x1), max(b[3], y1)

    matched_bboxes = []
    found_labels = set()
    for entry in lines.values():
        key = _label_key(" ".join(entry["words"]))
        hits = {canon for alias, canon in HEADER_FIELD_LABELS.items() if alias in key}
        if hits:
            found_labels |= hits
            matched_bboxes.append(entry["bbox"])

    if len(found_labels) < HEADER_MIN_LABELS or not matched_bboxes:
        return {
            "threshold_pt": fallback,
            "detected_by_content": False,
            "labels_found": sorted(found_labels),
            "bbox_mm": None, "width_mm": None, "height_mm": None,
            "vertical_position": None, "horizontal_position": None,
        }

    x0 = min(b[0] for b in matched_bboxes)
    y0 = min(b[1] for b in matched_bboxes)
    x1 = max(b[2] for b in matched_bboxes)
    y1 = max(b[3] for b in matched_bboxes)

    page_w = page.rect.width or 1.0
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    left_gap, right_gap = x0, page_w - x1
    if abs(left_gap - right_gap) <= max(page_w * 0.05, 6):
        horizontal = "centre"
    else:
        horizontal = "left" if cx < page_w / 2 else "right"

    third = (height_pt or 1.0) / 3
    vertical = "upper" if cy < third else ("middle" if cy < 2 * third else "lower")

    return {
        # Split just below the label block, not at an arbitrary page fraction.
        "threshold_pt": min(y1 + HEADER_PAD_PT, height_pt),
        "detected_by_content": True,
        "labels_found": sorted(found_labels),
        "bbox_mm": bbox_to_mm((x0, y0, x1, y1)),
        "width_mm": pt_to_mm(x1 - x0),
        "height_mm": pt_to_mm(y1 - y0),
        "vertical_position": vertical,
        "horizontal_position": horizontal,
    }


def extract_page_data(page: fitz.Page, page_index: int,
                      deadline: Optional[float] = None) -> dict:
    """Extract all data from a single page with enhanced features."""
    config = page_slot(page_index)

    rect = page.rect
    width_pt, height_pt = rect.width, rect.height

    # Header located by its field labels, with a top-of-page fallback (B-10).
    header_region = detect_header_region(page, height_pt)
    header_threshold = header_region["threshold_pt"]

    # 1. Paths first (zone detection needs them)
    paths = extract_paths_enhanced(page)

    # 2. Zone detection
    zones = detect_zones(paths, header_threshold)

    # 3. Text extraction — single pass; line_spacing is always computed and
    # stripped below for non-insert pages (no second full parse needed)
    sections = extract_text_spans_enhanced(page, header_threshold)

    # 4. Detect dynamic page name from extracted text
    detected_name, page_name_detected = _detect_page_name_from_content(sections, config["name"])

    # 5. Check if it's actually an insert based on REAL content
    is_insert = "insert" in detected_name.lower()

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
    def _col_sort_key(s, col_w_pt=COLUMN_WIDTH_PT):
        b = s["bbox"]
        left_x = b[0]   # left edge — consistent column alignment
        zone_id = s.get("zone") or "z_outside"
        col_bucket = round(left_x / col_w_pt)
        return (zone_id, col_bucket, b[1], b[0])

    sections["body"].sort(key=_col_sort_key)

    # 7b. Line spacing, now that zone and column are known. Non-inserts don't
    # expose it (preserves the original output shape), but a page whose name was
    # never detected keeps it — dropping it on a page that might BE an insert
    # loses the only data its own rule needs (B-11).
    compute_line_spacing(sections["body"])
    if not is_insert and page_name_detected:
        for span in sections["body"]:
            span.pop("line_spacing", None)
    for span in sections["body"] + sections["header_table"]:
        for k in ("_y0", "_y1", "_size"):
            span.pop(k, None)

    # 5. Images
    images = extract_images(page)

    # 6. Build convenience strings.
    # body_text keeps its original meaning (upright spans only) so existing
    # consumers don't shift under them. body_text_all is the complete one:
    # the SOP *requires* the AC reference and the batch-coding labels to be
    # rotated, so the elements most certain to be rotated were exactly the ones
    # a presence check on body_text could never find (B-03). Presence rules
    # should read body_text_all; rotation rules read rotated_elements.
    body_normal = [s["text"] for s in sections["body"] if s.get("rotation_deg", 0) == 0]
    body_text = " ".join(body_normal)
    body_text_all = " ".join(s["text"] for s in sections["body"])

    rotated_elements = [
        {"text": s["text"], "rotation_deg": s["rotation_deg"],
         "rotation_sop": s.get("rotation_sop"),
         "bbox_mm": s["bbox_mm"], "zone": s.get("zone")}
        for s in sections["body"] if s.get("rotation_deg", 0) != 0
    ]

    # Count repairs + unresolved corruption. "failed" counts too: it marks
    # U+FFFD damage we deliberately refused to guess at, which is precisely the
    # case a human needs to look at (B-06).
    UNRESOLVED = ("guessed", "ocr_disagreement", "failed")
    all_spans = sections["body"] + sections["header_table"]
    repair_count = sum(1 for s in all_spans if s.get("repaired"))
    unresolved_corruption_count = sum(
        1 for s in all_spans
        if s.get("corruption_confidence") in UNRESOLVED
    )
    guessed_fonts = sorted({
        s.get("font_name_full") for s in all_spans
        if s.get("corruption_confidence") in UNRESOLVED + ("ocr_confirmed",)
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
        "body_text_all": body_text_all,
        "rotated_elements": rotated_elements,
        "zones": clean_zones,
        # Measured header geometry, so the "header must be in the upper centre"
        # rule has something it can actually test (B-10).
        "header_region": {
            k: v for k, v in header_region.items() if k != "threshold_pt"
        },
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
            "production_marks_excluded": len(sections.get("production_marks", [])),
            "production_mark_near_misses": len(sections.get("production_mark_near_misses", [])),
            "zones_detected": len(zones),
            "rotated_elements_count": len(rotated_elements),
            "page_name_detected": page_name_detected,
            "header_detected_by_content": header_region["detected_by_content"],
            "ocr_spans": ocr_info.get("ocr_spans", 0),
            "ocr_skipped_reason": ocr_info.get("ocr_skipped_reason"),
        },
        # One clean object the workflow can branch on
        "extraction_confidence": {
            "repairs_applied": repair_count,
            "unresolved_corruption_count": unresolved_corruption_count,
            "annotations_filtered": len(sections["annotations"]),
            "annotation_near_misses": len(sections.get("annotation_near_misses", [])),
            "production_marks_excluded": len(sections.get("production_marks", [])),
            "production_mark_near_misses": len(sections.get("production_mark_near_misses", [])),
            "zones_detected": len(zones),
            "page_name_detected": page_name_detected,
            "header_detected_by_content": header_region["detected_by_content"],
            "ocr_available": TESSERACT_AVAILABLE,
            "ocr_skipped_reason": ocr_info.get("ocr_skipped_reason"),
            "needs_review": (
                unresolved_corruption_count > 0
                or not page_name_detected
                or not header_region["detected_by_content"]
                or len(sections.get("annotation_near_misses", [])) > 0
                or len(sections.get("production_mark_near_misses", [])) > 0
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
        page = doc[page_index]
        config = page_slot(page_index)
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
        # Surfaced so a deployment can be checked without reading logs: "eng"
        # alone on Spanish artwork means the spa pack is missing (B-09).
        "ocr_languages": get_ocr_lang() if TESSERACT_AVAILABLE else None,
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
    log_ocr_status()
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