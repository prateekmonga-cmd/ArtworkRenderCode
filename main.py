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
    KNOWN_CORRECTIONS, is_plausible_artwork_char,
    repair_mojibake, repair_text, is_annotation, is_production_mark,
    looks_like_production_mark, is_dimension_near_miss,
    _normalize_for_ocr_compare,
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


def _segment_vertical_xs(raw_drawings: list, header_threshold_pt: float,
                         min_fold_h: float) -> set:
    """Scan individual path items for vertical segments that are fold lines.

    CorelDRAW typically exports the carton dieline as one compound path whose
    overall rect spans the full carton width (>>3 pt), so the drawing-bbox scan
    in detect_zones() never finds the constituent fold lines.  Inspecting
    drawing["items"] catches them regardless of how they were grouped in the
    source file (B-12).
    """
    xs: set = set()
    for drawing in raw_drawings:
        rect = drawing.get("rect")
        if rect is None:
            continue
        # Skip drawings whose entire bounding box is in the header band
        if rect[3] <= header_threshold_pt:
            continue
        for item in drawing.get("items", []):
            try:
                itype = item[0]
                if itype == "l":          # straight line segment
                    p1, p2 = item[1], item[2]
                    seg_x0 = min(p1.x, p2.x)
                    seg_x1 = max(p1.x, p2.x)
                    seg_y0 = min(p1.y, p2.y)
                    seg_y1 = max(p1.y, p2.y)
                    if (seg_x1 - seg_x0 < 3
                            and seg_y1 - seg_y0 > min_fold_h
                            and seg_y1 > header_threshold_pt):
                        xs.add(round(seg_x0))
                elif itype == "re":       # rectangle sub-item
                    sr = item[1]          # fitz.Rect
                    sw = abs(sr.x1 - sr.x0)
                    sh = abs(sr.y1 - sr.y0)
                    if (sw < 3 and sh > min_fold_h
                            and sr.y1 > header_threshold_pt):
                        xs.add(round(sr.x0))
            except (IndexError, AttributeError, TypeError):
                continue
    return xs


def detect_zones(paths: list, header_threshold_pt: float,
                 raw_drawings: Optional[list] = None) -> list:
    """Detect panel zones from the artwork's vertical fold lines.

    Two-pass approach:
      Pass 1 — drawing bboxes (existing): catches fold lines drawn as individual
               narrow paths (each fold line its own drawing object).
      Pass 2 — drawing items (new): catches fold lines inside a compound path
               whose overall bbox is wide.  CorelDRAW commonly exports all
               dieline elements as one compound path, so pass 1 finds nothing
               and pass 2 is the only route (B-12).

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

    # Pass 1: drawing-bbox scan (narrow whole drawings = individual fold lines)
    vertical_lines: set = set()
    for path in body_paths:
        x0, y0, x1, y1 = path.get("bbox_pt", [0, 0, 0, 0])
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        if width < 3 and height > min_fold_h:
            vertical_lines.add(round(x0))

    # Pass 2: item-level scan — compound path fold lines (B-12)
    if len(vertical_lines) < 2 and raw_drawings:
        vertical_lines |= _segment_vertical_xs(
            raw_drawings, header_threshold_pt, min_fold_h
        )

    vertical_lines_sorted = sorted(vertical_lines)

    zones = []
    if len(vertical_lines_sorted) >= 2:
        # Bound the panels by the artwork, not the sheet, so the reported
        # zone widths are the real panel widths a placement rule can check.
        all_x = sorted(set([round(art_x0)] + vertical_lines_sorted + [round(art_x1)]))
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
                # Header field values are exempt (see is_dimension_near_miss).
                if is_dimension_near_miss(repaired_text, color_hex, in_header=in_header):
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


def extract_paths_enhanced(page: fitz.Page,
                           raw_drawings: Optional[list] = None) -> list:
    """Extract paths with raw pt bbox for zone detection.

    Accepts pre-fetched raw_drawings to avoid a redundant get_drawings() call
    when the caller already holds them (e.g. for segment-level fold detection).
    """
    paths = []
    try:
        drawings = raw_drawings if raw_drawings is not None else page.get_drawings()
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

# ─── Per-document colour profile ─────────────────────────────────
# ANNOTATION_COLORS is a closed vocabulary tuned to one vendor's palette, and a
# closed vocabulary is what caused B-01. Rather than enumerate every designer's
# blue, look at how each colour is actually USED on this page: a colour carrying
# several dimension callouts and no ordinary copy is a callout colour, whatever
# its hex.
#
# This REPORTS the inference; it does not act on it. Auto-filtering on a guess
# would delete artwork copy from a compliance review with no way to tell, and
# the inference has not been validated against a second real artwork yet. Flip
# ANNOTATION_COLOR_INFERENCE_ENFORCED once it has been.
ANNOTATION_COLOR_INFERENCE_MIN_SPANS = 2
ANNOTATION_COLOR_INFERENCE_ENFORCED = False


def profile_span_colors(sections: dict) -> dict:
    """Frequency of every colour on the page, and which unknown colours behave
    like callout colours.

    Returns the histogram (most used first) plus the inferred set, so a new
    vendor's palette surfaces as data instead of silently passing through as
    body text.
    """
    spans = sections.get("body", []) + sections.get("header_table", [])
    counts = {}
    dimension_only = {}

    for s in spans:
        color = (s.get("color_hex") or "").lower()
        if not color:
            continue
        counts[color] = counts.get(color, 0) + 1
        stats = dimension_only.setdefault(color, {"dimension": 0, "other": 0})
        if DIMENSION_PATTERN.match(s.get("text", "").strip()):
            stats["dimension"] += 1
        else:
            stats["other"] += 1

    # Callouts filtered as annotations never reach body/header_table, so count
    # them too — otherwise a correctly-detected colour looks unused here.
    for a in sections.get("annotations", []):
        color = (a.get("color_hex") or "").lower()
        if color:
            counts[color] = counts.get(color, 0) + 1

    inferred = sorted(
        color for color, stats in dimension_only.items()
        if stats["dimension"] >= ANNOTATION_COLOR_INFERENCE_MIN_SPANS
        and stats["other"] == 0
        and not is_annotation_color(color)
    )

    for color in inferred:
        logger.warning(
            "Colour %s carries %d dimension spans and no body copy — it looks "
            "like a callout colour missing from ANNOTATION_COLORS",
            color, dimension_only[color]["dimension"],
        )

    return {
        "histogram": [{"color_hex": c, "spans": n}
                      for c, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))],
        "unknown_colors": sorted(c for c in counts if not is_annotation_color(c)),
        "inferred_annotation_colors": inferred,
        "inference_enforced": ANNOTATION_COLOR_INFERENCE_ENFORCED,
    }


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
    # Cartons label this "Size in mm (LxWxH)", labels and inserts just
    # "Size (LxW)" -- which matched nothing, so the declared footprint was
    # unreadable on every component except the carton. Spelled out rather than
    # adding a bare "size", which would also swallow "artwork size" and
    # "flat size" (different canonicals, matched by substring).
    "size lxw": "dimension",
    "size lxwxh": "dimension",
}
HEADER_MIN_LABELS = 2          # one stray phrase is not a header
HEADER_FALLBACK_RATIO = 0.20   # top-of-page guess when content detection fails
HEADER_PAD_PT = 6              # tolerance below the lowest header row
HEADER_ROW_PITCH_TOLERANCE = 1.8   # a row this far past the normal pitch ends the table
HEADER_MAX_EXTENT_RATIO = 0.45     # never let the band swallow half the sheet


def _median(values: list) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _grow_header_band(rows: list, seed_idx: list, height_pt: float) -> list:
    """Extend a seed of matched label rows across the rest of the table.

    The header is a table, and only some of its labels are in the vocabulary.
    Stopping at the lowest *recognised* label left rows like "Size (W) | 60 mm"
    below the split, in the body, where their dimension values were then logged
    as annotation near-misses and forced needs_review on a clean page.

    Rows are absorbed while they keep the table's row pitch and overlap it
    horizontally; the first big vertical gap is the header/artwork boundary.
    """
    first, last = min(seed_idx), max(seed_idx)

    # Row pitch measured from the matched rows themselves, so a dense header
    # and an airy one each get their own gate rather than a fixed constant.
    seed_tops = [rows[i]["bbox"][1] for i in seed_idx]
    pitches = [b - a for a, b in zip(seed_tops, seed_tops[1:]) if b > a]
    if pitches:
        pitch = _median(pitches)
    else:
        heights = [rows[i]["bbox"][3] - rows[i]["bbox"][1] for i in seed_idx]
        pitch = max(_median(heights) * 2, 1.0)
    max_gap = pitch * HEADER_ROW_PITCH_TOLERANCE

    def overlaps(idx, lo, hi):
        b = rows[idx]["bbox"]
        return b[0] <= hi and b[2] >= lo

    band_x0 = min(rows[i]["bbox"][0] for i in seed_idx)
    band_x1 = max(rows[i]["bbox"][2] for i in seed_idx)
    ceiling = height_pt * HEADER_MAX_EXTENT_RATIO

    # Downward: the common case — unrecognised field rows under the last label.
    while last + 1 < len(rows):
        nxt = rows[last + 1]
        if nxt["bbox"][1] - rows[last]["bbox"][1] > max_gap:
            break
        if not overlaps(last + 1, band_x0, band_x1):
            break
        if nxt["bbox"][3] > ceiling:
            break
        last += 1
        band_x0 = min(band_x0, nxt["bbox"][0])
        band_x1 = max(band_x1, nxt["bbox"][2])

    # Upward: a title or code row sitting above the first recognised label.
    while first - 1 >= 0:
        prev = rows[first - 1]
        if rows[first]["bbox"][1] - prev["bbox"][1] > max_gap:
            break
        if not overlaps(first - 1, band_x0, band_x1):
            break
        first -= 1
        band_x0 = min(band_x0, prev["bbox"][0])
        band_x1 = max(band_x1, prev["bbox"][2])

    return list(range(first, last + 1))


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

    # Sorted top-to-bottom so the band can grow over adjacent rows.
    rows = sorted(lines.values(), key=lambda e: (e["bbox"][1], e["bbox"][0]))

    seed_idx = []
    found_labels = set()
    for i, entry in enumerate(rows):
        key = _label_key(" ".join(entry["words"]))
        hits = {canon for alias, canon in HEADER_FIELD_LABELS.items() if alias in key}
        if hits:
            found_labels |= hits
            seed_idx.append(i)

    if len(found_labels) < HEADER_MIN_LABELS or not seed_idx:
        return {
            "threshold_pt": fallback,
            "detected_by_content": False,
            "labels_found": sorted(found_labels),
            "rows_in_band": 0, "unlabelled_rows_absorbed": 0,
            "bbox_mm": None, "width_mm": None, "height_mm": None,
            "vertical_position": None, "horizontal_position": None,
        }

    band_idx = _grow_header_band(rows, seed_idx, height_pt)
    band = [rows[i]["bbox"] for i in band_idx]

    x0 = min(b[0] for b in band)
    y0 = min(b[1] for b in band)
    x1 = max(b[2] for b in band)
    y1 = max(b[3] for b in band)

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
        # How much of the band came from vocabulary hits vs. row-pitch growth.
        # A high absorbed count means this artwork's field labels are mostly
        # unknown to HEADER_FIELD_LABELS — worth a look before the vocabulary
        # drifts further out of date.
        "rows_in_band": len(band_idx),
        "unlabelled_rows_absorbed": len(band_idx) - len(seed_idx),
        "bbox_mm": bbox_to_mm((x0, y0, x1, y1)),
        "width_mm": pt_to_mm(x1 - x0),
        "height_mm": pt_to_mm(y1 - y0),
        "vertical_position": vertical,
        "horizontal_position": horizontal,
    }


# The SOP writes the artwork code as one string, ACnnnV.n, but the tabular
# header stores it as two separate fields. Anything that needs the whole value
# had to join them itself, and did so inconsistently run to run ("AC2499V.10"
# one time, "2499, 10" another), so a self-comparison failed on formatting
# rather than on content. Synthesised once here instead; downstream reads
# ac_reference and never rebuilds it.
AC_COMPOSITE_RE = re.compile(r"\bAC\s*(\d+)\s*V\.?\s*(\d+)\b", re.IGNORECASE)

# Row banding: spans whose vertical centres sit within this many points belong
# to the same header row. Header rows in real artwork run ~14-20 pt apart, and
# a value can be a couple of points off its label's baseline when font sizes
# differ within the row.
AC_ROW_BAND_PT = 6.0

# Shapes the two header values must take. Passed into the lookup so a prose
# neighbour is never selected in the first place.
AC_NUMBER_SHAPE = re.compile(r"\d+")
VERSION_SHAPE = re.compile(r"[\d.]+")


def _span_center_y(span: dict) -> float:
    bbox = span.get("bbox") or [0, 0, 0, 0]
    return (bbox[1] + bbox[3]) / 2


def _value_for_label(header_table: list, canonical: str,
                     shape: Optional[Any] = None) -> Optional[str]:
    """Find the value span sitting to the right of a header label.

    Matched by position, not list order. header_table is appended in raw
    PyMuPDF span order (unlike body, it is never sorted), so "the next item in
    the list" is not reliably the value. The header is also two label/value
    pairs wide -- "Product Name | ... | AC Reference | 2499" -- so the match has
    to be the NEAREST span to the right within the same row, not the last one.

    A label can occur more than once on a page. When the header band over-grows
    and absorbs body copy, an insert's own revision footer ("Version: AEMPS-
    Medaxone 1 g-V2-mar2025") lands in header_table alongside the real header
    field. Candidates are therefore resolved per label occurrence, topmost
    first, rather than by taking the smallest gap across all occurrences --
    that let the footer's neighbour, 18 pt away, outbid the genuine value
    89 pt away and null the whole field.

    shape is an optional compiled pattern the value must fullmatch. Applying it
    during selection rather than afterwards means a prose neighbour is skipped
    in favour of the real value, instead of winning and then being discarded.
    """
    labels = [alias for alias, canon in HEADER_FIELD_LABELS.items()
              if canon == canonical]

    def accepts(text: str) -> bool:
        return shape is None or shape.fullmatch(text.strip()) is not None

    # Header fields live in the header band at the top of the page; a body line
    # repeating a field label always sits below it. Ties go to the higher span.
    matches = [span for span in header_table
               if any(alias in _label_key(span.get("text", ""))
                      for alias in labels)]
    matches.sort(key=lambda s: (s.get("bbox") or [0, 0, 0, 0])[1])

    for span in matches:
        key = _label_key(span.get("text", ""))

        # A merged cell can hold both label and value ("Version# 10"); prefer
        # that reading when the label span itself carries trailing digits.
        inline = re.search(r"(\d[\d.\-/]*)\s*$", span.get("text", "").strip())
        if (inline and _label_key(inline.group(1)) != key
                and accepts(inline.group(1))):
            return inline.group(1)

        label_bbox = span.get("bbox") or [0, 0, 0, 0]
        label_mid = _span_center_y(span)

        best_value = None
        best_gap = None
        for other in header_table:
            if other is span:
                continue
            other_bbox = other.get("bbox") or [0, 0, 0, 0]
            if abs(_span_center_y(other) - label_mid) > AC_ROW_BAND_PT:
                continue
            gap = other_bbox[0] - label_bbox[2]
            if gap < 0:                      # left of the label, not its value
                continue
            text = (other.get("text") or "").strip()
            if not text:
                continue
            # Skip the next label in the same row (e.g. "AC Reference" sitting
            # to the right of "Product Name") -- a label is never a value.
            other_key = _label_key(text)
            if any(alias in other_key for alias in HEADER_FIELD_LABELS):
                continue
            if not accepts(text):
                continue
            if best_gap is None or gap < best_gap:
                best_gap, best_value = gap, text

        if best_value is not None:
            return best_value

    return None


# How close to an edge the on-pack instance must sit to count as an edge
# placement. The SOP requires this string on the right edge reading bottom-to-
# top, so the rule needs the side, not just the coordinates.
AC_EDGE_MARGIN_MM = 15.0


def _artboard_bounds_mm(all_spans: list, page_w_mm: Optional[float],
                        page_h_mm: Optional[float]) -> tuple:
    """Extent of the drawn artboard, not the sheet it sits on.

    The SOP's "right edge of the artboard" is the artwork's own edge. Artwork
    is placed on an oversized sheet -- the insert measures 148x250 mm on a
    250x350 mm page -- so measuring from the page put the rotated edge code
    51 mm inside the "right" margin and classified it as interior.
    """
    xs = [s["bbox_mm"][2] for s in all_spans if s.get("bbox_mm")]
    x0s = [s["bbox_mm"][0] for s in all_spans if s.get("bbox_mm")]
    ys = [s["bbox_mm"][3] for s in all_spans if s.get("bbox_mm")]
    y0s = [s["bbox_mm"][1] for s in all_spans if s.get("bbox_mm")]
    if not xs:
        return (0.0, 0.0, page_w_mm or 0.0, page_h_mm or 0.0)
    return (min(x0s), min(y0s), max(xs), max(ys))


def _edge_position(bbox_mm: list, bounds: tuple) -> Optional[str]:
    """Which artboard edge a span sits against, or "interior"."""
    if not bbox_mm or len(bbox_mm) < 4:
        return None
    x0, y0, x1, y1 = bbox_mm[:4]
    bx0, by0, bx1, by1 = bounds
    if bx1 - x1 <= AC_EDGE_MARGIN_MM:
        return "right"
    if x0 - bx0 <= AC_EDGE_MARGIN_MM:
        return "left"
    if y0 - by0 <= AC_EDGE_MARGIN_MM:
        return "top"
    if by1 - y1 <= AC_EDGE_MARGIN_MM:
        return "bottom"
    return "interior"


def synthesize_ac_reference(header_table: list, body_text_all: str,
                            all_spans: Optional[list] = None,
                            page_w_mm: Optional[float] = None,
                            page_h_mm: Optional[float] = None) -> dict:
    """Combine the header's split AC Reference / Version fields into the single
    canonical ACnnnV.n string, and report whether the same value appears
    on-pack (rotated on a flap or edge, per the SOP), including where.

    consistent is None -- not False -- when either side is missing: absence of
    an on-pack instance is a different finding from a genuine mismatch, and
    collapsing them would make a missing value read as a contradiction.

    all_spans is every span on the page, whatever section it was filed under.
    Searching body alone made this field depend on how the header band happened
    to be drawn: on insert pages the band over-grew and absorbed the rotated
    edge instance into header_table, so on_pack_instance came back null on
    pages 3 and 4 of Art-CommercialPDF-12168 while pages 1 and 2 resolved it.
    The code is printed on the pack either way; which bucket the band sorted it
    into is an extraction detail and must not change the finding.

    The matched span's coordinates and rotation are reported alongside, so a
    rule checking placement ("right edge, 90 CCW") reads a measured field
    instead of re-scanning spans it may not find.
    """
    ac_number = _value_for_label(header_table, "ac reference", AC_NUMBER_SHAPE)
    version_number = _value_for_label(header_table, "version", VERSION_SHAPE)

    header_combined = None
    if ac_number and version_number:
        header_combined = f"AC{ac_number.strip()}V.{version_number.strip()}"

    # Span-level first, so position and rotation come with the match. Header
    # label/value cells are skipped -- the header's own declaration is not an
    # on-pack instance of it, or every page would trivially "match itself".
    on_pack_instance = None
    on_pack_position = None
    bounds = _artboard_bounds_mm(all_spans or [], page_w_mm, page_h_mm)
    label_ids = {id(s) for s in (header_table or [])
                 if _label_key(s.get("text", "")) in HEADER_FIELD_LABELS}
    for span in (all_spans or []):
        if id(span) in label_ids:
            continue
        match = AC_COMPOSITE_RE.search(span.get("text") or "")
        if not match:
            continue
        bbox_mm = span.get("bbox_mm")
        on_pack_instance = match.group(0).strip()
        on_pack_position = {
            "bbox_mm": bbox_mm,
            "rotation_deg": span.get("rotation_deg"),
            "rotation_sop": span.get("rotation_sop"),
            "edge": _edge_position(bbox_mm, bounds),
            "artboard_bounds_mm": [round(v, 2) for v in bounds],
            "font_size_pt": span.get("font_size_pt"),
        }
        break

    # Fall back to the concatenated text: the composite can be split across
    # spans, which the per-span scan above cannot see. No position then.
    if on_pack_instance is None:
        text_match = AC_COMPOSITE_RE.search(body_text_all or "")
        on_pack_instance = text_match.group(0).strip() if text_match else None

    consistent = None
    if header_combined and on_pack_instance:
        # Compared on digits alone: the on-pack instance is set in tiny rotated
        # type where spacing round the "V." is unreliable, and a whitespace
        # difference is not a content difference.
        normalize = lambda s: re.sub(r"[^0-9a-z]", "", s.lower())
        consistent = normalize(header_combined) == normalize(on_pack_instance)

    return {
        "header_combined": header_combined,
        "ac_number": ac_number.strip() if ac_number else None,
        "version": version_number.strip() if version_number else None,
        "on_pack_instance": on_pack_instance,
        "on_pack_position": on_pack_position,
        "consistent": consistent,
    }


# ═══════════════════════════════════════════════════════════════
# DIMENSION CROSS-CHECK
# ═══════════════════════════════════════════════════════════════
#
# Three independent sources state the component's size and nothing compared
# them: the header's declared field ("38x38x67 mm"), the printed callouts
# ("38.00 mm"), and the drawn geometry. The callouts are read as text, not
# measured, so on their own they verify nothing -- a drawing that says 38 mm
# while measuring 42 mm read as perfectly consistent. Cross-checked here once,
# for the same reason ac_reference is: so no consumer re-derives it and gets a
# different answer each run.

DIMENSION_TOLERANCE_MM = 0.5      # print tolerance; observed error is ~0.02 mm
SHEET_TOLERANCE_MM = 1.0          # a rect this close to the page IS the sheet

# Declared footprints are written "38x38x67 mm", "70x30mm", "148x250mm".
DIMENSION_VALUE_SHAPE = re.compile(
    r"[\d.]+(?:\s*[x×]\s*[\d.]+){1,2}\s*(?:mm)?", re.IGNORECASE)
DIMENSION_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _dimension_numbers(text: Optional[str]) -> list:
    return [float(n) for n in DIMENSION_NUMBER_RE.findall(text or "")]


def _component_rects(paths: list, page_w_mm: float, page_h_mm: float) -> list:
    """Drawn rectangles that could be a component outline.

    The sheet rect is dropped -- every page carries one and it would match a
    declared size only by coincidence. Everything else is kept: matching is
    done on whole rects, not on edge lengths, because edges are not
    discriminating (page 1 has 37 distinct edge lengths, so a declared value
    finds a match by chance; it has only a handful of distinct large rects).
    """
    rects = []
    for path in paths:
        bbox = path.get("bbox_mm")
        if not bbox:
            continue
        width = round(bbox[2] - bbox[0], 2)
        height = round(bbox[3] - bbox[1], 2)
        if width <= 0 or height <= 0:
            continue
        if (abs(width - page_w_mm) <= SHEET_TOLERANCE_MM
                and abs(height - page_h_mm) <= SHEET_TOLERANCE_MM):
            continue
        rects.append((width, height))
    return rects


def _match_rect(rects: list, side_a: float, side_b: float) -> Optional[dict]:
    """Closest drawn rect to a declared side pair, in either orientation."""
    best = None
    for width, height in rects:
        for candidate in ((width, height), (height, width)):
            delta = max(abs(candidate[0] - side_a), abs(candidate[1] - side_b))
            if delta <= DIMENSION_TOLERANCE_MM and (best is None
                                                    or delta < best["delta_mm"]):
                best = {"measured_mm": [width, height],
                        "delta_mm": round(delta, 3)}
    return best


def synthesize_dimension_check(header_table: list, annotations: list,
                               near_misses: list, paths: list,
                               page_w_mm: float, page_h_mm: float) -> dict:
    """Reconcile declared size, printed callouts, and drawn geometry.

    consistent is None -- not False -- when a source is missing, matching
    ac_reference's contract: an unmeasurable page is not a failing page.
    """
    declared_raw = _value_for_label(header_table, "dimension",
                                    DIMENSION_VALUE_SHAPE)
    declared = _dimension_numbers(declared_raw)

    # Callouts normally arrive as classified annotations. On an unknown vendor
    # palette they land in near-misses instead; read those rather than let the
    # whole check go quiet on a colour we simply haven't seen before.
    callout_source = "annotations"
    spans = [a for a in annotations
             if DIMENSION_PATTERN.match((a.get("text") or "").strip())]
    if not spans:
        spans = [a for a in near_misses
                 if DIMENSION_PATTERN.match((a.get("text") or "").strip())]
        callout_source = "annotation_near_misses" if spans else None
    callouts = sorted({v for s in spans
                       for v in _dimension_numbers(s.get("text"))})

    rects = _component_rects(paths, page_w_mm, page_h_mm)

    # Every distinct pair of declared sides should be drawn somewhere: a
    # 38x38x67 carton shows a 38x67 face and a 38x38 face.
    pairs = sorted({(min(a, b), max(a, b))
                    for i, a in enumerate(declared)
                    for b in declared[i + 1:]})
    pair_results = []
    for side_a, side_b in pairs:
        match = _match_rect(rects, side_a, side_b)
        pair_results.append({
            "declared_mm": [side_a, side_b],
            "matched": match is not None,
            "measured_mm": match["measured_mm"] if match else None,
            "delta_mm": match["delta_mm"] if match else None,
        })

    measured_ok = None
    if pairs and rects:
        measured_ok = all(p["matched"] for p in pair_results)

    unmatched_callouts = [c for c in callouts
                          if not any(abs(c - d) <= DIMENSION_TOLERANCE_MM
                                     for d in declared)]
    callouts_ok = None
    if declared and callouts:
        callouts_ok = not unmatched_callouts

    consistent = None
    if measured_ok is not None or callouts_ok is not None:
        consistent = all(flag for flag in (measured_ok, callouts_ok)
                         if flag is not None)

    return {
        "declared_raw": declared_raw,
        "declared_mm": declared or None,
        "callouts_mm": callouts or None,
        "callout_source": callout_source,
        "measured_pairs": pair_results,
        "declared_vs_measured": measured_ok,
        "declared_vs_callouts": callouts_ok,
        "unmatched_callouts_mm": unmatched_callouts or None,
        "tolerance_mm": DIMENSION_TOLERANCE_MM,
        "consistent": consistent,
    }


# ═══════════════════════════════════════════════════════════════
# HEADER TABLE RECONSTRUCTION (position-based)
# ═══════════════════════════════════════════════════════════════
#
# Reconstructs the tabular header purely from drawing geometry -- no LLM
# reading involved. Ported from the n8n "Reconstruct Header Table" Code node
# (same algorithm, same field names) so it runs at extraction time instead of
# as a separate downstream step, and so nothing reading its output has to
# change shape.
#
# Two drawing conventions exist across artworks for the same visual table:
# some pages draw one filled/stroked rectangle per cell (Outer Carton), others
# draw a bare grid of horizontal/vertical line segments with no per-cell rects
# at all (Foil). Column boundaries are reliable in both styles; row boundaries
# are not (a grid can merge two logically separate rows, e.g. "AC Reference"
# and "Version#", into one region with no line between them). So drawn rows
# are never used for grouping -- only for a best-effort alignment bbox; row
# NUMBERS reported to the caller instead come from clustering the labels' own
# y-centers.
#
# Algorithm:
#   1. Derive column x-boundaries from whichever drawing style is present
#   2. Classify EVERY header span as a label or not, INDIVIDUALLY -- this is
#      what keeps two different labels correctly separate even when they
#      share a coarse drawn region
#   3. For each label, bound its value's vertical window using the MIDPOINT
#      between its own y-center and the next label's y-center in the same
#      column (not either label's y0 -- a multi-line value starts higher than
#      its single-line label to stay vertically centered against it, so a
#      y0-anchored boundary lets the previous row swallow that line)
#   4. Collect every non-label span in the value column whose y-center falls
#      inside that window -- this is the value, handling multi-line wraps
#      naturally without any fixed gap threshold
#   5. Assign a human-readable row number (1, 2, 3...) and column-pair number
#      (1, 2...) to every field, so "row 3, col-pair 2" points at the same
#      cell a reviewer would point at on the printed artwork
#   6. Report alignment (left/center/right, top/middle/bottom) from the
#      text's own bbox against its known column width
#   7. Attach a color swatch (fill_color_hex) if one is drawn inside the
#      value's region (e.g. the Pantone No. rows)
#   8. NEVER collapse mixed formatting within a value into a single value --
#      font_name/font_size_pt/is_bold/is_italic/color_hex are reported ONLY
#      when every span agrees; otherwise null, with mixed_formatting true and
#      the full per-span breakdown always in spans[]
#   9. Report the header table's own overall dimensions plus each border's
#      distance from the page's own top-left origin, explicitly named and in
#      both pt and mm
#  10. Every returned block is self-identifying (page_name, page_number)

RECON_FIELD_LABELS = {
    "product name": "product_name",
    "ac reference": "ac_reference",
    "ac ref": "ac_reference",
    "component type": "component_type",
    "substrate": "substrate",
    "pantone no": "pantone_no",
    "artwork size": "artwork_size",
    "flat size": "flat_size",
    "print side": "print_side",
    "printing side": "print_side",
    "colour code": "colour_code",
    "color code": "colour_code",
    "market": "market",
    "version": "version",
    "dimension": "dimension",
    "size in mm": "dimension",
    "size lxw": "dimension",
    "size lxwxh": "dimension",
    "size (w)": "dimension",
    "product code": "product_code",
    "style": "style",
    "barcode": "barcode",
    "printing overlay": "printing_overlay",
    "repeat": "repeat",
    "printing zone": "printing_zone",
    "dia of tube": "dia_of_tube",
}


def _recon_label_key(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[.#:]", "", (text or "").lower())).strip()


def _recon_canonical_for(text: Optional[str]) -> Optional[str]:
    key = _recon_label_key(text)
    if key in RECON_FIELD_LABELS:
        return RECON_FIELD_LABELS[key]
    for alias, canon in RECON_FIELD_LABELS.items():
        if key == alias or key.startswith(alias + " ") or key.endswith(" " + alias):
            return canon
    for alias, canon in RECON_FIELD_LABELS.items():
        if alias in key:
            return canon
    return None


def _recon_area_of(b: list) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _recon_column_bounds(paths: list) -> Optional[list]:
    """Column boundaries from real filled cell rects (Outer Carton) or a grid
    of vertical line segments (Foil) -- whichever drawing style is present."""
    rects = [p for p in (paths or [])
             if p.get("bbox") and not p.get("fill_color_hex") and _recon_area_of(p["bbox"]) > 0]
    v_lines = [p for p in (paths or [])
               if p.get("bbox") and (p["bbox"][2] - p["bbox"][0]) < 1 and (p["bbox"][3] - p["bbox"][1]) > 5]

    xs = set()
    if len(rects) > 1:
        max_area = max(_recon_area_of(r["bbox"]) for r in rects)
        cell_rects = [r for r in rects if _recon_area_of(r["bbox"]) < max_area * 0.9]
        for r in (cell_rects if cell_rects else rects):
            xs.add(round(r["bbox"][0]))
            xs.add(round(r["bbox"][2]))
    elif len(v_lines) >= 2:
        for l in v_lines:
            xs.add(round(l["bbox"][0]))

    sorted_xs = sorted(xs)
    if len(sorted_xs) < 2:
        return None  # no usable grid -- caller falls back
    return [[sorted_xs[i], sorted_xs[i + 1]] for i in range(len(sorted_xs) - 1)]


def _recon_border_stroke_values(paths: list) -> list:
    """Distinct border/stroke weights of the header's own cell grid, excluding
    the single outer border rect (its stroke can legitimately differ)."""
    with_stroke = [p for p in (paths or [])
                   if p.get("bbox") and isinstance(p.get("stroke_width_pt"), (int, float))
                   and not p.get("fill_color_hex")]
    if not with_stroke:
        return []
    max_area = max(_recon_area_of(r["bbox"]) for r in with_stroke)
    inner = [r for r in with_stroke if _recon_area_of(r["bbox"]) < max_area * 0.9]
    pool = inner if inner else with_stroke
    return sorted({r["stroke_width_pt"] for r in pool})


def _recon_swatch_for(bbox: list, swatches: list) -> Optional[dict]:
    for s in swatches:
        sb = s["bbox"]
        if (sb[0] >= bbox[0] - 2 and sb[1] >= bbox[1] - 2
                and sb[2] <= bbox[2] + 2 and sb[3] <= bbox[3] + 2):
            return {"fill_color_hex": s["fill_color_hex"], "bbox": sb}
    return None


def _recon_span_formatting(span_details: list) -> dict:
    def all_agree(key):
        return all(s[key] == span_details[0][key] for s in span_details)

    uniform = {k: all_agree(k) for k in ("font_name", "font_size_pt", "is_bold", "is_italic", "color_hex")}
    mixed = not all(uniform.values())
    first = span_details[0]
    return {
        "font_name": first["font_name"] if uniform["font_name"] else None,
        "font_size_pt": first["font_size_pt"] if uniform["font_size_pt"] else None,
        "is_bold": first["is_bold"] if uniform["is_bold"] else None,
        "is_italic": first["is_italic"] if uniform["is_italic"] else None,
        "color_hex": first["color_hex"] if uniform["color_hex"] else None,
        "mixed_formatting": mixed,
    }


def _recon_describe_group(group_spans: list, col_bounds: Optional[list], swatches: list) -> dict:
    sorted_spans = sorted(group_spans, key=lambda s: (s["bbox"][1], s["bbox"][0]))
    lines = [s["text"].strip() for s in sorted_spans]
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    text_bbox = [
        min(s["bbox"][0] for s in sorted_spans), min(s["bbox"][1] for s in sorted_spans),
        max(s["bbox"][2] for s in sorted_spans), max(s["bbox"][3] for s in sorted_spans),
    ]
    # Best-effort cell bbox: known column width x this group's own text extent
    # (padded a touch). Row height isn't reliably known in the line-grid case,
    # so vertical alignment is reported against this rather than a true row --
    # horizontal alignment (the more commonly meaningful check) is unaffected.
    pad = 3
    cell_bbox = ([col_bounds[0], text_bbox[1] - pad, col_bounds[1], text_bbox[3] + pad]
                 if col_bounds else text_bbox)

    left_gap, right_gap = text_bbox[0] - cell_bbox[0], cell_bbox[2] - text_bbox[2]
    top_gap, bottom_gap = text_bbox[1] - cell_bbox[1], cell_bbox[3] - text_bbox[3]
    if abs(left_gap - right_gap) <= 3:
        align_h = "center"
    elif right_gap < left_gap:
        align_h = "right"
    else:
        align_h = "left"
    if abs(top_gap - bottom_gap) <= 3:
        align_v = "middle"
    elif bottom_gap < top_gap:
        align_v = "bottom"
    else:
        align_v = "top"

    span_details = [{
        "text": s["text"], "bbox": s["bbox"], "font_name": s.get("font_name"),
        "font_size_pt": s.get("font_size_pt"), "is_bold": s.get("is_bold"),
        "is_italic": s.get("is_italic"), "color_hex": s.get("color_hex"),
    } for s in sorted_spans]

    return {
        "text": text, "lines": lines,
        "cell_bbox": cell_bbox, "text_bbox": text_bbox,
        **_recon_span_formatting(span_details),
        "spans": span_details,
        "align_h": align_h, "align_v": align_v,
        "swatch": _recon_swatch_for(cell_bbox, swatches),
    }


def _recon_col_index_for(x: float, col_bounds: list) -> int:
    for i, (c0, c1) in enumerate(col_bounds):
        if c0 - 2 <= x < c1 + 2:
            return i
    return -1


def build_header_reconstruction(header_spans: list, paths: list, page_meta: dict) -> dict:
    page_meta = page_meta or {}
    spans = [s for s in (header_spans or []) if s.get("bbox") and (s.get("text") or "").strip()]
    if not spans:
        return {"page_name": page_meta.get("page_name"), "page_number": page_meta.get("page_number"),
                "pairs": [], "unassigned_spans": []}

    # Scope paths to the header's own vertical extent (derived from the header
    # spans themselves, which the extractor already separated correctly) --
    # otherwise column detection picks up unrelated rects/lines from elsewhere
    # on the page and produces garbage boundaries.
    header_max_y = max(s["bbox"][3] for s in spans)
    all_paths = [p for p in (paths or []) if p.get("bbox") and p["bbox"][1] < header_max_y + 10]
    swatches = [p for p in all_paths if p.get("fill_color_hex")]
    col_bounds = _recon_column_bounds(all_paths)

    # Classify every span independently -- this is what stops two genuinely
    # different labels (e.g. AC Reference / Version#) from being merged just
    # because they happen to share a coarse drawn region.
    labeled, plain = [], []
    for span in spans:
        canon = _recon_canonical_for(span["text"])
        col = _recon_col_index_for((span["bbox"][0] + span["bbox"][2]) / 2, col_bounds) if col_bounds else -1
        if canon:
            labeled.append({"span": span, "canon": canon, "col": col})
        else:
            plain.append({"span": span, "col": col})

    by_label_col = {}
    for l in labeled:
        by_label_col.setdefault(l["col"], []).append(l)
    for group in by_label_col.values():
        group.sort(key=lambda l: l["span"]["bbox"][1])

    def y_center(span):
        return (span["bbox"][1] + span["bbox"][3]) / 2

    def window_for(l):
        siblings = by_label_col.get(l["col"], [l])
        idx = siblings.index(l)
        prev = siblings[idx - 1] if idx - 1 >= 0 else None
        nxt = siblings[idx + 1] if idx + 1 < len(siblings) else None
        window_start = (y_center(prev["span"]) + y_center(l["span"])) / 2 if prev else float("-inf")
        window_end = (y_center(l["span"]) + y_center(nxt["span"])) / 2 if nxt else float("inf")
        return window_start, window_end

    # Row number (1, 2, 3...) and column-pair number (1, 2...) for each field,
    # so a reviewer can point at "row 1, col-pair 2" the same way they'd point
    # at the printed table, without having to read raw pt coordinates.
    row_tolerance_pt = 8
    row_centers = sorted(y_center(l["span"]) for l in labeled)
    row_clusters = []
    for c in row_centers:
        if not row_clusters or c - row_clusters[-1] > row_tolerance_pt:
            row_clusters.append(c)

    def row_number_for(span):
        c = y_center(span)
        best, best_dist = 0, float("inf")
        for i, rc in enumerate(row_clusters):
            d = abs(rc - c)
            if d < best_dist:
                best_dist, best = d, i
        return best + 1

    # "Row count" for compliance purposes (e.g. a spec requiring exactly N
    # rows) is counted from the LEFT-MOST column only, not every label on the
    # page. A right-hand column can split into an extra sub-row that has no
    # corresponding split on the left -- that is one row as printed, not two,
    # and counting every label globally conflated the two, overcounting by
    # exactly the number of such splits.
    left_col_centers = sorted(y_center(l["span"]) for l in labeled if l["col"] == 0)
    left_col_clusters = []
    for c in left_col_centers:
        if not left_col_clusters or c - left_col_clusters[-1] > row_tolerance_pt:
            left_col_clusters.append(c)
    row_count = len(left_col_clusters) or len(row_clusters)

    pairs = []
    claimed_value_spans = set()
    for l in labeled:
        window_start, window_end = window_for(l)
        value_col = l["col"] + 1
        value_spans = sorted(
            (p["span"] for p in plain
             if p["col"] == value_col and window_start <= y_center(p["span"]) < window_end),
            key=lambda s: (s["bbox"][1], s["bbox"][0]),
        )
        for vs in value_spans:
            claimed_value_spans.add(id(vs))

        label_col = col_bounds[l["col"]] if col_bounds and l["col"] >= 0 else None
        val_col = (col_bounds[value_col] if col_bounds and l["col"] >= 0 and value_col < len(col_bounds)
                   else None)

        pairs.append({
            "row": row_number_for(l["span"]),
            "col_pair": (l["col"] // 2) + 1 if l["col"] >= 0 else None,
            "canonical": l["canon"],
            "label": _recon_describe_group([l["span"]], label_col, swatches),
            "value": _recon_describe_group(value_spans, val_col, swatches) if value_spans else None,
        })
    pairs.sort(key=lambda p: (p["row"], p["col_pair"] or 0))

    unassigned = [p["span"]["text"] for p in plain if id(p["span"]) not in claimed_value_spans]

    # Overall table extent -- the header table's own dimensions on the page,
    # not any individual cell's -- plus each border's distance from the page's
    # own top-left origin, explicitly named for compliance checks:
    #   left_from_page_left_pt/mm    -- table's left border
    #   right_from_page_left_pt/mm   -- table's right border (also measured
    #                                    from the page's LEFT edge)
    #   top_from_page_top_pt/mm      -- table's top border
    #   bottom_from_page_top_pt/mm   -- table's bottom border (also measured
    #                                    from the page's TOP edge)
    all_x0 = [s["bbox"][0] for s in spans]
    all_x1 = [s["bbox"][2] for s in spans]
    all_y0 = [s["bbox"][1] for s in spans]
    all_y1 = [s["bbox"][3] for s in spans]
    table_bbox = [min(all_x0), min(all_y0), max(all_x1), max(all_y1)]

    has_mm = all(isinstance(s.get("bbox_mm"), list) for s in spans)
    table_bbox_mm = None
    if has_mm:
        mm_x0 = [s["bbox_mm"][0] for s in spans]
        mm_x1 = [s["bbox_mm"][2] for s in spans]
        mm_y0 = [s["bbox_mm"][1] for s in spans]
        mm_y1 = [s["bbox_mm"][3] for s in spans]
        table_bbox_mm = [min(mm_x0), min(mm_y0), max(mm_x1), max(mm_y1)]

    return {
        "page_name": page_meta.get("page_name"),
        "page_number": page_meta.get("page_number"),
        "pairs": pairs,
        "unassigned_spans": unassigned,
        "table_bbox": table_bbox,
        "table_bbox_mm": table_bbox_mm,
        "table_width_pt": table_bbox[2] - table_bbox[0],
        "table_height_pt": table_bbox[3] - table_bbox[1],
        "table_width_mm": (table_bbox_mm[2] - table_bbox_mm[0]) if table_bbox_mm else None,
        "table_height_mm": (table_bbox_mm[3] - table_bbox_mm[1]) if table_bbox_mm else None,
        "left_from_page_left_pt": table_bbox[0],
        "right_from_page_left_pt": table_bbox[2],
        "top_from_page_top_pt": table_bbox[1],
        "bottom_from_page_top_pt": table_bbox[3],
        "left_from_page_left_mm": table_bbox_mm[0] if table_bbox_mm else None,
        "right_from_page_left_mm": table_bbox_mm[2] if table_bbox_mm else None,
        "top_from_page_top_mm": table_bbox_mm[1] if table_bbox_mm else None,
        "bottom_from_page_top_mm": table_bbox_mm[3] if table_bbox_mm else None,
        "row_count": row_count,
        "column_count": len(col_bounds) if col_bounds else None,
        "col_pair_count": math.ceil(len(col_bounds) / 2) if col_bounds else None,
        "border_stroke_pt": _recon_border_stroke_values(all_paths),
        # Passed through for downstream compliance checks that need the
        # page's own geometry (e.g. horizontal-centering) alongside the
        # header's own -- not re-derived here since it's already computed.
        "page_width_mm": page_meta.get("page_width_mm"),
        "page_height_mm": page_meta.get("page_height_mm"),
        "artboard_bounds_mm": page_meta.get("artboard_bounds_mm"),
        # Passed through as-is for rule 1-43: header_combined (format validity
        # -- only set when both the AC number and version resolve to their
        # required shape) and consistent (does the on-pack rotated instance
        # match the header-derived value) are already computed by
        # synthesize_ac_reference; no reason to re-derive them here.
        "ac_reference_computed": page_meta.get("ac_reference_computed"),
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

    # 1. Paths first (zone detection needs them). Fetch drawings once so both
    # extract_paths_enhanced and detect_zones share the same call result —
    # get_drawings() can be expensive on complex dieline art.
    raw_drawings = page.get_drawings()
    paths = extract_paths_enhanced(page, raw_drawings=raw_drawings)

    # 2. Zone detection — pass raw drawings so the item-level fold-line scan
    # (B-12) can recover panels from compound paths where the overall bbox is
    # too wide for the drawing-bbox pass to catch.
    zones = detect_zones(paths, header_threshold, raw_drawings=raw_drawings)

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

    color_profile = profile_span_colors(sections)

    # Named reasons rather than one opaque boolean. Ordered most to least
    # likely to mean the extraction itself is wrong, so a triage step can act
    # on the first entry.
    review_reasons = []
    if unresolved_corruption_count > 0:
        review_reasons.append("unresolved_corruption")
    if ocr_info.get("ocr_skipped_reason"):
        review_reasons.append("ocr_" + ocr_info["ocr_skipped_reason"])
    if not header_region["detected_by_content"]:
        # Everything positional downstream is a guess when this is set, so a
        # near-miss reported alongside it is far more likely a false alarm.
        review_reasons.append("header_not_detected")
    if not page_name_detected:
        review_reasons.append("page_name_not_detected")
    if sections.get("annotation_near_misses"):
        review_reasons.append("annotation_near_miss")
    if sections.get("production_mark_near_misses"):
        review_reasons.append("production_mark_near_miss")
    if color_profile["inferred_annotation_colors"]:
        review_reasons.append("unknown_annotation_color")

    # Canonical ACnnnV.n, synthesised once so no consumer re-derives it. Every
    # section is passed, not just body: the on-pack instance is on the pack
    # regardless of which bucket the header band sorted it into. Pulled out to
    # a local so header reconstruction below can reuse it (needs the on-pack
    # artboard bounds and the format-validity/consistency findings).
    ac_reference_result = synthesize_ac_reference(
        sections["header_table"], body_text_all,
        all_spans=(sections["body"] + sections["header_table"]
                   + sections["annotations"]
                   + sections.get("annotation_near_misses", [])),
        page_w_mm=pt_to_mm(width_pt), page_h_mm=pt_to_mm(height_pt),
    )
    on_pack_position = ac_reference_result.get("on_pack_position") or {}

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
        "ac_reference": ac_reference_result,
        # Header table rebuilt purely from span/path geometry -- column
        # boundaries, label/value pairing, row/col-pair numbers, per-cell
        # formatting and alignment. See build_header_reconstruction's own
        # docstring for the full algorithm.
        "header_reconstructed": build_header_reconstruction(
            sections["header_table"], clean_paths,
            {
                "page_name": detected_name,
                "page_number": page_index + 1,
                "page_width_mm": pt_to_mm(width_pt),
                "page_height_mm": pt_to_mm(height_pt),
                "artboard_bounds_mm": on_pack_position.get("artboard_bounds_mm"),
                "ac_reference_computed": ac_reference_result,
            },
        ),
        # Declared size vs printed callouts vs drawn geometry, reconciled once.
        "dimension_check": synthesize_dimension_check(
            sections["header_table"], sections["annotations"],
            sections.get("annotation_near_misses", []), clean_paths,
            pt_to_mm(width_pt), pt_to_mm(height_pt),
        ),
        # How colour is actually used on this page, so a new vendor's callout
        # palette surfaces as data instead of silently reading as body copy.
        "color_profile": color_profile,
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
            "header_unlabelled_rows_absorbed": header_region.get("unlabelled_rows_absorbed", 0),
            "inferred_annotation_colors": color_profile["inferred_annotation_colors"],
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
            # Why, not just whether. A near-miss on a page whose header was
            # never located is a very different thing from one on a page that
            # extracted cleanly, and collapsing both into a single boolean left
            # the workflow no way to tell them apart.
            "review_reasons": review_reasons,
            "needs_review": bool(review_reasons),
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
    review_reasons = sorted({
        r for p in result.values()
        for r in p.get("extraction_confidence", {}).get("review_reasons", [])
    })
    if failed_pages:
        review_reasons.insert(0, "page_extraction_failed")
    result["document_meta"] = {
        "filename": filename,
        "page_count": len([k for k in result if k.startswith("page")]),
        "failed_pages": failed_pages,
        "total_repairs_applied": total_repairs,
        "total_unresolved_corruption": total_unresolved,
        "ocr_available": TESSERACT_AVAILABLE,
        "ocr_skip_reasons": ocr_skip_reasons,
        "extraction_seconds": elapsed,
        "review_reasons": review_reasons,
        "needs_review": needs_review,
    }
    logger.info(
        "Extracted %s: pages=%d failed=%s repairs=%d unresolved_corruption=%d "
        "ocr_skips=%s needs_review=%s reasons=%s elapsed=%.2fs",
        filename or "<upload>", result["document_meta"]["page_count"],
        failed_pages, total_repairs, total_unresolved,
        ocr_skip_reasons, needs_review, review_reasons, elapsed,
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
        # Explicit charset: some HTTP clients (n8n's HTTP Request node
        # included) don't honour RFC 8259's "JSON is always UTF-8" and
        # sniff/guess the response encoding when Content-Type omits it —
        # observed decoding this body as GBK, turning "ó" into a CJK
        # character (U+8D38) despite the bytes on the wire being correct
        # UTF-8. Naming the charset removes the ambiguity outright.
        return JSONResponse(content=result, media_type="application/json; charset=utf-8")
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
        }, media_type="application/json; charset=utf-8")
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