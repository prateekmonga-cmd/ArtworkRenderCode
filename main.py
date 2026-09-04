"""
Artwork Compliance Extraction FastAPI Application v2.0
======================================================
Enhanced: zone detection, rotation, ligature repair, annotation filtering.
"""

import fitz  # PyMuPDF
import re
import os
import math
import io
import asyncio
import contextlib
import ctypes
import gc
import unicodedata
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

# ─── Concurrency bound ───────────────────────────────────────
# Heavy work runs via run_in_threadpool, whose default pool is 40 threads, so
# without a bound 40 documents can be parsed simultaneously and peak memory is
# whatever the traffic happens to be. On a 512 MB instance that is the one
# failure mode worth engineering away: memory should be a property of the
# CODE, not of the load.
#
# Measured peak RSS parsing the heaviest real artworks together (baseline
# ~85 MB): 1 -> 95 MB, 2 -> 106, 4 -> 129, 8 -> 146, 16 -> 150. Per-request
# cost falls as concurrency rises because the allocator reuses freed blocks,
# so the curve flattens rather than climbing linearly. 4 keeps the extraction
# side near ~130 MB while leaving room for upload buffers and JSON
# serialisation of several concurrent responses.
#
# Excess requests WAIT rather than fail: a queued execution is slower, a
# crashed instance takes every other execution down with it.
EXTRACT_CONCURRENCY = int(os.getenv("EXTRACT_CONCURRENCY", "8"))
_extract_slots: Optional[Any] = None   # (loop, asyncio.Semaphore)


def _slots():
    """Semaphore for the CURRENT event loop.

    An asyncio.Semaphore binds to the loop that first awaits it and raises
    "bound to a different event loop" anywhere else. Uvicorn serves on one loop
    so a plain global would happen to work in production, but it breaks under
    tests or any second loop -- and a concurrency guard that throws is worse
    than none. Keyed by loop so it is correct either way.
    """
    global _extract_slots
    loop = asyncio.get_running_loop()
    if _extract_slots is None or _extract_slots[0] is not loop:
        _extract_slots = (loop, asyncio.Semaphore(EXTRACT_CONCURRENCY))
    return _extract_slots[1]


# ─── Memory-aware admission ──────────────────────────────────
# A fixed request count is a poor proxy for memory: four sticker labels cost
# almost nothing, four 300-page inserts cost a lot. So the real gate is actual
# memory use, read from the container's own cgroup accounting -- the same
# number the OOM killer watches -- with the count above kept only as a backstop
# against unbounded thread growth.
#
# Nothing needs configuring: the limit is discovered from cgroup at startup.
# MEMORY_LIMIT_MB only exists as an override for environments that do not
# expose cgroup files.
MEMORY_HIGH_WATER = float(os.getenv("MEMORY_HIGH_WATER", "0.75"))  # admit below 75% of the limit
MEMORY_WAIT_TIMEOUT_S = float(os.getenv("MEMORY_WAIT_TIMEOUT_S", "60"))
MEMORY_POLL_S = 0.2

_CGROUP_LIMIT_FILES = (
    "/sys/fs/cgroup/memory.max",                      # cgroup v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",    # cgroup v1
)
_CGROUP_USAGE_FILES = (
    "/sys/fs/cgroup/memory.current",                  # cgroup v2
    "/sys/fs/cgroup/memory/memory.usage_in_bytes",    # cgroup v1
)

try:                     # optional: only used when cgroup files are absent
    import psutil        # noqa: F401
    _PROC = psutil.Process()
except Exception:
    _PROC = None


def _read_int_file(paths) -> Optional[int]:
    for path in paths:
        try:
            with open(path) as fh:
                raw = fh.read().strip()
            if raw == "max":
                continue
            value = int(raw)
            # cgroup v1 reports a near-2^63 sentinel when memory is unlimited.
            if 0 < value < (1 << 62):
                return value
        except Exception:
            continue
    return None


def _detect_memory_limit() -> Optional[int]:
    override = os.getenv("MEMORY_LIMIT_MB")
    if override:
        try:
            return int(float(override) * 1024 * 1024)
        except ValueError:
            logger.warning("Ignoring unparseable MEMORY_LIMIT_MB=%r", override)
    return _read_int_file(_CGROUP_LIMIT_FILES)


def _current_memory() -> Optional[int]:
    """Bytes currently charged to this container, or this process as a fallback."""
    usage = _read_int_file(_CGROUP_USAGE_FILES)
    if usage is not None:
        return usage
    if _PROC is not None:
        try:
            return _PROC.memory_info().rss
        except Exception:
            return None
    return None


MEMORY_LIMIT_BYTES = _detect_memory_limit()
_inflight = 0


# ── Ligature recovery ───────────────────────────────────────────────────────
#
# These artworks embed subset Calibri whose ToUnicode CMap is wrong: every
# ligature glyph is mapped to an unrelated codepoint. Read with the default
# flags, MuPDF gives up on those glyphs and returns U+FFFD -- which collapses
# ti, fi, fl, ft and tt onto ONE sentinel and throws away the only thing that
# tells them apart. The repair layer then had no choice but to GUESS, and it
# guessed "ti" every time: right on a carton (where the damage really is ti),
# wrong on an insert, where fi and fl also occur.
#
# TEXT_USE_CID_FOR_UNKNOWN_UNICODE makes MuPDF fall back to the glyph's CID
# instead of the sentinel, so each ligature arrives as its own codepoint and
# can be decoded EXACTLY. Measured over all 21 artworks in the corpus:
#   default flags : 1522 U+FFFD, every one of them guessed
#   with this flag: 0 U+FFFD, every character identified
# The constant has been spelled two ways across PyMuPDF releases:
# TEXT_CID_FOR_UNKNOWN_UNICODE (1.24.x) and TEXT_USE_CID_FOR_UNKNOWN_UNICODE
# (1.26+, which also keeps the old name as an alias). Naming only the newer one
# crashed this service on import against the pinned 1.24.9. Both are tried, then
# the literal bit, so the extractor runs on any version.
_CID_FLAG_BIT = 0x80
_CID_FLAG_NAME = next(
    (n for n in ("TEXT_USE_CID_FOR_UNKNOWN_UNICODE", "TEXT_CID_FOR_UNKNOWN_UNICODE")
     if hasattr(fitz, n)), None)
_CID_FLAG = getattr(fitz, _CID_FLAG_NAME) if _CID_FLAG_NAME else _CID_FLAG_BIT
TEXT_FLAGS = fitz.TEXT_PRESERVE_WHITESPACE | _CID_FLAG


def cid_flag_status() -> dict:
    """Whether the CID fallback is actually available, for /health and startup.

    If a future PyMuPDF drops the flag the extractor would quietly go back to
    U+FFFD and guessing, which is the failure this whole change exists to
    remove -- so it is reported rather than left to be discovered in a report.
    """
    return {
        "pymupdf_version": getattr(fitz, "VersionBind", "unknown"),
        "cid_flag_name": _CID_FLAG_NAME or "(not exported; using literal 0x80)",
        "cid_flag_value": _CID_FLAG,
        "resolved_by_name": _CID_FLAG_NAME is not None,
    }

def _memory_snapshot() -> dict:
    """Memory numbers for /health.

    Free Render plans paywall the memory graph entirely and the paid one is
    fixed at one sample per hour, which is far too coarse to catch a spike
    inside a single report run. The workflow already pings /health to wake the
    instance before every report, so reporting the cgroup counter -- the same
    number the OOM killer watches -- gives one reading per run, captured in the
    execution record, at no cost and on any plan.

    Every field is Optional on purpose: cgroup files do not exist outside a
    container, and this endpoint must never fail, because it is the wake-up
    probe the whole workflow gates on. Missing numbers are reported as null,
    never as an exception.
    """
    used = _current_memory()
    limit = MEMORY_LIMIT_BYTES
    return {
        "memory_used_mb": round(used / 1048576, 1) if used is not None else None,
        "memory_limit_mb": round(limit / 1048576, 1) if limit else None,
        "memory_pct": round(100.0 * used / limit, 1) if used is not None and limit else None,
    }


async def _wait_for_headroom() -> None:
    """Hold a new extraction until memory drops below the high-water mark.

    Deliberately never blocks the FIRST extraction: if a single document on its
    own exceeds the mark there is nothing in flight to wait for, and refusing to
    start would deadlock the service instead of merely slowing it. Same reason
    for the timeout -- degrade to "run it anyway" rather than hang forever.
    """
    if MEMORY_LIMIT_BYTES is None:
        return                      # cannot measure; the count cap is all we have
    ceiling = MEMORY_LIMIT_BYTES * MEMORY_HIGH_WATER
    loop = asyncio.get_running_loop()
    deadline = loop.time() + MEMORY_WAIT_TIMEOUT_S
    waited = False
    while True:
        used = _current_memory()
        if used is None or used < ceiling:
            break
        if _inflight == 0:
            break                   # nothing to wait for -- see docstring
        if loop.time() >= deadline:
            logger.warning(
                "Memory still at %.0f MB after waiting %.0fs; proceeding anyway",
                used / 1048576, MEMORY_WAIT_TIMEOUT_S)
            break
        if not waited:
            logger.info("Memory at %.0f MB of %.0f MB limit; queueing extraction",
                        used / 1048576, MEMORY_LIMIT_BYTES / 1048576)
            waited = True
        await asyncio.sleep(MEMORY_POLL_S)


def _trim_heap() -> None:
    """Return freed heap to the OS.

    Python frees the objects an extraction allocated, but glibc keeps the arenas
    and RSS never drops -- and RSS is what the cgroup counter and the OOM killer
    watch. Measured on the live service: 309 MB resident with reports_in_flight
    at 0 and nothing actually running, climbing to 400 MB (78% of a 512 MB cap)
    across a single execution. Chromium then needs another 150-350 MB on top,
    which is what kills the container mid-render and surfaces in n8n as
    "the connection was aborted, perhaps the server is offline".

    malloc_trim is glibc-only; anything else is a no-op, never an error.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


@contextlib.asynccontextmanager
async def _extraction_slot():
    """Admit one extraction: count cap first, then memory headroom."""
    global _inflight
    async with _slots():
        await _wait_for_headroom()
        _inflight += 1
        try:
            yield
        finally:
            _inflight -= 1
            # Give the memory back HERE rather than at the next admission: the
            # thing most likely to run next is a report, and it launches
            # Chromium without ever passing through this slot.
            _trim_heap()


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


# ─── Character-level geometry for alignment rules ────────────
# Several SOP requirements are about *alignment*, not content: the batch-coding
# and registration colons must line up vertically (§2.3.x), and composition
# claims plus "q.s." must be "perfectly right-aligned in a straight vertical
# line" (§1.3.4). Neither is answerable from the span bbox:
#
#   - the bbox includes trailing whitespace, so its right edge is not where the
#     visible text ends. Measured on real spans: "Código     : " and
#     "Vence       : " carry 2.5 pt (0.88 mm) of trailing-space inflation while
#     "Fecha Fab:" carries none, which alone is ~9x the tolerance these rules
#     imply — comparing raw x1 values would invent misalignment that isn't there
#     and hide misalignment that is;
#   - a colon in the middle of a span ("India; Para: Rhydburg LLP, India") has
#     no derivable position at all.
#
# rawdict already returns a bbox per character; the extractor was reading only
# the character itself and dropping the geometry. Only alignment-relevant
# anchors are kept — emitting every character would multiply the payload many
# times over for data no rule reads. Note the public "bbox" field is snapped to
# whole points (snap_coord), far too coarse here, so these are all derived from
# the unrounded boxes via bbox_to_mm (0.01 mm).

ALIGNMENT_CHARS = (":",)
SEGMENT_GAP_MIN = 2   # 2+ consecutive spaces reads as a column break, not a word space


def _visible(chars: list) -> list:
    return [c for c in chars if (c.get("c") or "").strip()]


def _char_segments(chars: list) -> list:
    """Split a span on runs of 2+ spaces — the artwork's own column separator.

    "Loratadina                    10 mg" is one span, but it is really two
    columns: the salt name and its claim. The claim's own right edge is what
    §1.3.4 requires to be aligned, and it is invisible without this split.
    """
    segments = []
    text_parts: list = []
    boxes: list = []
    gap = 0

    def flush():
        if boxes:
            # Repair here too. These segments are built straight from the raw
            # per-character stream, so they bypassed the repair the span's own
            # "text" field goes through -- leaving U+FFFD in every segment of
            # every insert while the span above it read correctly. Words already
            # in KNOWN_CORRECTIONS ("activo", "cantidad") were surviving intact
            # in this one field, which is how it was found.
            segments.append({
                "text": repair_text("".join(text_parts).strip())[0],
                "bbox_mm": bbox_to_mm((
                    min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes),
                )),
            })
        text_parts.clear()
        boxes.clear()

    for ch in chars:
        char = ch.get("c") or ""
        box = ch.get("bbox")
        if not char.strip():
            gap += 1
            continue
        if gap >= SEGMENT_GAP_MIN:
            flush()
        elif gap == 1 and text_parts:
            text_parts.append(" ")   # ordinary word space, stays in the segment
        gap = 0
        text_parts.append(char)
        if box:
            boxes.append(box)
    flush()
    return segments


def char_alignment_geometry(chars: list, span_bbox) -> dict:
    """Alignment anchors for one span, derived from its per-character boxes.

    Only keys that carry information are returned: a span with no colon gets no
    "colons", a span with no internal column gap gets no "segments", and
    "visible_bbox_mm" appears only when whitespace actually moved an edge — so
    the common case (a plain span) adds nothing to the payload.
    """
    if not chars:
        return {}

    out = {}
    vis = _visible(chars)
    boxed = [c for c in vis if c.get("bbox")]
    if boxed:
        visible_bbox = (
            min(c["bbox"][0] for c in boxed), min(c["bbox"][1] for c in boxed),
            max(c["bbox"][2] for c in boxed), max(c["bbox"][3] for c in boxed),
        )
        # Only worth reporting when it differs from the span's own box; equal
        # boxes would just duplicate bbox_mm on every span on the page.
        if span_bbox is None or any(
            abs(a - b) > 0.01 for a, b in zip(visible_bbox, tuple(span_bbox))
        ):
            out["visible_bbox_mm"] = bbox_to_mm(visible_bbox)

    colons = []
    for i, ch in enumerate(chars):
        if (ch.get("c") or "") in ALIGNMENT_CHARS and ch.get("bbox"):
            b = ch["bbox"]
            colons.append({
                "char": ch["c"],
                "index": i,
                "bbox_mm": bbox_to_mm(b),
                # x0 is the anchor an alignment check compares across rows --
                # glyph widths differ, so left edges line up where right edges
                # need not.
                "x0_mm": pt_to_mm(b[0]),
            })
    if colons:
        out["colons"] = colons

    segments = _char_segments(chars)
    if len(segments) > 1:
        out["segments"] = segments

    return out


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

    text_dict = page.get_text("rawdict", flags=TEXT_FLAGS)
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

                # Alignment anchors (colon positions, column segments, the
                # whitespace-free box). Derived from the raw glyph boxes, so it
                # reflects where ink actually sits regardless of any text repair
                # applied above. Omitted entirely when the span has nothing
                # alignment-relevant in it.
                alignment = char_alignment_geometry(chars, bbox)
                if alignment:
                    span_data["alignment"] = alignment

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
# Seed rows further apart than this belong to different clusters, and only one
# of them is the header. An Insert's revision footer ("Version: AEMPS-Medaxone
# 1 g-V2-Mar2025") matches the "version" alias 272 mm below the real header;
# the largest gap WITHIN a real header is 12 mm, so 50 mm separates the two
# cases with room to spare in both directions.
HEADER_SEED_CLUSTER_GAP_PT = 50.0 / PT_TO_MM
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

    # Keep only ONE cluster of seeds. _grow_header_band spans min(seed) to
    # max(seed) before any gap check runs, so a single false seed in the body
    # captures everything between: an Insert's revision footer matching
    # "version" 272 mm down took AC2500 p4's header from 18 rows to 130, and no
    # downstream filter can recover from a band that wrong.
    if seed_idx:
        clusters, current = [], [seed_idx[0]]
        for prev, nxt in zip(seed_idx, seed_idx[1:]):
            if rows[nxt]["bbox"][1] - rows[prev]["bbox"][1] > HEADER_SEED_CLUSTER_GAP_PT:
                clusters.append(current)
                current = []
            current.append(nxt)
        clusters.append(current)
        if len(clusters) > 1:
            # Most labels wins; ties go to the topmost, since SOP 2.1 puts the
            # header at the top of the artboard.
            def _label_count(cluster):
                return len({canon for i in cluster
                            for alias, canon in HEADER_FIELD_LABELS.items()
                            if alias in _label_key(" ".join(rows[i]["words"]))})
            best = max(clusters, key=lambda c: (_label_count(c), -rows[c[0]]["bbox"][1]))
            dropped = sum(len(c) for c in clusters) - len(best)
            logger.info("Header seeds formed %d clusters; kept %d rows, dropped %d "
                        "stray label match(es) in body copy",
                        len(clusters), len(best), dropped)
            seed_idx = best
            found_labels = {canon for i in seed_idx
                            for alias, canon in HEADER_FIELD_LABELS.items()
                            if alias in _label_key(" ".join(rows[i]["words"]))}

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
# BARCODE DECODING (from the drawn bars, not OCR)
# ═══════════════════════════════════════════════════════════════
#
# Rule 1-48 wants three barcode values compared: the header cell, the Art
# Creation record, and "the actual barcode graphic's underlying value". The
# third was unavailable -- the human-readable caption under the bars is not
# text. Probing the real artwork showed why: the caption is one filled drawing
# of 319 items, 288 of them CURVES, i.e. the digits were converted to outlines.
# There is no glyph and no barcode font on the page, so no text-extraction mode
# can ever read it.
#
# OCR would be the obvious fallback and is the wrong tool. The bars themselves
# are 30 filled rectangles, and EAN-13 is a fixed 95-module encoding, so the
# value can be recovered by arithmetic: exact, no Tesseract dependency, no
# confidence threshold, and it yields what a SCANNER would read rather than what
# a human sees printed -- the stronger reading of the rule, and the only one
# that catches bars and caption disagreeing.
#
# Verified against the real artwork: decoded 8908001212069 with valid check
# digit, matching the Art Creation record and mismatching the header cell
# (…2096), which is where the actual defect is.

# 95 modules: 3 start guard | 6x7 left | 5 centre guard | 6x7 right | 3 end guard
EAN13_MODULES = 95
EAN13_MIN_BARS = 20          # a real EAN-13 draws 30; fewer means it isn't one
EAN13_NOMINAL_MODULE_MM = 0.33   # module width at 100% magnification (GS1)

# Left-hand digits carry the leading digit in their parity choice: L (odd) or
# G (even). Right-hand digits use R, the complement of L.
_EAN_L = {"0001101": 0, "0011001": 1, "0010011": 2, "0111101": 3, "0100011": 4,
          "0110001": 5, "0101111": 6, "0111011": 7, "0110111": 8, "0001011": 9}
_EAN_G = {"0100111": 0, "0110011": 1, "0011011": 2, "0100001": 3, "0011101": 4,
          "0111001": 5, "0000101": 6, "0010001": 7, "0001001": 8, "0010111": 9}
_EAN_R = {"1110010": 0, "1100110": 1, "1101100": 2, "1000010": 3, "1011100": 4,
          "1001110": 5, "1010000": 6, "1000100": 7, "1001000": 8, "1110100": 9}
_EAN_FIRST = {"000000": 0, "001011": 1, "001101": 2, "001110": 3, "010011": 4,
              "011001": 5, "011100": 6, "010101": 7, "010110": 8, "011010": 9}


def _ean13_check_digit(twelve: str) -> int:
    total = sum(int(c) * (3 if i % 2 else 1) for i, c in enumerate(twelve))
    return (10 - total % 10) % 10


def _decode_ean13_bars(bars: list) -> dict:
    """Decode a left-to-right sorted list of bar (x0, x1) pairs."""
    x0, x1 = bars[0][0], bars[-1][1]
    span = x1 - x0
    if span <= 0:
        return {"value": None, "reason": "zero-width bar block"}
    module = span / EAN13_MODULES

    # Sample each module at its centre -- robust against sub-point rounding in
    # the bar edges, which a run-length walk would accumulate.
    pattern = "".join(
        "1" if any(b[0] - 1e-6 <= x0 + (i + 0.5) * module <= b[1] + 1e-6 for b in bars)
        else "0"
        for i in range(EAN13_MODULES)
    )

    guards_ok = (pattern[:3] == "101" and pattern[45:50] == "01010"
                 and pattern[92:] == "101")
    if not guards_ok:
        return {"value": None, "reason": "guard patterns do not match EAN-13",
                "module_pattern": pattern}

    parity, left_digits = "", []
    for i in range(6):
        chunk = pattern[3 + i * 7: 10 + i * 7]
        if chunk in _EAN_L:
            parity += "0"
            left_digits.append(_EAN_L[chunk])
        elif chunk in _EAN_G:
            parity += "1"
            left_digits.append(_EAN_G[chunk])
        else:
            return {"value": None, "reason": f"unrecognised left chunk {chunk}",
                    "module_pattern": pattern}

    right_digits = []
    for i in range(6):
        chunk = pattern[50 + i * 7: 57 + i * 7]
        if chunk not in _EAN_R:
            return {"value": None, "reason": f"unrecognised right chunk {chunk}",
                    "module_pattern": pattern}
        right_digits.append(_EAN_R[chunk])

    first = _EAN_FIRST.get(parity)
    if first is None:
        return {"value": None, "reason": f"parity {parity} is not a valid leading digit",
                "module_pattern": pattern}

    code = str(first) + "".join(str(d) for d in left_digits + right_digits)
    expected = _ean13_check_digit(code[:12])
    return {
        "value": code,
        "symbology": "EAN-13",
        "check_digit_valid": expected == int(code[12]),
        "check_digit_expected": expected,
        "module_width_pt": round(module, 4),
    }


BAR_MIN_HEIGHT_PT = 4.0      # shorter filled slivers are rules, ticks or glyph parts
BAR_TOP_EPS_PT = 1.0         # bars of one symbol share a top edge this closely
BAR_GAP_FACTOR = 12.0        # a horizontal gap this many bar-widths wide separates symbols


def decode_barcodes(raw_drawings: Optional[list]) -> list:
    """Find and decode barcode graphics on a page.

    Bars are collected PAGE-WIDE rather than per drawing. Grouping varies by
    how the artwork was produced: AC2491 draws all 30 bars inside one compound
    drawing, while AC2475/AC3146 spread the same 30 bars across separate
    drawing objects. An earlier version required one drawing to hold them all
    and therefore reported "no barcode" for the latter -- on artworks whose
    header declares a barcode, which would have read as a compliance defect
    that did not exist.

    Bars are then clustered by their shared TOP edge, not their full box: the
    guard bars of an EAN-13 are drawn longer than the data bars, so bottoms
    legitimately differ within one symbol.

    Candidates that cannot be decoded are still reported, with a reason -- a
    barcode we failed to read is a finding, not something to omit silently.
    """
    candidates = []
    for drawing in (raw_drawings or []):
        # Filled only. Stroked rects are table borders and dielines.
        if drawing.get("type") not in ("f", "fs"):
            continue
        for it in (drawing.get("items") or []):
            if it[0] != "re":
                continue
            r = fitz.Rect(it[1])
            w, h = r.x1 - r.x0, r.y1 - r.y0
            if w > 0 and h > w and h >= BAR_MIN_HEIGHT_PT:
                candidates.append(r)

    # Cluster by shared top edge.
    bands: list = []
    for r in sorted(candidates, key=lambda r: (r.y0, r.x0)):
        for band in bands:
            if abs(band[0].y0 - r.y0) <= BAR_TOP_EPS_PT:
                band.append(r)
                break
        else:
            bands.append([r])

    results = []
    for band in bands:
        band.sort(key=lambda r: r.x0)
        # Split a band into runs, so two symbols side by side at the same height
        # are not merged into one nonsense pattern.
        widths = sorted(r.x1 - r.x0 for r in band)
        typical = widths[len(widths) // 2] or 1.0
        runs, current = [], [band[0]]
        for prev, nxt in zip(band, band[1:]):
            if nxt.x0 - prev.x1 > typical * BAR_GAP_FACTOR:
                runs.append(current)
                current = [nxt]
            else:
                current.append(nxt)
        runs.append(current)

        for run in runs:
            if len(run) < EAN13_MIN_BARS:
                continue
            block = fitz.Rect(run[0].x0, min(r.y0 for r in run),
                              run[-1].x1, max(r.y1 for r in run))
            entry = {
                "bar_count": len(run),
                "bbox_mm": bbox_to_mm(block),
                "width_mm": pt_to_mm(block.x1 - block.x0),
                "height_mm": pt_to_mm(block.y1 - block.y0),
            }
            entry.update(_decode_ean13_bars([(r.x0, r.x1) for r in run]))

            mw = entry.get("module_width_pt")
            if mw:
                # Not pt_to_mm here: it rounds to 2 decimals, which turns a
                # 0.0782 mm module into 0.08 and skews the magnification below
                # by half a percent. A module is sub-0.1 mm, so it needs digits.
                module_mm = mw * PT_TO_MM
                entry["module_width_mm"] = round(module_mm, 4)
                # Printed size relative to the GS1 nominal. Scannability depends
                # on this, so it is reported as measured data for a size rule to
                # judge -- no threshold is applied here.
                entry["magnification_percent"] = round(
                    module_mm / EAN13_NOMINAL_MODULE_MM * 100, 1)
            results.append(entry)

    results.sort(key=lambda e: (e["bbox_mm"][1], e["bbox_mm"][0]) if e.get("bbox_mm") else (0, 0))
    return results


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
    "size lxh": "dimension",
    "size lxwxh": "dimension",
    "size (w)": "dimension",
    # Bare "Size" is safe as a last-resort alias: canonicalFor tries exact and
    # word-boundary matches across every alias BEFORE falling back to substring,
    # so "artwork size" and "flat size" are still claimed by their own entries.
    "size": "dimension",
    "product code": "product_code",
    "style": "style",
    "barcode": "barcode",
    "printing overlay": "printing_overlay",
    "repeat": "repeat",
    "printing zone": "printing_zone",
    "dia of tube": "dia_of_tube",
}


def _recon_label_key(text: Optional[str]) -> str:
    """Normalise a header label for alias matching.

    Parentheses must go, and become a SPACE rather than nothing: real headers
    write the size field as "Size in mm (LxWxH)", "Size (LxW)", "Size (LxH)" or
    bare "Size". Stripping only ".#:" left "size (lxw)", which matched no alias,
    so the declared dimension was silently absent from every Label and Insert
    header -- and on AC6241, where the label is just "Size", the value ran into
    the neighbouring Style cell instead ("Top Open Bottom Paste 78x70x52 mm").
    """
    cleaned = re.sub(r"[.#:()\[\]]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


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


# Tolerance for treating two drawn edges as the same grid line. Real headers
# land within ~0.05 pt of each other; 0.5 pt is loose enough to absorb that
# without merging genuinely distinct 6 pt rows.
GRID_EPS_PT = 0.5
GRID_MIN_LINE_PT = 5.0     # shorter than this is a tick or a swatch edge, not a rule
# The drawn header border sits a little below the lowest glyph (carton: text to
# 190.3 pt, border to 191.4 pt), so the band gets a small pad before the overlap
# test decides what belongs to it.
HEADER_BAND_PAD_PT = 20.0
HEADER_BAND_MIN_OVERLAP = 0.8


def _dedupe_edges(values: list, eps: float = GRID_EPS_PT) -> list:
    out = []
    for v in sorted(values):
        if not out or v - out[-1] > eps:
            out.append(v)
    return out


def _header_grid(paths: list) -> dict:
    """Recover the header table's drawn cell grid.

    Two conventions appear in real artwork and both are handled:

      rects  (Outer Carton) -- one stroked rectangle per cell, plus one outer
             border. Merged cells are drawn as a single taller rect, so the
             rects themselves carry the merge information and must be used
             as-is rather than re-derived from edges.

      lines  (Foil) -- a bare ruled grid with no per-cell rects. Cells are
             implied by the lines, and a PARTIAL rule (one spanning only part
             of the table width) splits only the columns it actually crosses:
             the Foil's 90 mm rule at y=23.76 divides AC Reference from
             Version# on the right half while the left half stays one tall row.

    Everything here is measured off the drawn geometry, never off text -- the
    text sits inside its cell, so text extent understates the table (measured:
    176.96 x 47.06 against a real 180.00 x 48.00).
    """
    usable = [p for p in (paths or []) if p.get("bbox")]
    stroked = [p for p in usable if not p.get("fill_color_hex")]

    rects = [p for p in stroked
             if (p["bbox"][2] - p["bbox"][0]) > GRID_EPS_PT
             and (p["bbox"][3] - p["bbox"][1]) > GRID_EPS_PT]
    h_lines = [p for p in stroked
               if (p["bbox"][3] - p["bbox"][1]) <= GRID_EPS_PT
               and (p["bbox"][2] - p["bbox"][0]) > GRID_MIN_LINE_PT]
    v_lines = [p for p in stroked
               if (p["bbox"][2] - p["bbox"][0]) <= GRID_EPS_PT
               and (p["bbox"][3] - p["bbox"][1]) > GRID_MIN_LINE_PT]

    empty = {"convention": None, "table_bbox": None, "table_bbox_mm": None,
             "cells": [], "col_edges": [], "border_stroke_pt": []}

    def _mm_extent(sources: list) -> Optional[list]:
        """Millimetre extent taken from the paths' own bbox_mm.

        The public "bbox" is snapped to whole points, which is fine for deciding
        which cell owns a span but not for a size the spec checks: 180 mm is
        510.24 pt, snaps to 510, and reads back as 179.92 mm -- a 0.08 mm error
        invented by rounding. bbox_mm is derived from the unsnapped rect.
        """
        boxed = [p for p in sources if isinstance(p.get("bbox_mm"), list)]
        if not boxed:
            return None
        return [
            min(p["bbox_mm"][0] for p in boxed), min(p["bbox_mm"][1] for p in boxed),
            max(p["bbox_mm"][2] for p in boxed), max(p["bbox_mm"][3] for p in boxed),
        ]

    if len(rects) > 1:
        outer = max(rects, key=lambda r: _recon_area_of(r["bbox"]))
        table_bbox = list(outer["bbox"])
        cells = [list(r["bbox"]) for r in rects if r is not outer]
        if not cells:
            return empty
        strokes = sorted({r["stroke_width_pt"] for r in rects if r is not outer
                          if isinstance(r.get("stroke_width_pt"), (int, float))})
        return {
            "convention": "rects",
            "table_bbox": table_bbox,
            "table_bbox_mm": _mm_extent([outer]),
            "cells": cells,
            "col_edges": _dedupe_edges([c[0] for c in cells] + [c[2] for c in cells]),
            "border_stroke_pt": strokes,
        }

    if len(v_lines) >= 2 and h_lines:
        col_edges = _dedupe_edges([l["bbox"][0] for l in v_lines])
        # The verticals span the table's full height, so they define top and
        # bottom -- the horizontals only tell us where rows divide, and the
        # bottom-most rule is not necessarily the table's bottom edge.
        top = min(l["bbox"][1] for l in v_lines)
        bottom = max(l["bbox"][3] for l in v_lines)
        table_bbox = [col_edges[0], top, col_edges[-1], bottom]

        cells = []
        for ci in range(len(col_edges) - 1):
            cx0, cx1 = col_edges[ci], col_edges[ci + 1]
            mid_x = (cx0 + cx1) / 2
            # Only rules that actually cross this column can divide it.
            ys = [top, bottom]
            for l in h_lines:
                b = l["bbox"]
                if b[0] - GRID_EPS_PT <= mid_x <= b[2] + GRID_EPS_PT:
                    ys.append(b[1])
            ys = _dedupe_edges([y for y in ys if top - GRID_EPS_PT <= y <= bottom + GRID_EPS_PT])
            for ri in range(len(ys) - 1):
                cells.append([cx0, ys[ri], cx1, ys[ri + 1]])

        strokes = sorted({l["stroke_width_pt"] for l in (h_lines + v_lines)
                          if isinstance(l.get("stroke_width_pt"), (int, float))})
        return {
            "convention": "lines",
            "table_bbox": table_bbox,
            # The verticals bound the table on all four sides here, so their
            # own mm extent is the table's.
            "table_bbox_mm": _mm_extent(v_lines),
            "cells": cells,
            "col_edges": col_edges,
            "border_stroke_pt": strokes,
        }

    return empty


def _cell_containing(bbox: list, cells: list) -> Optional[list]:
    """Smallest drawn cell containing a text box's centre.

    Smallest, not first: in the rects convention the outer border was already
    dropped, but a merged cell can still enclose a neighbour's area, and the
    tighter cell is always the right owner.
    """
    if not cells or not bbox:
        return None
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    best = None
    for c in cells:
        if (c[0] - GRID_EPS_PT <= cx <= c[2] + GRID_EPS_PT
                and c[1] - GRID_EPS_PT <= cy <= c[3] + GRID_EPS_PT):
            if best is None or _recon_area_of(c) < _recon_area_of(best):
                best = c
    return list(best) if best else None


def _recon_column_bounds_from_grid(grid: dict) -> Optional[list]:
    edges = grid.get("col_edges") or []
    if len(edges) < 2:
        return None
    return [[edges[i], edges[i + 1]] for i in range(len(edges) - 1)]


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


def _recon_describe_group(group_spans: list, cell_bbox: Optional[list],
                          swatches: list, col_bounds: Optional[list] = None) -> dict:
    sorted_spans = sorted(group_spans, key=lambda s: (s["bbox"][1], s["bbox"][0]))
    lines = [s["text"].strip() for s in sorted_spans]
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    text_bbox = [
        min(s["bbox"][0] for s in sorted_spans), min(s["bbox"][1] for s in sorted_spans),
        max(s["bbox"][2] for s in sorted_spans), max(s["bbox"][3] for s in sorted_spans),
    ]
    # cell_bbox is the cell as actually DRAWN, looked up from the grid, so
    # vertical alignment is measured against the real row -- not, as before, a
    # +/-3 pt pad around the text, which made align_v self-fulfilling (text
    # padded symmetrically always reads "middle").
    cell_source = "drawn_cell"
    if not cell_bbox:
        # No grid recovered for this group. Fall back to the column width where
        # one is known so horizontal alignment still means something, and mark
        # the row extent as unknown rather than inventing one.
        cell_source = "column_only" if col_bounds else "text_extent"
        cell_bbox = ([col_bounds[0], text_bbox[1], col_bounds[1], text_bbox[3]]
                     if col_bounds else list(text_bbox))

    left_gap, right_gap = text_bbox[0] - cell_bbox[0], cell_bbox[2] - text_bbox[2]
    top_gap, bottom_gap = text_bbox[1] - cell_bbox[1], cell_bbox[3] - text_bbox[3]
    if abs(left_gap - right_gap) <= 3:
        align_h = "center"
    elif right_gap < left_gap:
        align_h = "right"
    else:
        align_h = "left"
    # Only meaningful against a real row. Without one the cell's vertical
    # extent IS the text's, so every group would read "middle" -- report
    # nothing rather than a value that is true by construction.
    if cell_source != "drawn_cell":
        align_v = None
    elif abs(top_gap - bottom_gap) <= 3:
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
        "cell_bbox": cell_bbox,
        "cell_bbox_mm": bbox_to_mm(cell_bbox),
        "cell_source": cell_source,
        "text_bbox": text_bbox,
        "text_bbox_mm": bbox_to_mm(text_bbox),
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
    # Keep a path only if MOST of it lies in the header band.
    #
    # Testing the top edge alone admitted an Insert's page border -- which
    # begins just below the header text and then runs the whole sheet -- and the
    # grid picked that 148x210 rect as its outer border, reporting AC3146's
    # header as 180 x 152.96 mm instead of 180 x 30. But requiring strict
    # containment overcorrected and dropped legitimate borders that sit a little
    # below the lowest glyph, costing AC3146 p4 the correct 180 x 30 it already
    # had. An overlap RATIO handles both without a magic distance: the page
    # border overlaps the band by under 1% of its height, a real header rule by
    # ~100%, whatever the page size.
    header_max_y = max(s["bbox"][3] for s in spans)
    band_bottom = header_max_y + HEADER_BAND_PAD_PT

    def _mostly_in_band(box: list) -> bool:
        overlap = max(0.0, min(box[3], band_bottom) - box[1])
        height = box[3] - box[1]
        if height <= GRID_EPS_PT:      # a horizontal rule has no height to share
            return box[1] <= band_bottom
        return overlap / height >= HEADER_BAND_MIN_OVERLAP

    all_paths = [p for p in (paths or [])
                 if p.get("bbox") and _mostly_in_band(p["bbox"])]
    swatches = [p for p in all_paths if p.get("fill_color_hex")]
    grid = _header_grid(all_paths)
    col_bounds = _recon_column_bounds_from_grid(grid)
    cells = grid.get("cells") or []

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
    # Counted from the DRAWN cells: a row boundary only counts if it spans the
    # FULL table width. Where one side of a row splits into two while the other
    # stays a single tall cell, that is one printed row, not two.
    #
    # Counting the left-hand column instead gives the same answer on every
    # artwork seen so far -- the merge is always on the left (Carton at row 4,
    # Insert/Foil/Label at row 1) -- but it silently assumes that, and would
    # overcount a header that merged on the right. The full-width test does not
    # care which side merges.
    row_count = None
    row_heights_mm = None
    if cells and grid.get("table_bbox"):
        tb = grid["table_bbox"]
        span_needed = (tb[2] - tb[0]) - 2 * GRID_EPS_PT
        by_top = {}
        for c in cells:
            by_top.setdefault(round(c[1], 1), []).append(c)

        boundaries = []
        for top, row_cells in sorted(by_top.items()):
            covered = sum(c[2] - c[0] for c in row_cells)
            if covered >= span_needed:
                boundaries.append(min(c[1] for c in row_cells))
        if boundaries:
            edges = boundaries + [tb[3]]
            row_count = len(boundaries)
            row_heights_mm = [pt_to_mm(edges[i + 1] - edges[i])
                              for i in range(len(boundaries))]

    if row_count is None:
        # No grid -- fall back to clustering the left column's own labels.
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

        # The cell each side actually occupies, looked up from the drawn grid.
        label_cell = _cell_containing(l["span"]["bbox"], cells)
        value_cell = None
        if value_spans:
            value_text_bbox = [
                min(s["bbox"][0] for s in value_spans), min(s["bbox"][1] for s in value_spans),
                max(s["bbox"][2] for s in value_spans), max(s["bbox"][3] for s in value_spans),
            ]
            value_cell = _cell_containing(value_text_bbox, cells)

        pairs.append({
            "row": row_number_for(l["span"]),
            "col_pair": (l["col"] // 2) + 1 if l["col"] >= 0 else None,
            "canonical": l["canon"],
            "label": _recon_describe_group([l["span"]], label_cell, swatches, label_col),
            "value": (_recon_describe_group(value_spans, value_cell, swatches, val_col)
                      if value_spans else None),
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
    #
    # Taken from the DRAWN border when the grid was recovered. Text extent is
    # not the table: text sits inside its cell, so measuring the spans gave
    # 176.96 x 47.06 for a header drawn at exactly 180.00 x 48.00 -- enough to
    # fail a 180x48 spec on a page that is dimensionally perfect. The text
    # extent is still reported separately as text_extent_bbox_mm, since a
    # "content must stay inside the table" check needs both.
    text_bbox = [
        min(s["bbox"][0] for s in spans), min(s["bbox"][1] for s in spans),
        max(s["bbox"][2] for s in spans), max(s["bbox"][3] for s in spans),
    ]
    has_span_mm = all(isinstance(s.get("bbox_mm"), list) for s in spans)
    text_bbox_mm = ([
        min(s["bbox_mm"][0] for s in spans), min(s["bbox_mm"][1] for s in spans),
        max(s["bbox_mm"][2] for s in spans), max(s["bbox_mm"][3] for s in spans),
    ] if has_span_mm else bbox_to_mm(text_bbox))

    if grid.get("table_bbox"):
        table_bbox = list(grid["table_bbox"])
        table_bbox_source = "drawn_border"
        # Prefer the paths' own bbox_mm over converting the snapped points.
        table_bbox_mm = grid.get("table_bbox_mm") or bbox_to_mm(table_bbox)
    else:
        table_bbox = list(text_bbox)
        table_bbox_source = "text_extent"
        table_bbox_mm = text_bbox_mm

    return {
        "page_name": page_meta.get("page_name"),
        "page_number": page_meta.get("page_number"),
        "pairs": pairs,
        "unassigned_spans": unassigned,
        "table_bbox": table_bbox,
        "table_bbox_mm": table_bbox_mm,
        # Which geometry the size above came from, so a consumer never has to
        # guess whether a near-miss is a real defect or a measurement artefact.
        "table_bbox_source": table_bbox_source,
        "text_extent_bbox_mm": text_bbox_mm,
        "table_width_pt": table_bbox[2] - table_bbox[0],
        "table_height_pt": table_bbox[3] - table_bbox[1],
        # Rounded: subtracting two 2-dp millimetre values reintroduces binary
        # float noise, so an exact 48 mm header printed as 47.99999999999999.
        "table_width_mm": round(table_bbox_mm[2] - table_bbox_mm[0], 2) if table_bbox_mm else None,
        "table_height_mm": round(table_bbox_mm[3] - table_bbox_mm[1], 2) if table_bbox_mm else None,
        "left_from_page_left_pt": table_bbox[0],
        "right_from_page_left_pt": table_bbox[2],
        "top_from_page_top_pt": table_bbox[1],
        "bottom_from_page_top_pt": table_bbox[3],
        "left_from_page_left_mm": table_bbox_mm[0] if table_bbox_mm else None,
        "right_from_page_left_mm": table_bbox_mm[2] if table_bbox_mm else None,
        "top_from_page_top_mm": table_bbox_mm[1] if table_bbox_mm else None,
        "bottom_from_page_top_mm": table_bbox_mm[3] if table_bbox_mm else None,
        "row_count": row_count,
        # Left-column row heights, from the drawn cells -- makes a merged row
        # visible as a taller cell instead of an unexplained low row_count.
        "row_heights_mm": row_heights_mm,
        "column_count": len(col_bounds) if col_bounds else None,
        "col_pair_count": math.ceil(len(col_bounds) / 2) if col_bounds else None,
        "border_stroke_pt": grid.get("border_stroke_pt") or _recon_border_stroke_values(all_paths),
        # How the grid was drawn ("rects" per-cell, "lines" ruled, or None when
        # neither was found) and every cell it yielded, so a downstream check can
        # address a specific cell rather than re-deriving the grid.
        "grid_convention": grid.get("convention"),
        "cells_mm": [bbox_to_mm(c) for c in cells] if cells else None,
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


# ═══════════════════════════════════════════════════════════════
# ARTWORK RECONSTRUCTION (the printed component, below the header)
# ═══════════════════════════════════════════════════════════════
#
# Same discipline as the header table: rebuild from drawn geometry, never from
# where text happens to sit. That distinction matters more here than anywhere
# else -- if the component's boundary were derived from its own text, a layout
# shifted 5 mm off-centre would shift the reference frame with it and the defect
# would measure as perfect. The boundary has to come from the dieline.
#
# The dieline IS present in these PDFs (no CDR or separate technical file
# needed). On the Outer Carton it is nine stroked rectangles tiling edge to
# edge -- front/back 72x32, top/bottom 72x20, side panels 20x32, tuck flaps
# 4x32, pasting flap 72x4.4 -- matching the declared 72x20x32 exactly. On the
# Foil it is the 70x30 strip plus a separate 70x30 blister carrying ten 8x8
# cavities.
#
# Two filters are essential, both learned from getting it wrong first:
#   1. Drop annotation-coloured paths. The blue dimension callouts ("72.00 mm")
#      sit OUTSIDE the component and inflated the boundary to
#      [31.99, 107.78, 157.32, 224.30] before they were excluded.
#   2. Drop rects nested inside other rects. Panels tile edge to edge; content
#      boxes (the composition table's rows, the storage box) nest inside a
#      panel and are not panels themselves.

ARTWORK_MIN_PANEL_MM = 3.0     # smaller than this is a tick, a swatch or a cavity
ARTWORK_NEST_EPS_MM = 0.5      # containment slack, in mm
# A dimension callout is a rule or an arrowhead: degenerate in one axis, or
# tiny. Measured on real artwork -- arrowheads are ~7.5 mm2, while the smallest
# printed colour band is 691 mm2, so neither bound is delicate.
ARTWORK_CALLOUT_MAX_THIN_MM = 0.5
ARTWORK_CALLOUT_MAX_AREA_MM2 = 25.0


def _mm_box(path: dict) -> Optional[list]:
    b = path.get("bbox_mm")
    return b if isinstance(b, list) and len(b) >= 4 else None


def _mm_area(b: list) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _contains_mm(outer: list, inner: list) -> bool:
    """True when outer encloses inner and the two are not the same box."""
    if outer is inner:
        return False
    e = ARTWORK_NEST_EPS_MM
    same = all(abs(a - b) <= e for a, b in zip(outer, inner))
    if same:
        return False
    return (outer[0] <= inner[0] + e and outer[1] <= inner[1] + e
            and outer[2] >= inner[2] - e and outer[3] >= inner[3] - e)


def _relative_position(box: list, page_w_mm, page_h_mm,
                       artboard: Optional[list], component: Optional[list]) -> dict:
    """Every edge distance a placement rule might ask for, explicitly named.

    Three frames, because they answer different questions and only the last is
    immune to a whole-layout shift:
      page      -- where it sits on the sheet (sheet size is arbitrary, so this
                   is for reference, not compliance)
      artboard  -- against the text-derived artboard, kept for continuity with
                   the existing AC-reference edge check
      component -- against the DIELINE. This is the one a placement rule wants.
    """
    out = {}
    if page_w_mm is not None and page_h_mm is not None:
        out["page"] = {
            "left_from_page_left_mm": round(box[0], 2),
            "right_from_page_left_mm": round(box[2], 2),
            "top_from_page_top_mm": round(box[1], 2),
            "bottom_from_page_top_mm": round(box[3], 2),
            "right_margin_mm": round(page_w_mm - box[2], 2),
            "bottom_margin_mm": round(page_h_mm - box[3], 2),
        }
    if artboard:
        out["artboard"] = {
            "left_from_artboard_left_mm": round(box[0] - artboard[0], 2),
            "right_from_artboard_right_mm": round(artboard[2] - box[2], 2),
            "top_from_artboard_top_mm": round(box[1] - artboard[1], 2),
            "bottom_from_artboard_bottom_mm": round(artboard[3] - box[3], 2),
        }
    if component:
        cw, ch = component[2] - component[0], component[3] - component[1]
        out["component"] = {
            "left_from_component_left_mm": round(box[0] - component[0], 2),
            "right_from_component_right_mm": round(component[2] - box[2], 2),
            "top_from_component_top_mm": round(box[1] - component[1], 2),
            "bottom_from_component_bottom_mm": round(component[3] - box[3], 2),
            # Signed offset of this box's centre from the component's centre --
            # what a "must be centred" rule reads directly.
            "center_offset_x_mm": round(
                ((box[0] + box[2]) / 2) - (component[0] + cw / 2), 2),
            "center_offset_y_mm": round(
                ((box[1] + box[3]) / 2) - (component[1] + ch / 2), 2),
        }
    return out


PANEL_ROW_EPS_MM = 2.0     # panels whose tops agree this closely share a row
# A vertical whitespace channel at least this wide separates columns WITHIN a
# panel. Measured on Face 3, whose real gutters are 2.49 mm and 8.34 mm while
# the widest gap inside a column is under 1 mm -- so the threshold is not
# delicate, but it is the number to revisit if a panel over- or under-splits.
COLUMN_GUTTER_MM = 2.0


def _split_panel_columns(text_elements: list, inner_boxes: list) -> list:
    """Find content columns inside one panel, from vertical whitespace.

    Projects everything the panel holds onto the x-axis, merges the occupied
    bands, and treats any surviving gap as a column break.

    Inner boxes are projected as well as text, and that is what makes the split
    correct rather than merely plausible: Face 3's barcode is outlined vector
    with no text layer at all, so a text-only projection finds two columns and
    silently loses the barcode column entirely.
    """
    spans = [(e["bbox_mm"][0], e["bbox_mm"][2]) for e in (text_elements or [])
             if isinstance(e.get("bbox_mm"), list)]
    spans += [(b[0], b[2]) for b in (inner_boxes or [])]
    if not spans:
        return []

    merged = []
    for x0, x1 in sorted(spans):
        if merged and x0 <= merged[-1][1] + COLUMN_GUTTER_MM:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    columns = []
    for i, (x0, x1) in enumerate(merged, start=1):
        in_col = [e for e in (text_elements or [])
                  if isinstance(e.get("bbox_mm"), list)
                  and x0 - 0.01 <= (e["bbox_mm"][0] + e["bbox_mm"][2]) / 2 <= x1 + 0.01]
        for e in in_col:
            e["column"] = i
        boxes = [b for b in (inner_boxes or [])
                 if x0 - 0.01 <= (b[0] + b[2]) / 2 <= x1 + 0.01]
        columns.append({
            "column": i,
            "x0_mm": round(x0, 2),
            "x1_mm": round(x1, 2),
            "width_mm": round(x1 - x0, 2),
            "gutter_before_mm": (round(x0 - merged[i - 2][1], 2) if i > 1 else None),
            "text_span_count": len(in_col),
            "inner_box_count": len(boxes),
        })
    return columns


def _address_panels(panels: list, component_bbox: list) -> None:
    """Give every panel a grid address and its offset within its own component.

    Addressed the way the header's cells are -- row/col a reviewer can point at
    -- rather than by raw coordinates. Rows are banded on the top edge because
    panels in a row are flush at the top but may differ in height (a 4mm tuck
    flap sits beside a 32mm side panel).
    """
    tops = sorted({round(p["bbox_mm"][1], 1) for p in panels})
    bands = []
    for t in tops:
        if not bands or t - bands[-1] > PANEL_ROW_EPS_MM:
            bands.append(t)

    for p in panels:
        b = p["bbox_mm"]
        row = min(range(len(bands)), key=lambda i: abs(bands[i] - b[1]))
        p["grid_row"] = row + 1
        p["offset_in_component_mm"] = {
            "from_left_mm": round(b[0] - component_bbox[0], 2),
            "from_top_mm": round(b[1] - component_bbox[1], 2),
        }

    for row in {p["grid_row"] for p in panels}:
        same = sorted((p for p in panels if p["grid_row"] == row),
                      key=lambda p: p["bbox_mm"][0])
        for i, p in enumerate(same):
            p["grid_col"] = i + 1

    panels.sort(key=lambda p: (p["grid_row"], p["grid_col"]))


def _infer_layout(panels: list, declared_mm: Optional[list]) -> dict:
    """Work out what the drawn blank actually IS, from its own geometry.

    A carton's four body faces wrap around the pack, so on the flat blank they
    always form one contiguous run. The AXIS of that run fixes where the pack
    opens, which is exactly what the Style field names:

      run is vertical   -> fold lines horizontal -> closures left/right
                           => "Side Open Side Open"
      run is horizontal -> fold lines vertical   -> closures top/bottom
                           => "Top Open Bottom Paste" / "Top Open Bottom Lock"

    Verified against real artwork of both kinds: AC2491 (72x20x32) stacks two
    72x32 faces vertically with 20x32 side panels and 4x32 tucks to left and
    right; AC2676 (30x30x68) runs four 30x68 faces horizontally with the
    closure flaps above. Derived independently of the header, so it can be
    CHECKED against the Style cell rather than trusting it.

    Paste vs Lock is not decidable from the blank outline alone -- both draw the
    same rectangles -- so the derived value stays "Top Open Bottom *" and the
    caller compares only the part geometry can support.
    """
    out = {"body_face_run_axis": None, "body_face_count": None,
           "derived_style": None, "note": None}
    if len(panels) < 3:
        out["note"] = "too few panels to infer a carton layout"
        return out

    # Body faces are the largest panels; flaps and tucks are strictly smaller.
    areas = [(p["width_mm"] * p["height_mm"], p) for p in panels]
    biggest = max(a for a, _ in areas)
    body = [p for a, p in areas if a >= biggest * 0.45]
    if len(body) < 2:
        out["note"] = "only one large panel; not a wrap-around blank"
        return out

    xs = {round(p["bbox_mm"][0], 1) for p in body}
    ys = {round(p["bbox_mm"][1], 1) for p in body}
    out["body_face_count"] = len(body)

    if len(ys) > len(xs):
        out["body_face_run_axis"] = "vertical"
        out["derived_style"] = "Side Open Side Open"
    elif len(xs) > len(ys):
        out["body_face_run_axis"] = "horizontal"
        out["derived_style"] = "Top Open Bottom *"
    else:
        out["note"] = "body faces do not form a clear run on either axis"
        return out

    if declared_mm and len(declared_mm) >= 3:
        # Cross-check: every body face should measure L or W by H.
        h = max(declared_mm)
        matches = sum(1 for p in body
                      if abs(max(p["width_mm"], p["height_mm"]) - h) <= 1.0)
        out["body_faces_matching_declared_height"] = f"{matches}/{len(body)}"
    return out


# ═══════════════════════════════════════════════════════════════
# LEGEND IDENTIFICATION (SOP section 3)
# ═══════════════════════════════════════════════════════════════
#
# Names each block of copy on the artwork by matching it against the SOP's own
# legend tables (3.1 common legends, 3.3 route of administration, 3.4 warnings).
# This is IDENTIFICATION, not verification: it answers "which legend is this",
# so a later compliance pass can ask "is its wording, placement and alignment
# right". Matching is therefore deliberately lenient -- case- and accent-
# insensitive, prefix-based -- because a block with a typo is still that legend,
# and calling it unknown would hide the very defect worth reporting.

BLOCK_GAP_MM = 2.5     # vertical gap that separates two blocks of copy


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def _legend_key(text: Optional[str]) -> str:
    """Fold a block of copy down to something matchable."""
    folded = _strip_accents((text or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%/:.\s-]", " ", folded)).strip()


# (legend_id, [phrases]) -- a block matches if it STARTS WITH any phrase, or for
# the multi-line ones, contains it. Both language variants are listed because
# the same artwork family ships in English and Spanish.
LEGEND_PATTERNS = [
    ("composition",           ["composicion:", "composition:", "composicion "]),
    ("route_of_administration", ["via de administracion:", "route of administration:"]),
    ("storage",               ["almacenamiento:", "storage:", "conservar en un lugar seco",
                               "store in a dry place"]),
    ("pediatric_warning",     ["producto medicinal, mantengase fuera",
                               "medicinal product, keep out of reach"]),
    ("additional_info",       ["para informacion adicional:", "for additional information:",
                               "forma de preparacion:", "method of preparation:"]),
    ("prescription_note",     ["venta bajo prescripcion medica",
                               "sale under medical prescription"]),
    ("otc_note",              ["producto de venta libre", "over-the-counter product"]),
    ("hospital_use",          ["uso hospitalario", "for hospital use"]),
    # Not keyed on "Código" alone: SOP 2.5 puts only Lot No. / Mfg. Date /
    # Exp. Date in a Foil's stereo-print box, with no Código line at all, so a
    # Código-only pattern missed the batch block on every Foil.
    ("batch_coding",          ["codigo", "code ", "lote no", "lot no",
                               "fecha fab", "mfg. date"]),
    ("registration_numbers",  ["registro sanitario", "registration no"]),
    ("manufacturer_address",  ["fabricado en la india por", "manufactured in india by"]),
    ("route_short",           ["para uso oral", "para uso i.m", "para uso i.v",
                               "para uso topico", "para uso oftalmico", "para uso otico",
                               "para uso nasal", "para uso vaginal", "para uso rectal",
                               "for oral use", "for im use", "for iv use", "for im/iv use",
                               "for topical use", "for ophthalmic use"]),
    ("stereo_print",          ["stereo print"]),
    ("equivalence",           ["eq. a ", "eq. to "]),
    ("reconstitution_note",   ["despues de reconstituido", "discard any remaining portion"]),
    ("tube_closure_note",     ["mantenga el tubo bien cerrado", "keep the tube tightly closed"]),
    ("external_use_note",     ["solo para uso externo", "for external use only"]),
    ("shake_note",            ["agitese antes de usar", "shake before use"]),
]

# SOP 3.4 warnings are keyed by the excipient they name, since the body text
# varies by substance but the leading term does not.
WARNING_TERMS = [
    "aspartame", "tartrazina", "tartrazine", "alcohol bencilico", "bencilic alcohol",
    "benzyl alcohol", "tetraciclinas", "tetracyclines", "acido acetil salicilico",
    "acetil salicilic acid", "acetaminofen", "acetaminophen", "gluten", "lactosa",
    "lactose", "sodio", "sodium", "opio", "opium",
]


# Legends whose text is product-specific, so no fixed phrase can find them.
# Keyed off structure instead, per SOP 1.4.1's construction formula
# [API] [Rhydburg] [Strength] [Dosage Form].
LEGEND_REGEXES = [
    # "Rhydburg" not followed by the company suffix -- that is the address, and
    # it is matched by its own leading phrase. The strength may be a percentage
    # rather than a mass: topicals print "Rhydburg 0.1% + 0.1%", which a
    # mass-only pattern missed on every Mono Carton and Tube.
    ("trade_name", re.compile(
        r"rhydburg(?!\s+(?:pharmaceuticals|llp))\s*[\d.]+\s*(?:mg|mcg|g|ml|iu|%)")),
    ("ac_reference", re.compile(r"\bac\s*\d+\s*v\.?\s*\d+\b")),
    # Countable forms ("30 Tabletas") and weight/volume presentations
    # ("20 g", "10 ml") are both the unit of dosage form.
    ("unit_of_dosage", re.compile(
        r"^\d+\s*(?:tabletas|tablets|capsulas|capsules|viales|vials|sobres|"
        r"sachets|ampollas|ampoules|comprimidos|g|kg)\b")),
    ("container_volume", re.compile(r"^\d+\s*ml\b")),
    # Route wording varies in gender and preposition across artworks
    # ("Para uso tópico" / "tópica"), so match the construction, not a list.
    ("route_short", re.compile(r"\bpara uso [a-z.]+|\bfor [a-z./]+ use\b")),
    # Insert leaflets are prose, not legends. Their numbered section headings
    # are still worth naming so the body of an insert is addressable.
    ("insert_section", re.compile(r"^\d+\.\s+(?:que |como |posibles |contenido |conservacion)")),
    ("shelf_life", re.compile(r"\bvida util\b|\bshelf life\b")),
]


def _identify_legends(text: str) -> list:
    """Every SOP legend present in a block of copy, in the order they appear.

    Returns a LIST because one block legitimately carries more than one legend:
    on Face 1 the unit of dosage form and the route of administration sit
    0.24 mm apart -- closer than lines within a single legend elsewhere -- so no
    gap threshold can separate them, and reporting only the first would silently
    drop the other.
    """
    key = _legend_key(text)
    if not key:
        return []

    hits = []
    for term in WARNING_TERMS:
        for marker in (term + ":", term + " :"):
            idx = key.find(marker)
            if idx != -1:
                hits.append((idx, f"warning_{term.replace(' ', '_')}"))
                break
    for legend_id, phrases in LEGEND_PATTERNS:
        for phrase in phrases:
            # Normalise the PATTERN the same way as the text. Written literally
            # they drift apart: "producto medicinal, mantengase fuera" never
            # matched, because the normaliser turns the comma into a space.
            idx = key.find(_legend_key(phrase))
            if idx != -1:
                hits.append((idx, legend_id))
                break
    for legend_id, pattern in LEGEND_REGEXES:
        m = pattern.search(key)
        if m:
            hits.append((m.start(), legend_id))

    seen, ordered = set(), []
    for _, legend_id in sorted(hits):
        if legend_id not in seen:
            seen.add(legend_id)
            ordered.append(legend_id)
    return ordered


def _group_blocks(text_elements: list, inner_boxes: list) -> list:
    """Group a panel's spans into blocks of copy.

    Uses the drawn boxes where they exist and vertical gaps where they do not,
    because that is how the artwork itself communicates grouping: Face 3's
    left column boxes every legend individually (7 boxes, 7 legends), while its
    middle column has none and relies on spacing -- ~1.3 mm between lines of one
    block against 4.5-6.3 mm between blocks.
    """
    elements = [e for e in (text_elements or []) if isinstance(e.get("bbox_mm"), list)]
    if not elements:
        return []

    blocks, claimed = [], set()
    for box in sorted(inner_boxes or [], key=lambda b: (b[1], b[0])):
        inside = [e for e in elements
                  if id(e) not in claimed
                  and box[0] - 0.5 <= (e["bbox_mm"][0] + e["bbox_mm"][2]) / 2 <= box[2] + 0.5
                  and box[1] - 0.5 <= (e["bbox_mm"][1] + e["bbox_mm"][3]) / 2 <= box[3] + 0.5]
        if inside:
            for e in inside:
                claimed.add(id(e))
            blocks.append((inside, [round(v, 2) for v in box]))

    loose = sorted((e for e in elements if id(e) not in claimed),
                   key=lambda e: (e["bbox_mm"][1], e["bbox_mm"][0]))
    run = []
    for e in loose:
        if run and e["bbox_mm"][1] - run[-1]["bbox_mm"][3] > BLOCK_GAP_MM:
            blocks.append((run, None))
            run = []
        run.append(e)
    if run:
        blocks.append((run, None))

    out = []
    for members, box in blocks:
        ordered = sorted(members, key=lambda e: (e["bbox_mm"][1], e["bbox_mm"][0]))
        text = " ".join((e.get("text") or "").strip() for e in ordered).strip()
        text = re.sub(r"\s+", " ", text)
        bbox = [min(e["bbox_mm"][0] for e in ordered), min(e["bbox_mm"][1] for e in ordered),
                max(e["bbox_mm"][2] for e in ordered), max(e["bbox_mm"][3] for e in ordered)]
        legends = _identify_legends(text)
        legend = legends[0] if legends else None
        for e in ordered:
            e["legend"] = legend
        out.append({
            "legend": legend,
            "legends": legends or None,
            "text": text,
            "bbox_mm": [round(v, 2) for v in bbox],
            "boxed": box is not None,
            "box_mm": box,
            "column": ordered[0].get("column"),
            "span_count": len(ordered),
        })
    out.sort(key=lambda b: (b["bbox_mm"][1], b["bbox_mm"][0]))
    return out


DIM_LINE_EPS_MM = 0.6      # a rule is "at" a coordinate within this
DIM_MATCH_TOL_MM = 0.5     # printed label vs measured geometry


def _resolve_dimension_callouts(annotations: list, callouts: list,
                                components: list,
                                declared_regions: Optional[dict] = None) -> list:
    """Pair each printed dimension label with the geometry it measures.

    An engineering dimension is drawn as two extension lines BRACKETING the
    label, joined by a rule with arrowheads. So the two extension lines nearest
    the label -- one either side -- are what it measures, and the distance
    between them is the true value. That makes the printed text checkable
    against the drawing instead of merely reported: AC2491's "72.00 mm" sits
    between extension lines at x=55.99 and x=127.99, which really are 72.00 mm
    apart.

    Also resolves WHAT is being measured by matching the bracketed span against
    the panels and the blank, so a callout reads "panel 2.1 width" rather than
    a bare pair of coordinates.
    """
    v_lines, h_lines = [], []
    for p in callouts:
        b = _mm_box(p)
        if not b:
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        if w <= DIM_LINE_EPS_MM and h > DIM_LINE_EPS_MM:
            v_lines.append(b)
        elif h <= DIM_LINE_EPS_MM and w > DIM_LINE_EPS_MM:
            h_lines.append(b)

    def describe(span, axis):
        """Name what a bracketed span corresponds to."""
        lo, hi = span
        length = hi - lo
        for ci, comp in enumerate(components, start=1):
            cb = comp["bbox_mm"]
            c_lo, c_hi = (cb[0], cb[2]) if axis == "x" else (cb[1], cb[3])
            if abs(lo - c_lo) <= DIM_MATCH_TOL_MM and abs(hi - c_hi) <= DIM_MATCH_TOL_MM:
                return f"component {ci} full {'width' if axis == 'x' else 'height'}"
            for panel in comp["panels"]:
                pb = panel["bbox_mm"]
                p_lo, p_hi = (pb[0], pb[2]) if axis == "x" else (pb[1], pb[3])
                if abs(lo - p_lo) <= DIM_MATCH_TOL_MM and abs(hi - p_hi) <= DIM_MATCH_TOL_MM:
                    return (f"panel {panel['grid_row']}.{panel['grid_col']} "
                            f"{'width' if axis == 'x' else 'height'}")
        # Not every dimension brackets a drawn edge. A Foil's Printing Zone and
        # Repeat measure regions INSIDE the strip, so they match no panel -- but
        # they are declared in the header, which names them exactly.
        for field, value in (declared_regions or {}).items():
            if value is not None and abs(length - value) <= DIM_MATCH_TOL_MM:
                return f"declared {field.replace('_', ' ')}"
        return f"unmatched span of {length:.2f} mm"

    results = []
    for ann in (annotations or []):
        text = (ann.get("text") or "").strip()
        box = ann.get("bbox_mm")
        if not box or len(box) < 4:
            continue
        numbers = DIMENSION_NUMBER_RE.findall(text)
        if not numbers:
            continue
        declared = float(numbers[0])
        tw, th = box[2] - box[0], box[3] - box[1]

        if th > tw:      # label set vertically -> it measures a height
            axis, cx = "y", (box[1] + box[3]) / 2
            rules = [ln for ln in h_lines
                     if ln[0] - DIM_LINE_EPS_MM <= (box[0] + box[2]) / 2 <= ln[2] + DIM_LINE_EPS_MM]
            coords = sorted({round(ln[1], 2) for ln in rules})
        else:
            axis, cx = "x", (box[0] + box[2]) / 2
            rules = [ln for ln in v_lines
                     if ln[1] - DIM_LINE_EPS_MM <= (box[1] + box[3]) / 2 <= ln[3] + DIM_LINE_EPS_MM]
            coords = sorted({round(ln[0], 2) for ln in rules})

        before = [c for c in coords if c <= cx]
        after = [c for c in coords if c >= cx]
        entry = {
            "label": text,
            "declared_mm": declared,
            "orientation": "vertical" if axis == "y" else "horizontal",
            "label_bbox_mm": [round(v, 2) for v in box],
        }
        if before and after:
            lo, hi = max(before), min(after)
            measured = round(hi - lo, 2)
            entry.update({
                "measured_mm": measured,
                "span_mm": [lo, hi],
                "matches_label": abs(measured - declared) <= DIM_MATCH_TOL_MM,
                "measures": describe((lo, hi), axis),
            })
        else:
            # Reported, not dropped: a callout whose extension lines cannot be
            # found is a thing a reviewer should see, not a silent omission.
            entry.update({"measured_mm": None, "span_mm": None,
                          "matches_label": None,
                          "measures": "extension lines not found"})
        results.append(entry)

    results.sort(key=lambda e: (e["orientation"], e["label_bbox_mm"][1]))
    return results


def _layout_with_style_check(layout: dict, declared_style: Optional[str]) -> dict:
    """Compare the geometrically-derived layout against the header's Style cell.

    Reported as three separate facts -- what the blank is drawn as, what the
    header claims, and whether they agree -- so a disagreement points at which
    side is wrong instead of collapsing to a bare pass/fail. Comparison is on
    the opening axis only, since Paste vs Lock is not visible in the outline.
    """
    layout = dict(layout)
    layout["declared_style"] = declared_style
    derived = layout.get("derived_style")
    if not derived or not declared_style:
        layout["style_agrees"] = None
        return layout

    def axis_of(text: str) -> Optional[str]:
        t = re.sub(r"[^a-z ]", " ", (text or "").lower())
        if "side open" in t:
            return "side"
        if "top open" in t:
            return "top"
        return None

    d_axis, h_axis = axis_of(derived), axis_of(declared_style)
    layout["style_agrees"] = None if h_axis is None else (d_axis == h_axis)
    if layout["style_agrees"] is False:
        layout["note"] = (f"blank is drawn as {derived!r} but the header Style "
                          f"cell says {declared_style!r}")
    return layout


def build_artwork_reconstruction(body_spans: list, paths: list, page_meta: dict,
                                 header_bottom_mm: Optional[float]) -> dict:
    """Rebuild the printed component below the header from its drawn geometry."""
    page_meta = page_meta or {}
    page_w_mm = page_meta.get("page_width_mm")
    page_h_mm = page_meta.get("page_height_mm")
    artboard = page_meta.get("artboard_bounds_mm")

    below = []
    for p in (paths or []):
        b = _mm_box(p)
        if not b:
            continue
        # A path belongs to the header if it lies ENTIRELY within the header
        # band. Testing the top edge alone (b[1] < bottom) left the header's own
        # closing rule behind: it is a zero-height line sitting exactly ON the
        # boundary, so it is not strictly above it and leaked into the artwork
        # as a phantom 180mm element.
        if (header_bottom_mm is not None
                and b[3] <= header_bottom_mm + ARTWORK_NEST_EPS_MM):
            continue
        below.append(p)

    # Callouts are excluded from the geometry but reported, so a dimension
    # cross-check can still read them and nobody wonders where they went.
    #
    # Colour alone cannot decide this. ANNOTATION_COLORS was built for dimension
    # TEXT and, at tolerance 24, the brand's Pantone Process Blue C fill
    # (#008ccc) matches the annotation blue #0091d2 -- which filed the carton's
    # blue band as a callout and removed the single element SOP 2.3.x's
    # "70% white / 30% blue" rule depends on.
    #
    # Role separates them cleanly: a callout is a rule or an arrowhead, so it is
    # either degenerate in one axis or tiny (arrowheads measure ~7.5 mm2). A
    # printed colour band is a substantial filled area (the blue band is
    # 691 mm2). Two orders of magnitude apart, so the threshold is not delicate.
    callouts, structural = [], []
    for p in below:
        b = _mm_box(p)
        w, h = b[2] - b[0], b[3] - b[1]
        colour = p.get("stroke_color_hex") or p.get("fill_color_hex") or ""
        is_marker = (min(w, h) < ARTWORK_CALLOUT_MAX_THIN_MM
                     or (w * h) <= ARTWORK_CALLOUT_MAX_AREA_MM2)
        (callouts if is_annotation_color(colour) and is_marker
         else structural).append(p)

    # Every structural box, whatever its size -- the composition table's thinner
    # rows (2.58-2.97 mm tall) are real elements and must stay reportable. Only
    # PANEL candidacy needs a size floor, so the two lists are kept apart; an
    # earlier version filtered both together and silently lost those rows.
    structural_boxes = [_mm_box(p) for p in structural]
    panel_candidates = [p for p in structural
                        if (_mm_box(p)[2] - _mm_box(p)[0]) >= ARTWORK_MIN_PANEL_MM
                        and (_mm_box(p)[3] - _mm_box(p)[1]) >= ARTWORK_MIN_PANEL_MM]

    cand_mm = [_mm_box(p) for p in panel_candidates]
    panel_paths = [p for p, b in zip(panel_candidates, cand_mm)
                   if not any(_contains_mm(o, b) for o in cand_mm)]
    panel_paths.sort(key=lambda p: (_mm_box(p)[1], _mm_box(p)[0]))

    # Group panels into connected components. A carton's panels TILE -- they
    # share edges, so they are one flat blank. A Foil sheet instead carries two
    # disjoint drawings, the 70x30 strip and the 70x30 blister sitting 61 mm
    # below it; unioning those produced a meaningless 71.58 x 91.48 "component"
    # matching nothing in the header. Adjacency separates the two cases without
    # needing to know the component type.
    boxes_mm = [_mm_box(p) for p in panel_paths]

    def touching(a: list, b: list) -> bool:
        gap = ARTWORK_NEST_EPS_MM
        return (a[0] - gap <= b[2] and b[0] - gap <= a[2]
                and a[1] - gap <= b[3] and b[1] - gap <= a[3])

    group_of = [None] * len(panel_paths)
    groups = []
    for i in range(len(panel_paths)):
        if group_of[i] is not None:
            continue
        stack, members = [i], []
        group_of[i] = len(groups)
        while stack:
            cur = stack.pop()
            members.append(cur)
            for j in range(len(panel_paths)):
                if group_of[j] is None and touching(boxes_mm[cur], boxes_mm[j]):
                    group_of[j] = group_of[i]
                    stack.append(j)
        groups.append(sorted(members))

    group_bbox = []
    for members in groups:
        bs = [boxes_mm[m] for m in members]
        group_bbox.append([min(b[0] for b in bs), min(b[1] for b in bs),
                           max(b[2] for b in bs), max(b[3] for b in bs)])

    def build_panel(idx: int) -> dict:
        p, b = panel_paths[idx], boxes_mm[idx]
        own_component = group_bbox[group_of[idx]]
        inner = [o for o in structural_boxes if _contains_mm(b, o)]
        spans_in = []
        for s in (body_spans or []):
            sb = s.get("bbox_mm")
            if not isinstance(sb, list) or len(sb) < 4:
                continue
            cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
            if (b[0] <= cx <= b[2]) and (b[1] <= cy <= b[3]):
                spans_in.append(s)
        inner_rounded = [[round(v, 2) for v in i] for i in inner]
        text_elements = [{
            "text": s.get("text"),
            "bbox_mm": s.get("bbox_mm"),
            "rotation_deg": s.get("rotation_deg"),
            "font_name": s.get("font_name"),
            "font_size_pt": s.get("font_size_pt"),
            # Unsnapped size as well: font_size_pt is snapped for grouping and
            # display, so a 2.75 pt span reads as 3.0 and would silently satisfy
            # a "minimum 3 pt" rule. Any minimum-size check must use this one.
            "font_size_pt_raw": s.get("font_size_pt_raw"),
            "is_bold": s.get("is_bold"),
            "is_italic": s.get("is_italic"),
            "color_hex": s.get("color_hex"),
            "alignment": s.get("alignment"),
        } for s in spans_in]
        # Tags each element with its column as a side effect, so an element
        # carries its own address rather than the caller re-deriving it.
        columns = _split_panel_columns(text_elements, inner_rounded)
        # Tags each element with its legend as a side effect, same as columns.
        blocks = _group_blocks(text_elements, inner_rounded)

        return {
            "bbox_mm": [round(v, 2) for v in b],
            "width_mm": round(b[2] - b[0], 2),
            "height_mm": round(b[3] - b[1], 2),
            "stroke_width_pt": p.get("stroke_width_pt"),
            "fill_color_hex": p.get("fill_color_hex"),
            # Measured against this panel's OWN component group, not a union of
            # unrelated drawings.
            "position": _relative_position(b, page_w_mm, page_h_mm, artboard, own_component),
            "inner_box_count": len(inner),
            "inner_boxes_mm": inner_rounded or None,
            # Content columns within this panel, found from vertical whitespace.
            # Face 3 splits into composition | batch-coding+registration |
            # barcode; a single-content panel reports one column.
            "column_count": len(columns),
            "columns": columns or None,
            # Blocks of copy, each named against the SOP's legend tables where
            # it could be identified. This is what turns "column 2 holds 15
            # spans" into "column 2 holds the batch-coding block and the
            # registration numbers".
            "block_count": len(blocks),
            "blocks": blocks or None,
            "unidentified_block_count": sum(1 for b in blocks if b["legend"] is None),
            "text_span_count": len(spans_in),
            # Text WITH its geometry, not bare strings. The alignment rules live
            # inside a panel -- the batch-coding colons and the composition
            # claims are both in Face 3 -- so a check reading this panel needs
            # the anchors here rather than re-deriving which spans fall inside
            # which panel from sections.body. Each element also carries its
            # own "column".
            "text_elements": text_elements or None,
        }

    components = []
    for gi, members in enumerate(groups):
        cb = group_bbox[gi]
        panels = [build_panel(m) for m in members]
        _address_panels(panels, cb)
        components.append({
            "bbox_mm": [round(v, 2) for v in cb],
            "width_mm": round(cb[2] - cb[0], 2),
            "height_mm": round(cb[3] - cb[1], 2),
            # The whole component's own placement, in the same three frames the
            # panels use -- "where does the artwork sit on the sheet" is a
            # different question from "where does this face sit in the artwork",
            # and both get asked.
            "position": _relative_position(cb, page_w_mm, page_h_mm, artboard, None),
            "layout": _layout_with_style_check(
                _infer_layout(panels, page_meta.get("declared_dimensions_mm")),
                page_meta.get("declared_style")),
            "panel_count": len(members),
            "panels": panels,
        })
    components.sort(key=lambda c: (-(c["width_mm"] * c["height_mm"]), c["bbox_mm"][1]))

    placed = [p["bbox_mm"] for c in components for p in c["panels"]]
    unplaced = []
    for s in (body_spans or []):
        sb = s.get("bbox_mm")
        if not isinstance(sb, list) or len(sb) < 4:
            continue
        cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
        if not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in placed):
            unplaced.append(s.get("text"))

    primary = components[0] if components else None
    return {
        "page_name": page_meta.get("page_name"),
        "page_number": page_meta.get("page_number"),
        # Each connected group of dieline panels is one drawn component. The
        # largest is reported as the primary one, but the list is authoritative:
        # a Foil sheet legitimately has two (strip and blister).
        "component_count": len(components),
        "components": components,
        "primary_component_bbox_mm": primary["bbox_mm"] if primary else None,
        "primary_component_width_mm": primary["width_mm"] if primary else None,
        "primary_component_height_mm": primary["height_mm"] if primary else None,
        "boundary_source": "dieline_paths" if components else None,
        "panel_count": sum(c["panel_count"] for c in components),
        "callout_path_count": len(callouts),
        # Each printed dimension label paired with the geometry it brackets,
        # so the callout can be CHECKED against the drawing rather than just
        # transcribed.
        "dimension_callouts": _resolve_dimension_callouts(
            page_meta.get("annotations"), callouts, components,
            page_meta.get("declared_regions_mm")) or None,
        "unplaced_texts": unplaced or None,
        "page_width_mm": page_w_mm,
        "page_height_mm": page_h_mm,
        "artboard_bounds_mm": artboard,
    }


# A page with no text and no fonts needs enough drawn geometry to be a real
# component rather than a stray rule or crop mark. The genuine drawing sheets
# in the corpus sit at 79 paths; the outlined components at 172, 281 and 603.
OUTLINED_TEXT_MIN_PATHS = int(os.getenv("OUTLINED_TEXT_MIN_PATHS", "120"))


def classify_page_kind(sections: dict, paths: list, has_header_grid: bool,
                       font_count: int = -1) -> str:
    """What kind of page this is, so consumers can skip what does not apply.

    Keyed on the presence of a DRAWN header table, because SOP 2.1 makes that
    the defining feature: "Every artwork component must include a tabular
    header". No table therefore means no component -- an illustration or an
    engineering drawing.

    An earlier version keyed on "carries no text at all", which happened to work
    on the 10 ml vial sheets (0 text spans, 79 paths) but would have
    misclassified any illustration that carried so much as a dimension label.
    The grid test separates the real cases cleanly: the vial page recovers no
    grid at all, while AC3146's insert pages recover 43 and 18 cells and are
    genuinely components despite pairing badly.

    The SOP has nothing to say about drawing pages -- its sections run header,
    cartons, labels, foil, insert, legends, with no drawing section anywhere --
    so they are labelled and deliberately left alone rather than measured
    against rules that do not exist for them.
    """
    if has_header_grid:
        return "component"
    text_count = sum(len(sections.get(b) or []) for b in
                     ("header_table", "body", "annotations",
                      "annotation_near_misses", "production_marks"))
    if not paths and not text_count:
        return "blank"

    # A page carrying NO text and NO fonts, yet plenty of drawn geometry, is not
    # a drawing -- it is a component whose text was converted to outlines before
    # the PDF was saved. The distinction matters enormously downstream:
    # "technical_drawing" means no rule applies, while this means every rule
    # applies and none of them could be read. Reporting the first for the second
    # silently passed a Mono Carton and a Tube with zero checks (exec 87213).
    #
    # font_count is the discriminator, not the path count: a real dimension
    # sheet still labels its dimensions, and those labels carry a font.
    if text_count == 0 and font_count == 0 and len(paths) >= OUTLINED_TEXT_MIN_PATHS:
        return "outlined_text"

    # No tabular header: not a printed component. Distinguished only for
    # reporting -- both are left alone.
    return "technical_drawing" if text_count == 0 else "illustration"


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

    return {
        "name": detected_name,
        # "component" | "technical_drawing" | "unclassified" | "blank".
        # A technical drawing is labelled and otherwise left alone: the SOP
        # specifies nothing for it, so there is nothing to check it against.
        # Provisional: set properly in extract_artwork once the header grid is
        # known, since the drawn table is what decides this.
        "page_kind": None,
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
        # Canonical ACnnnV.n, synthesised once so no consumer re-derives it.
        # Every section is passed, not just body: the on-pack instance is on the
        # pack regardless of which bucket the header band sorted it into.
        "ac_reference": synthesize_ac_reference(
            sections["header_table"], body_text_all,
            all_spans=(sections["body"] + sections["header_table"]
                       + sections["annotations"]
                       + sections.get("annotation_near_misses", [])),
            page_w_mm=pt_to_mm(width_pt), page_h_mm=pt_to_mm(height_pt),
        ),
        # Machine-readable barcode values, decoded from the drawn bars. The
        # printed caption is outlined vector, so this is the only way to read
        # the graphic's own value -- see decode_barcodes for why not OCR.
        "barcodes": decode_barcodes(raw_drawings),
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
        # Fonts declared in the page's own resources. A page cannot draw a glyph
        # without a font unless that glyph has been converted to a path, so zero
        # fonts on a page that still renders ink means the text is OUTLINED --
        # see classify_page_kind(). Measured on Art-CommercialPDF-14635--v3:
        # pages 1-3 have 0 fonts and 0 characters while rendering MORE ink than
        # page 4, which has 2 fonts and reads normally.
        "font_count": len(page.get_fonts()),
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
    reconstructed = {}
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

        # Reconstructions rebuilt purely from this page's own extracted
        # geometry live in their own top-level object, separate from the raw
        # per-page extraction -- header today, artwork-level reconstruction
        # to follow, so both land in the same place instead of each being
        # buried inside its own page's dict.
        if "sections" in page_data:
            on_pack_position = (page_data.get("ac_reference") or {}).get("on_pack_position") or {}
            dims = page_data.get("page_dimensions") or {}
            meta = {
                "page_name": page_data.get("name"),
                "page_number": page_index + 1,
                "page_width_mm": dims.get("width_mm"),
                "page_height_mm": dims.get("height_mm"),
                "artboard_bounds_mm": on_pack_position.get("artboard_bounds_mm"),
                "ac_reference_computed": page_data.get("ac_reference"),
                # Dimension callouts are filed as annotations at extraction
                # time; the artwork pass needs them to pair each label with the
                # geometry it measures.
                "annotations": page_data["sections"].get("annotations", []),
            }
            header = build_header_reconstruction(
                page_data["sections"]["header_table"], page_data.get("paths", []), meta,
            )
            # Feed the header's own declared Style and size into the artwork
            # pass so the drawn blank can be CHECKED against them. The artwork
            # side derives its layout from geometry alone and never reads these
            # to decide anything -- otherwise the comparison would be circular.
            hdr_fields = {p["canonical"]: (p["value"]["text"] if p.get("value") else None)
                          for p in (header.get("pairs") or [])}
            meta["declared_style"] = hdr_fields.get("style")
            meta["declared_dimensions_mm"] = _dimension_numbers(hdr_fields.get("dimension")) or None
            # Header-declared regions that are NOT panel edges (a Foil's
            # Printing Zone and Repeat sit inside the strip), so a dimension
            # callout over one of them can still be named rather than reported
            # as an unmatched span.
            meta["declared_regions_mm"] = {
                field: (_dimension_numbers(hdr_fields.get(field)) or [None])[0]
                for field in ("printing_zone", "repeat")
            }
            # Split the page at the header's own drawn bottom edge where we have
            # it, so the artwork pass never sees the header's grid lines. The
            # 20%-of-height fallback mirrors HEADER_FALLBACK_RATIO, which is what
            # the span split itself used when content detection failed.
            header_bottom_mm = None
            if header.get("table_bbox_mm"):
                header_bottom_mm = header["table_bbox_mm"][3]
            elif dims.get("height_mm") is not None:
                header_bottom_mm = dims["height_mm"] * HEADER_FALLBACK_RATIO

            # SOP 2.1 makes the drawn tabular header the mark of a component,
            # so classification waits until the grid has been recovered.
            page_data["page_kind"] = classify_page_kind(
                page_data["sections"], page_data.get("paths", []),
                bool(header.get("grid_convention")),
                page_data.get("font_count", -1))

            reconstructed[config["key"]] = {
                "header": header,
                "artwork": build_artwork_reconstruction(
                    page_data["sections"].get("body", []), page_data.get("paths", []),
                    meta, header_bottom_mm,
                ),
            }

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
    # Kept out of the per-page dicts and out of the aggregation loops above --
    # this is derived data, not raw extraction, and grouping every
    # reconstruction (header now, artwork later) under one key means a
    # consumer reads one place for "what did we rebuild" instead of one key
    # per page per reconstruction type.
    result["reconstructed"] = reconstructed
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
        # Run in threadpool to avoid blocking the async event loop for large
        # PDFs, but only EXTRACT_CONCURRENCY at a time so peak memory stays a
        # property of the code rather than of the traffic.
        async with _extraction_slot():
            result = await run_in_threadpool(
                extract_artwork, pdf_bytes, file.filename, render, render_dpi
            )
        del pdf_bytes
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
        # Rasterising is the heaviest thing here -- a 600 dpi A4 pixmap is a
        # ~100 MB C-side buffer -- so it shares the same bound as /extract.
        async with _extraction_slot():
            result = await run_in_threadpool(_render_all_pages, pdf_bytes, dpi)
        del pdf_bytes
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
    text_dict = page.get_text("rawdict", flags=TEXT_FLAGS)
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
        # Reports still rendering when this probe ran. The workflow wakes the
        # instance before every report, so a non-zero value here is a direct
        # record of runs overlapping -- which is what made the shared browser
        # get closed out from under a render before _release_browser was
        # refcounted.
        "reports_in_flight": _browser_users,
        "ligature_recovery": cid_flag_status(),
        **_memory_snapshot(),
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


# Chromium flags chosen for a small container: the default /dev/shm on Render is
# 64 MB, and Chromium falls back to it for shared memory unless told otherwise,
# which is a classic source of both crashes and inflated RSS.
_CHROMIUM_ARGS = [
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--disable-extensions",
    "--disable-background-networking",
    "--no-first-run",
]

# Whether to keep Chromium resident between report requests. Off by default:
# holding it costs 150-350 MB for the whole process life, and measured peak
# under a 40-request extraction burst is already ~179 MB -- together those can
# breach a 512 MB instance. Report generation happens once per review, so
# paying ~1-2 s to relaunch is a far better trade than a crash that takes every
# concurrent execution down with it. Set KEEP_BROWSER_WARM=1 if report latency
# ever matters more than headroom.
KEEP_BROWSER_WARM = os.getenv("KEEP_BROWSER_WARM", "0") == "1"

# ─── Report rendering bound ──────────────────────────────────
# Chromium is the most expensive thing this service can hold, so only
# PDF_CONCURRENCY reports render at a time and the rest WAIT -- same trade as
# EXTRACT_CONCURRENCY above. Default 1: report generation runs a few dozen
# times a day and takes seconds, so serialising costs nothing while keeping
# peak to one browser with one tab.
PDF_CONCURRENCY = int(os.getenv("PDF_CONCURRENCY", "1"))
_pdf_slots: Optional[Any] = None   # (loop, asyncio.Semaphore)

# How long a single report may take inside Chromium. Playwright's default is
# 30 s, which is a bound on the MACHINE, not on the work: the same report that
# lays out in ~2 s on a dev box exceeded 30 s on a throttled shared CPU and
# failed with "Page.set_content: Timeout 30000ms exceeded" -- with nothing
# external to fetch, so it was pure layout time. Report size grows with the
# artwork, so this is set high enough that a slow instance finishes rather than
# a large report being the thing that breaks.
#
# This is a backstop against a genuine hang, not a latency target: reports are
# generated a few dozen times a day and PDF_CONCURRENCY=1 already serialises
# them, so a long ceiling costs nothing when things are healthy.
PDF_TIMEOUT_MS = int(os.getenv("PDF_TIMEOUT_MS", "180000"))

# A report must clear a LOWER bar than an extraction, because what comes next is
# a Chromium launch worth 150-350 MB, not another ~90 MB document. Admitting at
# the extraction water mark (75%) leaves ~128 MB of a 512 MB cap for a browser
# that needs multiples of that, which is precisely the OOM being fixed.
RENDER_HIGH_WATER = float(os.getenv("RENDER_HIGH_WATER", "0.45"))


def _pdf_slot():
    """Report-rendering semaphore for the CURRENT event loop (see _slots)."""
    global _pdf_slots
    loop = asyncio.get_running_loop()
    if _pdf_slots is None or _pdf_slots[0] is not loop:
        _pdf_slots = (loop, asyncio.Semaphore(PDF_CONCURRENCY))
    return _pdf_slots[1]


_browser_lock = asyncio.Lock()
_browser_users = 0   # in-flight /html-to-pdf requests holding _browser


async def _get_browser():
    """Launch Chromium on first use, not at startup.

    Measured: the extraction path holds ~93 MB steady with no leak across 45
    consecutive documents, but Chromium adds 150-350 MB the moment it starts.
    Launching it eagerly meant every instance paid that for the whole process
    life even on runs that only ever called /extract -- which is what pushed a
    512 MB instance over. Only /html-to-pdf needs a browser, so only
    /html-to-pdf pays for one.

    Every successful call MUST be paired with exactly one _release_browser():
    the browser is shared, so it is refcounted and torn down only once the last
    in-flight report is finished with it.
    """
    global _playwright, _browser, _browser_users
    async with _browser_lock:
        # Everything runs under the lock, with no unlocked fast path. A fast
        # path could hand back a browser that a finishing request is closing in
        # _release_browser right now, and the caller's next new_page() would
        # then die with "Target page, context or browser has been closed".
        if _browser is None or not _browser.is_connected():
            if _playwright is None:
                _playwright = await async_playwright().start()
            logger.info("Launching headless Chromium for /html-to-pdf")
            _browser = await _playwright.chromium.launch(args=_CHROMIUM_ARGS)
        _browser_users += 1
        return _browser


async def _wait_for_render_headroom() -> None:
    """Hold a report until there is room to launch Chromium.

    Deliberately NOT _wait_for_headroom(): that one returns immediately when
    _inflight is 0, because for an extraction there would be nothing to wait
    for. A report is the opposite case -- the memory in the way is usually left
    over from extractions that have already finished, so _inflight is 0 and the
    gate would wave it straight through into an OOM. Measured on the live
    service at exactly that moment: 400 MB of a 512 MB cap, reports_in_flight 0.

    So: trim first, and if that is not enough, wait and keep trimming. On
    timeout it proceeds anyway -- the same trade as the extraction gate, since
    refusing to render turns a slow report into a permanently broken one.
    """
    if MEMORY_LIMIT_BYTES is None:
        return
    _trim_heap()
    ceiling = MEMORY_LIMIT_BYTES * RENDER_HIGH_WATER
    loop = asyncio.get_running_loop()
    deadline = loop.time() + MEMORY_WAIT_TIMEOUT_S
    waited = False
    while True:
        used = _current_memory()
        if used is None or used < ceiling:
            if waited:
                logger.info("Memory down to %.0f MB; starting report render",
                            (used or 0) / 1048576)
            return
        if loop.time() >= deadline:
            logger.warning(
                "Memory still at %.0f MB of %.0f MB after %.0fs; rendering anyway",
                used / 1048576, MEMORY_LIMIT_BYTES / 1048576, MEMORY_WAIT_TIMEOUT_S)
            return
        if not waited:
            logger.info("Memory at %.0f MB of %.0f MB limit; holding report render",
                        used / 1048576, MEMORY_LIMIT_BYTES / 1048576)
            waited = True
        await asyncio.sleep(MEMORY_POLL_S)
        _trim_heap()


async def _release_browser():
    """Drop one claim on Chromium; shut it down once nobody holds it.

    The refcount is what makes concurrent reports safe. The browser is global,
    so a request finishing its PDF used to close it out from under every other
    request still rendering, which then failed at new_page() with "Target page,
    context or browser has been closed". Overlapping runs are the norm here --
    the schedule fires several a minute and each execution lasts minutes -- so
    the LAST one out turns off the lights, not the first.
    """
    global _playwright, _browser, _browser_users
    async with _browser_lock:
        if _browser_users > 0:
            _browser_users -= 1
        if KEEP_BROWSER_WARM or _browser_users > 0 or _browser is None:
            return
        try:
            await _browser.close()
        except Exception:
            logger.warning("Could not close Chromium cleanly")
        finally:
            _browser = None
        # Stop the driver too. async_playwright().start() spawns a Node process
        # that outlives every browser it launches, so closing only the browser
        # left it resident for the life of the container -- part of the ~309 MB
        # measured with nothing in flight. _get_browser() restarts it on demand.
        if _playwright is not None:
            try:
                await _playwright.stop()
            except Exception:
                logger.warning("Could not stop the Playwright driver cleanly")
            finally:
                _playwright = None
        _trim_heap()
        logger.info("Released headless Chromium and the Playwright driver")


@app.on_event("startup")
async def startup_banner():
    log_ocr_status()
    logger.info("Extraction concurrency limit: %d | report concurrency limit: %d "
                "| keep browser warm: %s",
                EXTRACT_CONCURRENCY, PDF_CONCURRENCY, KEEP_BROWSER_WARM)
    st = cid_flag_status()
    if st["resolved_by_name"]:
        logger.info("Ligature recovery: %s = %d on PyMuPDF %s",
                    st["cid_flag_name"], st["cid_flag_value"], st["pymupdf_version"])
    else:
        logger.warning(
            "Ligature recovery: PyMuPDF %s exports neither TEXT_USE_CID_FOR_UNKNOWN_UNICODE "
            "nor TEXT_CID_FOR_UNKNOWN_UNICODE; falling back to the literal bit 0x80. If "
            "extraction still yields U+FFFD, this build does not support the flag.",
            st["pymupdf_version"])


@app.on_event("shutdown")
async def shutdown_browser():
    global _playwright, _browser, _browser_users
    if _browser:
        await _browser.close()
        _browser = None
    _browser_users = 0
    if _playwright:
        await _playwright.stop()
        _playwright = None


@app.post("/html-to-pdf")
async def html_to_pdf(request: Request):
    """Convert HTML body to PDF using headless Chromium."""
    try:
        data = await request.json()
        html_string = data.get("html", "")
        if not html_string:
            raise HTTPException(status_code=400, detail="Missing 'html' field in request body")
        # The slot is held until BOTH the tab and the browser claim are handed
        # back, so the next report starts from a clean at-most-one-tab state
        # instead of overlapping this one's teardown.
        async with _pdf_slot():
            # Inside the slot, not outside: waiting for headroom while another
            # report still holds Chromium would be waiting for memory that
            # cannot be freed until that one finishes.
            await _wait_for_render_headroom()
            page = None
            # Outside the try below on purpose: if the launch itself fails there
            # is no claim to release, and releasing one would unbalance the
            # refcount for whoever else is rendering.
            browser = await _get_browser()
            try:
                page = await browser.new_page()
                # Covers every operation on this tab, including the internal
                # protocol wait inside page.pdf(). That call takes NO timeout
                # argument of its own -- passing one raises "Page.pdf() got an
                # unexpected keyword argument 'timeout'" and fails the render
                # outright (exec 87042), so the page default is the only way to
                # bound it.
                page.set_default_timeout(PDF_TIMEOUT_MS)
                # "networkidle" waits on external resources and times out on HTML
                # that references anything unreachable. Combined with the
                # un-guarded close below, every such timeout used to strand an
                # open tab holding tens of MB -- a handful of failed report runs
                # was enough to exhaust the box.
                #
                # "domcontentloaded", not "load": the report embeds everything --
                # measured on the real payload of exec 86874, zero <img>, zero
                # <link>, zero @font-face, zero external URLs -- so there are no
                # subresources for "load" to add, only a longer wait.
                #
                # The explicit timeout is the actual fix for that execution.
                # Playwright defaults to 30 s and the call was left on the
                # default, so a 290 KB report (731 rows, 75 tables) laying out on
                # a throttled shared CPU died with "Timeout 30000ms exceeded"
                # while doing nothing wrong. Layout time scales with the report,
                # which grows with the artwork -- so the bound has to be one a
                # big report can actually meet, not one that fits a small one.
                await page.set_content(
                    html_string, wait_until="domcontentloaded", timeout=PDF_TIMEOUT_MS)
                pdf_bytes = await page.pdf(
                    format="A4",
                    print_background=True,
                    margin={"top": "15mm", "bottom": "15mm", "left": "10mm", "right": "10mm"},
                    # No timeout= here: page.pdf() does not accept one. It is
                    # bounded by set_default_timeout() above instead.
                )
            finally:
                # Must run on the failure paths too -- that is the whole point.
                if page is not None:
                    try:
                        await page.close()
                    except Exception:
                        logger.warning("Could not close Chromium page after PDF generation")
                await _release_browser()
        # Streaming happens after the slot is free: the bytes are already fully
        # in memory, so holding the slot across the response would only make
        # queued reports wait on the client's download.
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=compliance_report.pdf"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")