"""
Artwork Compliance Extraction FastAPI Application v2.0
======================================================
Enhanced: zone detection, rotation, ligature repair, annotation filtering.
"""

import fitz  # PyMuPDF
import re
import math
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Any, Optional
from xhtml2pdf import pisa

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
SKIP_PAGES = [4]

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

def repair_text(text: str) -> tuple:
    """Repair ligature corruption. Returns (repaired, was_repaired, repairs)."""
    if '\ufffd' not in text and '\xad' not in text and '\ufb01' not in text and '\ufb02' not in text:
        return text, False, []

    repairs = []
    repaired = text

    # 1. Known word corrections (case-sensitive)
    for broken, fixed in KNOWN_CORRECTIONS.items():
        if broken in repaired:
            repaired = repaired.replace(broken, fixed)
            repairs.append({"from": broken, "to": fixed, "type": "known_word"})

    # 2. Remaining U+FFFD → assume "ti" ligature (dominant in pharma fonts)
    if '\ufffd' in repaired:
        repaired = repaired.replace('\ufffd', 'ti')
        repairs.append({"from": "U+FFFD", "to": "ti", "type": "ligature_ti"})

    # 3. Soft hyphens — only replace with 'ti' when between letters (ligature artifact)
    if '\xad' in repaired:
        # Only replace \xad when it sits between two word characters (ligature gap)
        new_text = re.sub(r'(?<=\w)\xad(?=\w)', 'ti', repaired)
        if new_text != repaired:
            repairs.append({"from": "U+00AD", "to": "ti", "type": "soft_hyphen_ligature"})
            repaired = new_text
        else:
            # Standalone soft hyphen — just remove it
            repaired = repaired.replace('\xad', '')
            repairs.append({"from": "U+00AD", "to": "", "type": "soft_hyphen_removed"})

    # 4. Standard ligature chars
    for lig, rep in {'\ufb01': 'fi', '\ufb02': 'fl'}.items():
        if lig in repaired:
            repaired = repaired.replace(lig, rep)
            repairs.append({"from": f"U+{ord(lig):04X}", "to": rep, "type": "ligature"})

    return repaired, len(repairs) > 0, repairs


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

def extract_text_spans_enhanced(page: fitz.Page, header_threshold: float,
                                 is_insert: bool) -> dict:
    """Extract text with rotation, repair, and annotation filtering."""
    header_table = []
    body = []
    annotations = []

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

                repaired_text, was_repaired, repair_log = repair_text(raw_text)

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

                span_data = {
                    "text": repaired_text,
                    "text_raw": raw_text if was_repaired else None,
                    "repaired": was_repaired,
                    "repair_log": repair_log if was_repaired else None,
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

    # Line spacing for inserts
    for i, info in enumerate(body_spans_with_pos):
        sd = info["data"].copy()
        if is_insert and i < len(body_spans_with_pos) - 1:
            gap = body_spans_with_pos[i + 1]["y0"] - info["y1"]
            fs = info["font_size"]
            sd["line_spacing"] = round((gap + fs) / fs, 2) if fs > 0 else None
        elif is_insert:
            sd["line_spacing"] = None
        body.append(sd)

    return {"header_table": header_table, "body": body, "annotations": annotations}


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


def extract_images(page: fitz.Page, doc: fitz.Document) -> list:
    """Extract all embedded images with metadata."""
    images = []
    try:
        image_list = page.get_images(full=True)
    except Exception:
        return images

    for img_info in image_list:
        xref = img_info[0]
        try:
            img_rects = page.get_image_rects(xref)
            for rect in img_rects:
                try:
                    base_image = doc.extract_image(xref)
                    width_px = base_image.get("width", 0)
                    height_px = base_image.get("height", 0)
                except Exception:
                    width_px = img_info[2] if len(img_info) > 2 else 0
                    height_px = img_info[3] if len(img_info) > 3 else 0

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

def extract_page_data(page: fitz.Page, doc: fitz.Document, page_index: int) -> dict:
    """Extract all data from a single page with enhanced features."""
    config = PAGE_CONFIG.get(page_index, {"key": f"page{page_index+1}", "name": "unknown"})

    rect = page.rect
    width_pt, height_pt = rect.width, rect.height
    header_threshold = height_pt * 0.20
    is_insert = page_index in [2, 3]

    # 1. Paths first (zone detection needs them)
    paths = extract_paths_enhanced(page)

    # 2. Zone detection
    zones = detect_zones(paths, width_pt, height_pt, header_threshold)

    # 3. Text extraction
    sections = extract_text_spans_enhanced(page, header_threshold, is_insert)

    # 4. Assign spans to zones
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
    images = extract_images(page, doc)

    # 6. Build convenience strings
    body_normal = [s["text"] for s in sections["body"] if s.get("rotation_deg", 0) == 0]
    body_text = " ".join(body_normal)

    rotated_elements = [
        {"text": s["text"], "rotation_deg": s["rotation_deg"],
         "bbox_mm": s["bbox_mm"], "zone": s.get("zone")}
        for s in sections["body"] if s.get("rotation_deg", 0) != 0
    ]

    # Count repairs
    repair_count = sum(1 for s in sections["body"] if s.get("repaired"))
    repair_count += sum(1 for s in sections["header_table"] if s.get("repaired"))

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
        "name": config["name"],
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
            "annotations_filtered": len(sections["annotations"]),
            "zones_detected": len(zones),
            "rotated_elements_count": len(rotated_elements),
        },
    }


# ═══════════════════════════════════════════════════════════════
# DOCUMENT-LEVEL EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_artwork(pdf_bytes: bytes) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    result = {}

    for page_index in range(min(len(doc), 5)):
        if page_index in SKIP_PAGES or page_index > 3:
            continue
        page = doc[page_index]
        config = PAGE_CONFIG.get(page_index)
        if config:
            result[config["key"]] = extract_page_data(page, doc, page_index)

    doc.close()
    return result


# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/extract")
async def extract_pdf(file: UploadFile = File(...)):
    """Extract artwork compliance data from a PDF file."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    try:
        pdf_bytes = await file.read()
        result = extract_artwork(pdf_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "artwork-extractor", "version": "2.0.0"}


@app.get("/")
async def root():
    return {
        "service": "Artwork Compliance Extractor",
        "version": "2.0.0",
        "features": [
            "Zone detection (panel boundaries from fold lines)",
            "Rotation detection (flap text, rotated elements)",
            "Ligature repair (ti, fi, fl auto-fix with audit trail)",
            "Annotation filtering (dimension markers separated)",
        ],
        "endpoints": {
            "POST /extract": "Upload PDF and extract artwork data",
            "POST /html-to-pdf": "Convert HTML string to PDF binary",
            "GET /health": "Health check",
        },
    }


@app.post("/html-to-pdf")
async def html_to_pdf(request: Request):
    """Convert HTML body to PDF and return binary stream."""
    try:
        data = await request.json()
        html_string = data.get("html", "")
        if not html_string:
            raise HTTPException(status_code=400, detail="Missing 'html' field in request body")
        pdf_buffer = io.BytesIO()
        pisa_status = pisa.CreatePDF(io.StringIO(html_string), dest=pdf_buffer)
        if pisa_status.err:
            raise HTTPException(status_code=500, detail="PDF rendering failed")
        pdf_buffer.seek(0)
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=compliance_report.pdf"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")