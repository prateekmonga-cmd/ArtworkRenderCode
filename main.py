"""
Artwork Compliance Extraction FastAPI Application
==================================================
Extracts ALL measurable properties from pharmaceutical artwork PDFs
for compliance checking. Returns structured JSON.

Usage:
    uvicorn artwork_extractor_api:app --host 0.0.0.0 --port 8000

Requirements:
    pip install fastapi uvicorn python-multipart PyMuPDF
"""

import fitz  # PyMuPDF
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Any

app = FastAPI(
    title="Artwork Compliance Extractor",
    description="Extracts text, paths, and images from pharmaceutical artwork PDFs",
    version="1.0.0"
)

# Conversion factor: 1 PDF point = 0.3528 mm
PT_TO_MM = 0.3528

# Page configuration
PAGE_CONFIG = {
    0: {"key": "page1", "name": "outer carton"},
    1: {"key": "page2", "name": "sticker label"},
    2: {"key": "page3", "name": "insert front"},
    3: {"key": "page4", "name": "insert back"},
}
SKIP_PAGES = [4]  # 0-indexed: page 5


def pt_to_mm(value: float) -> float:
    """Convert PDF points to millimeters."""
    return round(value * PT_TO_MM, 2)


def bbox_to_mm(bbox: tuple) -> list:
    """Convert bounding box from points to mm."""
    return [pt_to_mm(v) for v in bbox]


def rgb_to_hex(color: Any) -> str:
    """Convert RGB tuple (0-1 range) or grayscale to hex string."""
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
        # CMYK to RGB approximation
        c, m, y, k = color
        r = int(255 * (1 - c) * (1 - k))
        g = int(255 * (1 - m) * (1 - k))
        b = int(255 * (1 - y) * (1 - k))
        return f"#{r:02x}{g:02x}{b:02x}"
    return "#000000"


def int_color_to_hex(color: int) -> str:
    """Convert PyMuPDF integer color to hex string."""
    r = (color >> 16) & 0xFF
    g = (color >> 8) & 0xFF
    b = color & 0xFF
    return f"#{r:02x}{g:02x}{b:02x}"


def is_bold(font_name: str, flags: int) -> bool:
    """Determine if font is bold from name or flags."""
    if flags & (1 << 18):  # Bold flag in PDF spec
        return True
    bold_patterns = ["bold", "black", "heavy", "demi", "semibold", "extrabold"]
    font_lower = font_name.lower()
    return any(p in font_lower for p in bold_patterns)


def is_italic(font_name: str, flags: int) -> bool:
    """Determine if font is italic from name or flags."""
    if flags & (1 << 6):  # Italic flag in PDF spec
        return True
    italic_patterns = ["italic", "oblique", "slant"]
    font_lower = font_name.lower()
    return any(p in font_lower for p in italic_patterns)


def get_base_font_name(font_name: str) -> str:
    """Extract base font family name, removing subset prefix and style suffixes."""
    name = font_name
    # Remove subset prefix (e.g., "ABCDEF+")
    if "+" in name:
        name = name.split("+")[-1]
    # Remove common style suffixes
    for suffix in ["-Bold", "-Italic", "-BoldItalic", "-Regular",
                   "Bold", "Italic", "Regular", "Light", "Medium",
                   "-MT", "MT", "-PS", "PS"]:
        name = name.replace(suffix, "")
    return name.strip()


def snap_size(size: float, precision: float = 0.5) -> float:
    """Snap font size to nearest precision (default 0.5pt)."""
    return round(size / precision) * precision


def snap_coord(coord: float) -> float:
    """Snap coordinate to nearest integer point."""
    return round(coord)


def extract_text_spans(page: fitz.Page, header_threshold: float, is_insert: bool) -> dict:
    """
    Extract all text spans with properties.
    Returns dict with header_table and body arrays.
    """
    header_table = []
    body = []
    
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    
    # Collect all body spans with y-position for line spacing calculation
    body_spans_with_pos = []
    
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # Skip non-text blocks
            continue
        
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    continue
                
                bbox = span.get("bbox", (0, 0, 0, 0))
                font_name = span.get("font", "Unknown")
                font_size = snap_size(span.get("size", 0))
                color = span.get("color", 0)
                flags = span.get("flags", 0)
                
                # Convert color
                if isinstance(color, int):
                    color_hex = int_color_to_hex(color)
                else:
                    color_hex = rgb_to_hex(color)
                
                span_data = {
                    "text": text,
                    "font_name": get_base_font_name(font_name),
                    "font_name_full": font_name,
                    "font_size_pt": font_size,
                    "is_bold": is_bold(font_name, flags),
                    "is_italic": is_italic(font_name, flags),
                    "color_hex": color_hex,
                    "bbox": [snap_coord(b) for b in bbox],
                    "bbox_mm": bbox_to_mm(bbox)
                }
                
                # Classify into header or body based on y-position
                if bbox[1] < header_threshold:
                    header_table.append(span_data)
                else:
                    body_spans_with_pos.append({
                        "data": span_data,
                        "y0": bbox[1],
                        "y1": bbox[3],
                        "font_size": font_size
                    })
    
    # Sort body spans by vertical position (top to bottom)
    body_spans_with_pos.sort(key=lambda x: (x["y0"], x["data"]["bbox"][0]))
    
    # Calculate line spacing for insert pages (pages 3 and 4)
    for i, span_info in enumerate(body_spans_with_pos):
        span_data = span_info["data"].copy()
        
        if is_insert:
            # Calculate line spacing to next span
            if i < len(body_spans_with_pos) - 1:
                next_span = body_spans_with_pos[i + 1]
                gap = next_span["y0"] - span_info["y1"]
                font_size = span_info["font_size"]
                if font_size > 0:
                    # Line spacing = (gap + font_size) / font_size
                    line_spacing = round((gap + font_size) / font_size, 2)
                    span_data["line_spacing"] = line_spacing
                else:
                    span_data["line_spacing"] = None
            else:
                span_data["line_spacing"] = None
        
        body.append(span_data)
    
    return {
        "header_table": header_table,
        "body": body
    }


def extract_paths(page: fitz.Page) -> list:
    """Extract all drawn paths with stroke properties."""
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
        
        path_data = {
            "stroke_width_pt": round(stroke_width, 2) if stroke_width else 0,
            "stroke_color_hex": rgb_to_hex(stroke_color) if stroke_color else None,
            "fill_color_hex": rgb_to_hex(fill_color) if fill_color else None,
            "color_hex": rgb_to_hex(stroke_color) if stroke_color else rgb_to_hex(fill_color),
            "bbox": [snap_coord(r) for r in rect],
            "bbox_mm": bbox_to_mm(rect)
        }
        
        paths.append(path_data)
    
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
                    "width_px": width_px,
                    "height_px": height_px,
                    "width_mm": pt_to_mm(rect.width),
                    "height_mm": pt_to_mm(rect.height)
                }
                
                if height_px > 0:
                    image_data["aspect_ratio"] = round(width_px / height_px, 2)
                
                images.append(image_data)
                
        except Exception:
            continue
    
    return images


def extract_page_data(page: fitz.Page, doc: fitz.Document, page_index: int) -> dict:
    """Extract all data from a single page."""
    config = PAGE_CONFIG.get(page_index, {"key": f"page{page_index + 1}", "name": "unknown"})
    
    # Page dimensions
    rect = page.rect
    width_pt = rect.width
    height_pt = rect.height
    
    # Header threshold: top 15% of page
    header_threshold = height_pt * 0.15
    
    # Determine if this is an insert page (pages 3 or 4, index 2 or 3)
    is_insert = page_index in [2, 3]
    
    # Extract components
    sections = extract_text_spans(page, header_threshold, is_insert)
    paths = extract_paths(page)
    images = extract_images(page, doc)
    
    return {
        "name": config["name"],
        "sections": sections,
        "page_dimensions": {
            "width_pt": round(width_pt, 2),
            "height_pt": round(height_pt, 2),
            "width_mm": pt_to_mm(width_pt),
            "height_mm": pt_to_mm(height_pt)
        },
        "paths": paths,
        "images": images
    }


def extract_artwork(pdf_bytes: bytes) -> dict:
    """
    Main extraction function.
    
    Args:
        pdf_bytes: PDF file content as bytes
    
    Returns:
        Dictionary with all extracted data
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    result = {}
    
    for page_index in range(min(len(doc), 5)):
        # Skip page 5 (index 4)
        if page_index in SKIP_PAGES:
            continue
        
        # Only process pages 1-4 (indices 0-3)
        if page_index > 3:
            continue
        
        page = doc[page_index]
        config = PAGE_CONFIG.get(page_index)
        
        if config:
            page_data = extract_page_data(page, doc, page_index)
            result[config["key"]] = page_data
    
    doc.close()
    
    return result


@app.post("/extract")
async def extract_pdf(file: UploadFile = File(...)):
    """
    Extract artwork compliance data from a PDF file.
    
    - Accepts a PDF file upload
    - Extracts text, paths, and images from pages 1-4
    - Skips page 5 (technical drawing)
    - Returns structured JSON for compliance checking
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF"
        )
    
    try:
        # Read file content
        pdf_bytes = await file.read()
        
        # Extract data
        result = extract_artwork(pdf_bytes)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "artwork-extractor"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Artwork Compliance Extractor",
        "version": "1.0.0",
        "endpoints": {
            "POST /extract": "Upload PDF and extract artwork data",
            "GET /health": "Health check"
        }
    }