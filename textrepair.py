# -*- coding: utf-8 -*-
"""Pure helpers shared by the extractor and its tests.

Nothing here imports fitz, PIL, or fastapi, so the test suites can import this
module directly. The suites used to slice main.py and exec the source, which
broke the moment a type annotation moved into the sliced region.
"""
import math
import re
from typing import Any, Optional

# Exact, not the rounded 0.3528 this used to hold. The rounded constant was
# high by 0.0063%, so every measurement read slightly large — +0.022 mm on a
# 350 mm insert. Below any artwork tolerance, but it was a systematic bias in
# the one number every dimension check divides through.
PT_TO_MM = 25.4 / 72


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


# ─── Annotation detection ────────────────────────────────────
ANNOTATION_COLORS = {"#0000ff", "#0000cd", "#0054a6", "#0091d2", "#1a73e8",
                     "#196ea6", "#1a6ea6", "#3373a8"}


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


ANNOTATION_RGB = [_hex_to_rgb(c) for c in ANNOTATION_COLORS]
# Per-channel tolerance. An exact-match allowlist breaks on the next designer's
# palette drift — #196ea6 vs #1a6ea6 (one hex digit) let 12 dimension callouts
# through as body copy (B-01). Rich black (#231f20) header values stay far
# outside this window.
#
# #3373a8 was listed rather than handled by widening the tolerance: it sits
# exactly 25 away from #1a6ea6 on red, one unit past the window, so it near-
# missed and 16 dimension callouts on Art-CommercialPDF-12168 were classified
# as body copy. Bumping the tolerance to 25 would admit it, but widens the
# window for every colour on every artwork to buy one palette entry.
ANNOTATION_COLOR_TOLERANCE = 24


def is_annotation_color(color_hex: str) -> bool:
    try:
        r, g, b = _hex_to_rgb(color_hex.lower())
    except (ValueError, IndexError, TypeError):
        return False
    return any(abs(r - cr) <= ANNOTATION_COLOR_TOLERANCE
               and abs(g - cg) <= ANNOTATION_COLOR_TOLERANCE
               and abs(b - cb) <= ANNOTATION_COLOR_TOLERANCE
               for cr, cg, cb in ANNOTATION_RGB)


DIMENSION_PATTERN = re.compile(r'^\s*[\d.]+\s*mm\s*$', re.IGNORECASE)
ARROW_CHARS = {'◄', '►', '▲', '▼', '←', '→', '↑', '↓', '◀', '▶'}

# ─── Print-production marks ──────────────────────────────────
# Instructions addressed to the printer, not copy addressed to the patient:
# "Pasting Side", "Stereo print", varnish and die-line callouts. They are on the
# artwork file but not on the finished pack, so they are not artwork content and
# must not be judged as such — they are the reason rule 1-1 reported an Arial
# font violation and rule 1-18 reported English words on Spanish artwork.
#
# Matched as whole spans only. A production term appearing inside a real sentence
# is left alone; these marks always sit in their own span.
PRODUCTION_MARK_TERMS = {
    "pasting side", "stereo print", "stereo", "die line", "dieline", "die cut",
    "cut line", "cutline", "crease line", "fold line", "bleed line", "trim line",
    "varnish", "varnish free", "no varnish", "matt varnish", "gloss varnish",
    "spot uv", "overprint", "emboss", "debossing", "hot foil", "foil stamp",
    "kiss cut", "tuck flap", "glue flap", "gusset",
    "reverse side", "inner side", "outer side", "non printing area",
    "not to scale", "scale 1:1",
    # NOT in this set (B-02): "pantone", "cmyk", "artwork size", "flat size",
    # "print side", "colour code", "color code" — those are tabular header
    # field labels (Artwork Headers V2 §4.1–4.7) that appear on every artwork
    # by design. With the suffix tolerance they matched "Pantone No." etc. and
    # deleted mandatory header labels, producing false FAILs. A genuine
    # standalone CMYK/Pantone swatch callout is set in the annotation palette
    # and surfaces via the production-mark near-miss log instead.
}
# A short standalone English phrase set in the annotation palette is the shape a
# production mark takes. Anything matching that shape but absent from the
# vocabulary is logged as a near-miss so new terms surface instead of silently
# becoming compliance failures — same contract as annotation_near_misses.
PRODUCTION_MARK_SHAPE = re.compile(r'^[A-Za-z][A-Za-z0-9 .:/\-]{2,28}$')

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
    # B-09: these were reaching the ti-guess and depending on OCR to confirm.
    'Prin\ufffdng': 'Printing', 'prin\ufffdng': 'printing',
    'Prin\ufffdng Overlay': 'Printing Overlay',
    'Prin\ufffdng Zone': 'Printing Zone',
    'Non Prin\ufffdng Area': 'Non Printing Area',
    'Pas\ufffdng Side': 'Pasting Side', 'pas\ufffdng side': 'pasting side',
}


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
      - "guessed" -> an isolated U+FFFD between letters was read as the "ti"
                     ligature, or a soft hyphen was; text is unverified
      - "failed"  -> U+FFFD damage that is NOT ligature-shaped (adjacent runs,
                     word-edge). Nothing was invented: the sentinel is left in
                     place so the corruption stays visible downstream.
    """
    has_ligature_damage = ('\ufffd' in text or '\xad' in text
                           or '\ufb01' in text or '\ufb02' in text)
    has_mojibake = bool(MOJIBAKE_RUN.search(text))
    if not has_ligature_damage and not has_mojibake:
        return text, False, [], None

    repairs = []
    repaired = text
    guessed = False
    failed = False

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

    # 2. Remaining U+FFFD. Only a SINGLE sentinel sitting between two letters
    #    has the shape of a dropped "ti" ligature, so only that is guessed.
    #    Anything else (adjacent sentinels as in "Composici" + 2x U+FFFD + "n",
    #    or damage at a word edge) is left visible — blanket replacement turned
    #    two lost characters into "Composicititin", which reads like real text
    #    and sends a reviewer chasing a content mismatch instead of a corrupt
    #    extraction (B-06).
    if '\ufffd' in repaired:
        single = re.sub(r'(?<=[^\W\d_])\ufffd(?=[^\W\d_])',
                        'ti', repaired)
        if single != repaired:
            repairs.append({"from": "U+FFFD", "to": "ti", "type": "ligature_ti",
                            "confidence": "guessed"})
            repaired = single
            guessed = True
        if '\ufffd' in repaired:
            repairs.append({"from": "U+FFFD", "to": None,
                            "type": "unrecoverable", "confidence": "failed"})
            failed = True

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

    confidence = ("failed" if failed else
                  "guessed" if guessed else
                  "high" if repairs else None)
    return repaired, len(repairs) > 0, repairs, confidence


def is_annotation(text: str, color_hex: str, in_header: bool = False) -> bool:
    """Detect dimension marker annotations.

    in_header exempts the tabular header: a dimension there is a declared field
    value ("60 mm" as Artwork size), not a callout, and must survive (B-01).
    """
    if in_header:
        return False
    color_match = is_annotation_color(color_hex)
    pattern_match = bool(DIMENSION_PATTERN.match(text.strip()))
    arrow_match = text.strip() in ARROW_CHARS
    return (color_match and pattern_match) or arrow_match


# How much trailing noise may follow a production term and still be the same
# mark. "Pasting Side 1" and "Varnish Free 2" qualify; a sentence that merely
# begins with a production word does not.
PRODUCTION_MARK_SUFFIX_MAX = 4


def _production_key(text: str) -> str:
    """Normalize a span for vocabulary lookup: case, punctuation, spacing.

    The colon is kept because "scale 1:1" is itself a term, then stripped from
    the ends so "Pasting Side:" still matches "pasting side".
    """
    key = re.sub(r'[^a-z0-9: ]', ' ', text.lower())
    return re.sub(r'\s+', ' ', key).strip().strip(':').strip()


def is_production_mark(text: str) -> bool:
    """True when a span is a printer instruction rather than artwork copy."""
    key = _production_key(text)
    if not key:
        return False
    if key in PRODUCTION_MARK_TERMS:
        return True
    # Tolerate a short suffix ("Pasting Side 1") but never a full sentence that
    # happens to open with a production word ("Stereo printing instructions
    # are documented in the batch record" is artwork copy, not a mark).
    for term in PRODUCTION_MARK_TERMS:
        if key.startswith(term + " ") and len(key) - len(term) <= PRODUCTION_MARK_SUFFIX_MAX:
            return True
    return False


def looks_like_production_mark(text: str, color_hex: str) -> bool:
    """Shape of a production mark without a vocabulary hit — worth logging."""
    stripped = text.strip()
    if not PRODUCTION_MARK_SHAPE.match(stripped):
        return False
    if not is_annotation_color(color_hex):
        return False
    return not is_production_mark(stripped)


def is_dimension_near_miss(text: str, color_hex: str, in_header: bool = False) -> bool:
    """Dimension-shaped text in a colour we don't recognise as a callout.

    The signal this exists to catch is a callout colour missing from
    ANNOTATION_COLORS, or a stray dimension loose in the body. A dimension
    inside the tabular header is neither — it is a declared field value
    ("Size (W): 60 mm"), which is what a foil header is supposed to contain.
    Without the in_header guard every such value logged a near-miss and forced
    needs_review=True on a page that extracted perfectly.
    """
    if in_header:
        return False
    if not DIMENSION_PATTERN.match(text.strip()):
        return False
    return not is_annotation_color(color_hex)


def _normalize_for_ocr_compare(text: str) -> str:
    """Normalize text for OCR-vs-repair comparison (whitespace/case tolerant)."""
    return re.sub(r'\s+', ' ', text).strip().lower()


def direction_to_rotation(dir_x: float, dir_y: float) -> int:
    """Convert direction vector to rotation degrees (0, 90, 180, 270)."""
    angle = math.degrees(math.atan2(dir_y, dir_x))
    angle = round(angle)
    if angle < 0:
        angle += 360
    return (round(angle / 90) * 90) % 360


# The number returned above is a PDF text-direction angle. The SOP describes the
# same geometry in the opposite sense: bottom-to-top text (dir 0,-1 -> 270 here)
# is what the SOP calls "90 degrees counter-clockwise". A rule written against
# the SOP wording would compare to 90 and fail every compliant artwork, so both
# vocabularies are emitted side by side (B-04).
ROTATION_SOP = {0: "0", 90: "90_CW", 180: "180", 270: "90_CCW"}


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


def _label_key(text: str) -> str:
    """Normalise a line for label matching: lowercase, strip punctuation."""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()
