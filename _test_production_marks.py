"""Print-production-mark exclusion checks.

A production mark is a printer instruction that sits on the artwork file but not
on the finished pack ("Pasting Side", "Stereo print"). Judging it as artwork copy
is what made rule 1-1 report an Arial font violation and rule 1-18 report English
words on Spanish artwork. These tests pin both directions: marks are recognised,
and real artwork copy is never mistaken for one.
"""
import io
import sys
import types

# Import the detection helpers without pulling in fitz/fastapi/playwright.
src = io.open("main.py", encoding="utf-8").read()
start = src.index("# ─── Annotation detection ──")
end = src.index("# ═══", src.index("def looks_like_production_mark"))
mod = types.ModuleType("prodmod")
mod.__dict__["re"] = __import__("re")
exec(compile(src[start:end], "main.py-slice", "exec"), mod.__dict__)
is_production_mark = mod.is_production_mark
looks_like_production_mark = mod.looks_like_production_mark

ANNOTATION_BLUE = "#0000ff"
BLACK = "#000000"

# ── Spans that MUST be excluded ──────────────────────────────────────────────
MARKS = [
    "Pasting Side",
    "PASTING SIDE",
    "pasting side",
    "Pasting Side:",
    "Pasting Side 1",
    "Stereo print",
    "Stereo Print",
    "Die Line",
    "Dieline",
    "Varnish Free",
    "Non Printing Area",
    "Not to scale",
    "Pantone",
    "CMYK",
]

# ── Real artwork copy that must NEVER be excluded ────────────────────────────
ARTWORK = [
    "ESCITALOPRAM Rhydburg 20 mg Tabletas recubiertas",
    "Composición:",
    "Color: Dióxido de Titanio",
    "AC2490V.3",
    "Vía de administración: oral",
    "20 mg",
    "Rhydburg Pharmaceuticals Ltd.",
    "Manténgase fuera del alcance de los niños",
    "BAJO PRESCRIPCIÓN MÉDICA",
    "Escitalopram Oxalate USP",
    # Contains a production word inside a real sentence — must stay.
    "Imprimir el lote en el lado de pegado del estuche",
    "Stereo printing instructions are documented in the batch record",
    "",
    "   ",
]

failures = []

for text in MARKS:
    ok = is_production_mark(text)
    if not ok:
        failures.append(("not excluded", text))
    print(("ok   " if ok else "FAIL ") + "mark      " + repr(text))

print()
for text in ARTWORK:
    ok = not is_production_mark(text)
    if not ok:
        failures.append(("wrongly excluded", text))
    print(("ok   " if ok else "FAIL ") + "artwork   " + repr(text))

print()
# ── Near-miss logging: unknown terms in the annotation palette must surface ──
NEAR_MISS_CASES = [
    # (text, colour, should_be_near_miss)
    ("Perforation Line", ANNOTATION_BLUE, True),   # unknown term, annotation colour
    ("Braille Position", ANNOTATION_BLUE, True),   # unknown term, annotation colour
    ("Pasting Side", ANNOTATION_BLUE, False),      # known -> excluded, not a near-miss
    ("Perforation Line", BLACK, False),            # black = artwork copy, not a mark
    ("Composición:", ANNOTATION_BLUE, False),      # non-ASCII -> fails the shape test
    ("ESCITALOPRAM Rhydburg 20 mg Tabletas recubiertas", ANNOTATION_BLUE, False),  # too long
]
for text, color, expected in NEAR_MISS_CASES:
    got = looks_like_production_mark(text, color)
    ok = got == expected
    if not ok:
        failures.append(("near-miss %s expected %s" % (repr(text), expected), got))
    print(("ok   " if ok else "FAIL ") + "near-miss " + repr(text)
          + " " + color + " -> " + str(got))

print()
if failures:
    print(str(len(failures)) + " FAILED")
    for f in failures:
        print("   " + repr(f))
else:
    print("all passed")
sys.exit(1 if failures else 0)
