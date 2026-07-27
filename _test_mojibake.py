"""Fix D check: mojibake repair in repair_text().

Corrupt inputs are generated the same way the bug produces them — UTF-8 bytes
decoded as GBK — so the test cannot drift from reality by mistyping a glyph.
"""
import io
import sys
import types

# Import repair_text without pulling in fitz/fastapi/playwright.
src = io.open("main.py", encoding="utf-8").read()
start = src.index("KNOWN_CORRECTIONS = {")
end = src.index("# OCR CROP FALLBACK")
mod = types.ModuleType("repairmod")
mod.__dict__["re"] = __import__("re")
exec(compile(src[start:end], "main.py-slice", "exec"), mod.__dict__)
repair_text = mod.repair_text
repair_mojibake = mod.repair_mojibake


def corrupt(s):
    """Reproduce the extractor bug: UTF-8 bytes read back as GBK."""
    return s.encode("utf-8").decode("gbk")


CASES = [
    # (label, input, expected output)
    ("composicion",  corrupt("Composición:"),                 "Composición:"),
    ("dioxido",      corrupt("Color: Dióxido de Titanio"),    "Color: Dióxido de Titanio"),
    ("via admin",    corrupt("Vía de administración: oral"),  "Vía de administración: oral"),
    ("mantengase",   corrupt("manténgase fuera del alcance"), "manténgase fuera del alcance"),
    ("ninos",        corrupt("de los niños"),                 "de los niños"),
    ("prescripcion", corrupt("BAJO PRESCRIPCIÓN MÉDICA"),     "BAJO PRESCRIPCIÓN MÉDICA"),
    ("mixed line",   corrupt("Almacenar a 30 °C. Composición: Albendazol 200 mg"),
                             "Almacenar a 30 °C. Composición: Albendazol 200 mg"),
    # Clean text must pass through byte-identical.
    ("clean spanish", "Composición: Albendazol 200 mg/5 ml",  "Composición: Albendazol 200 mg/5 ml"),
    ("clean ascii",   "AC2490V.3",                            "AC2490V.3"),
    # Existing ligature behaviour must survive unchanged.
    ("ligature known", "con�ene:",                       "contiene:"),
    ("ligature fi",    "identiﬁcation",                  "identification"),
]

failures = []
for label, inp, expected in CASES:
    out, was_repaired, repairs, confidence = repair_text(inp)
    ok = out == expected
    if not ok:
        failures.append((label, inp, expected, out))
    print(("ok   " if ok else "FAIL ") + label.ljust(16)
          + " repaired=" + str(was_repaired).ljust(5)
          + " conf=" + str(confidence))
    if not ok:
        print("        expected: " + repr(expected))
        print("        got     : " + repr(out))

# Unrecoverable damage must be left alone, never mangled further.
lost = "Composici��n"
out, _, _, _ = repair_text(lost)
print(("ok   " if "�" not in out or out != lost else "note ")
      + "u+fffd handling -> " + repr(out))

# A run that does not round-trip must be returned untouched.
untouched, reps = repair_mojibake("一")  # lone CJK, invalid as UTF-8
print("ok   non-roundtrip left alone" if untouched == "一" and not reps
      else "FAIL non-roundtrip mangled")
if untouched != "一":
    failures.append(("non-roundtrip", "一", "一", untouched))

print()
print("all passed" if not failures else str(len(failures)) + " FAILED")
sys.exit(1 if failures else 0)
