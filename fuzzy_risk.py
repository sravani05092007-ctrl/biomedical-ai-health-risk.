"""Fuzzy risk assessment (Mamdani-style, pure NumPy - no extra dependency).

Inputs : age (yrs), trestbps (resting BP mmHg), chol (mg/dl), thalach (max heart rate)
Output : fuzzy risk score 0-100 (centroid defuzzification)
Membership functions: triangular / shoulder.  Rules: min for AND, max for OR/aggregation.
"""
import numpy as np

def tri(x, a, b, c):
    """Triangular membership; a==b or b==c gives a shoulder."""
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / (b - a + 1e-12), (x >= b).astype(float))
    right = np.where(c > b, (c - x) / (c - b + 1e-12), (x <= b).astype(float))
    return np.clip(np.minimum(left, right), 0, 1)

# ---- membership functions (a, b, c) ----
AGE  = {"young": (0, 30, 45), "middle": (40, 52, 64), "old": (58, 72, 120)}
BP   = {"normal": (0, 110, 130), "elevated": (120, 138, 155), "high": (145, 170, 260)}
CHOL = {"normal": (0, 170, 210), "borderline": (200, 225, 250), "high": (240, 290, 600)}
HR   = {"low": (0, 100, 130), "medium": (120, 145, 165), "high": (155, 180, 260)}  # max HR achieved
RISK_UNIVERSE = np.arange(0, 101, 1.0)
RISK = {"low": (0, 0, 40), "medium": (30, 50, 70), "high": (60, 100, 100)}

def mu(mfs, name, v):
    return float(tri(v, *mfs[name]))

def _rules(m):
    a, b, c, h = m["age"], m["bp"], m["chol"], m["hr"]
    return [
        ("high",   min(a["old"], b["high"])),
        ("high",   min(b["high"], c["high"])),
        ("high",   min(c["high"], h["low"])),
        ("high",   min(a["old"], h["low"])),
        ("medium", min(a["middle"], max(b["elevated"], c["borderline"]))),
        ("medium", min(a["old"], b["normal"], c["normal"])),
        ("medium", min(b["elevated"], c["borderline"])),
        ("medium", min(a["middle"], h["medium"], max(b["high"], c["high"]))),
        ("low",    min(a["young"], b["normal"], c["normal"])),
        ("low",    min(b["normal"], c["normal"], h["high"])),
        ("low",    min(a["young"], h["high"])),
        ("low",    min(a["middle"], b["normal"], c["normal"], h["medium"])),
    ]

def fuzzify(age, bp, chol, hr):
    return {
        "age":  {k: mu(AGE, k, age) for k in AGE},
        "bp":   {k: mu(BP, k, bp) for k in BP},
        "chol": {k: mu(CHOL, k, chol) for k in CHOL},
        "hr":   {k: mu(HR, k, hr) for k in HR},
    }

def fuzzy_risk(age, bp, chol, hr, explain=False):
    m = fuzzify(age, bp, chol, hr)
    fired = _rules(m)
    agg = np.zeros_like(RISK_UNIVERSE)
    for label, strength in fired:
        agg = np.maximum(agg, np.minimum(strength, tri(RISK_UNIVERSE, *RISK[label])))
    score = float((agg * RISK_UNIVERSE).sum() / agg.sum()) if agg.sum() > 0 else 50.0
    if explain:
        return score, [(l, round(s, 2)) for l, s in fired], m
    return score

def category(score):
    return "Low" if score < 35 else ("Medium" if score <= 65 else "High")

if __name__ == "__main__":
    for p in [(30, 115, 180, 175), (52, 140, 240, 140), (68, 165, 290, 105)]:
        s = fuzzy_risk(*p); print(p, "->", round(s, 1), category(s))
