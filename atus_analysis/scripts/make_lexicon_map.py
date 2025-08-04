#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a CSV that maps ATUS 6-digit activity codes (from the official lexicon PDF)
to your taxonomy: (sub_name, major_class).

Output CSV columns:
  code,text,major2,sub_name,major_class,count

Key choices baked in:
- Exact CODE_OVERRIDES applied FIRST (trump everything).
- 020299 → (DISHWASHING, ELECTRICITY_CONSUMING) per request.
- 020101/020199 → CLEANING_VACUUM (electric), so “Interior cleaning” never
  lands in OTHER_NON_ELEC.
- Reasonable keyword + prefix defaults, then conservative fallbacks.
- Optional strict mode gate with a report.

Install (once):
  pip install pdfplumber pandas

Run (PowerShell example):
  python ".\\atus_analysis\\scripts\\make_lexicon_map.py" ^
    --pdf ".\\atus_analysis\\data\\sequences\\lexiconnoex0323.pdf" ^
    --out ".\\atus_analysis\\assets\\atus_lexicon_map.csv" ^
    --report ".\\atus_analysis\\assets\\lexicon_report.txt" ^
    --strict
"""

import argparse
import collections
import re
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd

try:
    import pdfplumber
except Exception:
    pdfplumber = None


# ---------------------------- Taxonomy ---------------------------------------

MAJOR_CLASSES = {
    "AMBIGUOUS": 0,
    "ELECTRICITY_CONSUMING": 1,
    "NON_ELECTRIC": 2,
    "OUT_OF_HOME": 3,
}

SUB_NAMES = {
    # Electricity-consuming (home)
    "COOKING",
    "CLEANING_VACUUM",
    "DISHWASHING",
    "LAUNDRY",
    "IRONING",
    "WORK_ENTERTAINMENT",       # TV/audio/games/PC/phone all merged here
    "HVAC_ADJUST",
    "LIGHTING_ONLY",
    "APPLIANCE_TOOL_USE",
    "OTHER_ELECTRIC",

    # Non-electric (home)
    "SLEEP",
    "EAT_DRINK",
    "PERSONAL_CARE",
    "CHILDCARE",
    "READ_PAPERWORK_NON_ELEC",
    "SOCIAL_HOME",
    "EXERCISE_NO_MACHINE",
    "OTHER_NON_ELEC",

    # Out-of-home
    "TRAVEL",
    "SHOPPING",
    "DINING_OUT",
    "WORK_SCHOOL",
    "HEALTHCARE_OUT",
    "ENTERTAINMENT_OUT",
    "OUTDOOR_EXERCISE",
    "GOV_SERVICES",
    "OTHER_OUT",
    "OUTSIDE_HOUSE",  # around-the-house outdoor maintenance
}


# ----------------------- Exact code overrides (first) ------------------------

# Map CODE -> (sub_name, major_class). STRING keys, zero-padded 6 digits.
CODE_OVERRIDES: Dict[str, Tuple[str, str]] = {
    # Household/cleaning core
    "020101": ("CLEANING_VACUUM", "ELECTRICITY_CONSUMING"),           # Interior cleaning
    "020199": ("CLEANING_VACUUM", "ELECTRICITY_CONSUMING"),           # Housework, n.e.c.
    "020299": ("DISHWASHING", "ELECTRICITY_CONSUMING"),               # *** per request ***
    "020301": ("OTHER_NON_ELEC", "NON_ELECTRIC"),                     # Interior decor/repairs
    "020302": ("APPLIANCE_TOOL_USE", "ELECTRICITY_CONSUMING"),        # Tools/furniture build/repair
    "020303": ("HVAC_ADJUST", "ELECTRICITY_CONSUMING"),               # Heating/cooling
    "020399": ("OTHER_NON_ELEC", "NON_ELECTRIC"),
    "020401": ("OUTSIDE_HOUSE", "NON_ELECTRIC"),
    "020402": ("OUTSIDE_HOUSE", "NON_ELECTRIC"),
    "020499": ("OUTSIDE_HOUSE", "NON_ELECTRIC"),
    "020501": ("OUTSIDE_HOUSE", "NON_ELECTRIC"),
    "020502": ("OUTSIDE_HOUSE", "NON_ELECTRIC"),                      # yard/pool upkeep default
    "020599": ("OUTSIDE_HOUSE", "NON_ELECTRIC"),
    "020681": ("OTHER_NON_ELEC", "NON_ELECTRIC"),                     # pet/animal care (not vet)
    "020699": ("OTHER_NON_ELEC", "NON_ELECTRIC"),
    "020701": ("APPLIANCE_TOOL_USE", "ELECTRICITY_CONSUMING"),        # vehicle maintenance (self)
    "020799": ("APPLIANCE_TOOL_USE", "ELECTRICITY_CONSUMING"),
    "020801": ("APPLIANCE_TOOL_USE", "ELECTRICITY_CONSUMING"),
    "020899": ("APPLIANCE_TOOL_USE", "ELECTRICITY_CONSUMING"),
    "020901": ("ADMIN_AT_HOME", "NON_ELECTRIC"),
    "020902": ("ADMIN_AT_HOME", "NON_ELECTRIC"),
    "020903": ("ADMIN_AT_HOME", "NON_ELECTRIC"),
    "020904": ("ADMIN_AT_HOME", "ELECTRICITY_CONSUMING"),             # e-mail/messages explicit
    "020905": ("OTHER_NON_ELEC", "NON_ELECTRIC"),

    # Sleep & personal care
    "010101": ("SLEEP", "NON_ELECTRIC"),
    "010102": ("SLEEP", "NON_ELECTRIC"),
    "010199": ("SLEEP", "NON_ELECTRIC"),

    # Eating & drinking (at home)
    "110101": ("EAT_DRINK", "NON_ELECTRIC"),
    "110199": ("EAT_DRINK", "NON_ELECTRIC"),

    # Entertainment/ICT at home
    "120303": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),
    "120304": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),
    "120305": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),
    "120306": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),
    "120307": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),
    "120308": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),

    # Leisure non-ICT at home
    "120301": ("OTHER_NON_ELEC", "NON_ELECTRIC"),
    "120302": ("OTHER_NON_ELEC", "NON_ELECTRIC"),
    "120309": ("OTHER_NON_ELEC", "NON_ELECTRIC"),
    "120310": ("READ_PAPERWORK_NON_ELEC", "NON_ELECTRIC"),
    "120311": ("OTHER_NON_ELEC", "NON_ELECTRIC"),
    "120313": ("OTHER_NON_ELEC", "NON_ELECTRIC"),
    "120399": ("OTHER_NON_ELEC", "NON_ELECTRIC"),

    # Social at home
    "120101": ("SOCIAL_HOME", "NON_ELECTRIC"),
    "120199": ("SOCIAL_HOME", "NON_ELECTRIC"),
    "120201": ("SOCIAL_HOME", "NON_ELECTRIC"),
    "120202": ("SOCIAL_HOME", "NON_ELECTRIC"),

    # Attending arts/museums/movies/gambling (out of home)
    "120401": ("ENTERTAINMENT_OUT", "OUT_OF_HOME"),
    "120402": ("ENTERTAINMENT_OUT", "OUT_OF_HOME"),
    "120403": ("ENTERTAINMENT_OUT", "OUT_OF_HOME"),
    "120404": ("ENTERTAINMENT_OUT", "OUT_OF_HOME"),
    "120405": ("ENTERTAINMENT_OUT", "OUT_OF_HOME"),

    # Phone calls → ICT bucket
    "160101": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),
    "160102": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),
    "160103": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),
    "160104": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),
    "160105": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),
    "160106": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),
    "160107": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),
    "160108": ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING"),

    # Exercise equipment
    "130128": ("APPLIANCE_TOOL_USE", "ELECTRICITY_CONSUMING"),

    # Volunteer food prep must be cooking
    "150201": ("COOKING", "ELECTRICITY_CONSUMING"),

    # Professional & personal services
    "080101": ("CHILDCARE", "NON_ELECTRIC"),
    "080501": ("PERSONAL_CARE", "NON_ELECTRIC"),
    "080502": ("PERSONAL_CARE", "NON_ELECTRIC"),
    "080599": ("PERSONAL_CARE", "NON_ELECTRIC"),
}


# ----------------------- Keyword & prefix rules (second) ---------------------

# Order matters within this list.
KEYWORDS: List[Tuple[str, Tuple[str, str]]] = [
    # Sleep
    (r"\bSleep|Sleeplessness\b", ("SLEEP", "NON_ELECTRIC")),

    # Cleaning/cooking/dishes/laundry/ironing
    (r"\bInterior cleaning|Housework\b", ("CLEANING_VACUUM", "ELECTRICITY_CONSUMING")),
    (r"\bFood (and )?drink preparation|Food presentation\b", ("COOKING", "ELECTRICITY_CONSUMING")),
    (r"\bKitchen.*clean-?up|clean-?up\b", ("DISHWASHING", "ELECTRICITY_CONSUMING")),
    (r"\bLaundry\b", ("LAUNDRY", "ELECTRICITY_CONSUMING")),
    (r"\bIroning\b", ("IRONING", "ELECTRICITY_CONSUMING")),

    # HVAC / lighting / appliance & tools
    (r"\bHeating and cooling|HVAC|thermostat\b", ("HVAC_ADJUST", "ELECTRICITY_CONSUMING")),
    (r"\b(appliance|tool|maintenance|repair|setup|set-up|furniture)\b",
     ("APPLIANCE_TOOL_USE", "ELECTRICITY_CONSUMING")),

    # Admin at home
    (r"\b(mail|email|e-?mail|messages|Financial management|organization|planning|Household management)\b",
     ("ADMIN_AT_HOME", "NON_ELECTRIC")),

    # Outside house
    (r"\bExterior|Lawn|garden|houseplant|yard|outdoor\b", ("OUTSIDE_HOUSE", "NON_ELECTRIC")),

    # Personal care / eating
    (r"\bWashing, dressing|grooming|Personal/Private activities\b", ("PERSONAL_CARE", "NON_ELECTRIC")),
    (r"\bEating and drinking\b", ("EAT_DRINK", "NON_ELECTRIC")),

    # Work/school
    (r"\bWork|job|income-?generating|job searching|interviewing\b", ("WORK_SCHOOL", "OUT_OF_HOME")),
    (r"\bTaking class|Research/homework|Administrative.*education|Extracurricular\b", ("WORK_SCHOOL", "OUT_OF_HOME")),

    # Shopping
    (r"\bGrocery shopping|Purchasing|Shopping|prices and products|Comparison shopping\b", ("SHOPPING", "OUT_OF_HOME")),

    # Professional/Personal services
    (r"\blegal services|banking|financial services|real estate\b", ("GOV_SERVICES", "OUT_OF_HOME")),
    (r"\bpersonal care services\b", ("PERSONAL_CARE", "NON_ELECTRIC")),
    (r"\bmedical|doctor|hospital|clinic|health care\b", ("HEALTHCARE_OUT", "OUT_OF_HOME")),
    (r"\b(veterinary|vet)\b", ("OTHER_NON_ELEC", "NON_ELECTRIC")),

    # Entertainment (home ICT)
    (r"\bTelevision|movies|radio|music|Computer use|Playing games\b", ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING")),
    (r"\bReading for personal interest\b", ("READ_PAPERWORK_NON_ELEC", "NON_ELECTRIC")),
    (r"\bRelaxing, thinking|Writing for personal interest|Hobbies\b", ("OTHER_NON_ELEC", "NON_ELECTRIC")),
    (r"\bAttending performing arts|Attending (movies|museums|gambling)\b", ("ENTERTAINMENT_OUT", "OUT_OF_HOME")),
    (r"\bSocializing and communicating|hosting parties|Meeting new people\b", ("SOCIAL_HOME", "NON_ELECTRIC")),

    # Sports & exercise
    (r"\bUsing cardiovascular equipment\b", ("APPLIANCE_TOOL_USE", "ELECTRICITY_CONSUMING")),
    (r"\bWatching .*sports|Attending sporting events\b", ("ENTERTAINMENT_OUT", "OUT_OF_HOME")),
    (r"\bYoga|aerobics|weightlifting|working out|Running|Hiking|Walking\b", ("OUTDOOR_EXERCISE", "OUT_OF_HOME")),

    # Religion
    (r"\breligious|spiritual\b", ("OTHER_NON_ELEC", "NON_ELECTRIC")),

    # Phone calls
    (r"\bTelephone calls?\b", ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING")),
]

# Prefix defaults for broad families (near-last resort)
PREFIX_DEFAULTS: List[Tuple[str, Tuple[str, str]]] = [
    ("18", ("TRAVEL", "OUT_OF_HOME")),
    ("05", ("WORK_SCHOOL", "OUT_OF_HOME")),
    ("06", ("WORK_SCHOOL", "OUT_OF_HOME")),
    ("07", ("SHOPPING", "OUT_OF_HOME")),
    ("10", ("GOV_SERVICES", "OUT_OF_HOME")),
    ("11", ("EAT_DRINK", "NON_ELECTRIC")),
    ("14", ("OTHER_NON_ELEC", "NON_ELECTRIC")),
    ("15", ("OTHER_NON_ELEC", "NON_ELECTRIC")),
    ("16", ("WORK_ENTERTAINMENT", "ELECTRICITY_CONSUMING")),
    ("50", ("OTHER_NON_ELEC", "NON_ELECTRIC")),
]

# Conservative fallback by major2
MAJOR2_FALLBACK: Dict[str, Tuple[str, str]] = {
    "01": ("PERSONAL_CARE", "NON_ELECTRIC"),
    "02": ("OTHER_NON_ELEC", "NON_ELECTRIC"),
    "03": ("OTHER_NON_ELEC", "NON_ELECTRIC"),
    "04": ("OTHER_NON_ELEC", "NON_ELECTRIC"),
    "08": ("OTHER_NON_ELEC", "NON_ELECTRIC"),
    "09": ("OTHER_NON_ELEC", "NON_ELECTRIC"),
    "12": ("OTHER_NON_ELEC", "NON_ELECTRIC"),
    "13": ("OUTDOOR_EXERCISE", "OUT_OF_HOME"),
}


# ----------------------------- Core mapping ----------------------------------

def normalize_code(c: str) -> str:
    c = str(c).strip()
    c = re.sub(r"[^\d]", "", c)
    return c.zfill(6)


def parse_lexicon(pdf_path: Path) -> pd.DataFrame:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")
    if not pdf_path.exists():
        raise FileNotFoundError(f"Lexicon PDF not found: {pdf_path}")

    code_re = re.compile(r"^(\d{6})\s+(.*)$")
    rows: List[Tuple[str, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw in text.splitlines():
                line = raw.strip()
                m = code_re.match(line)
                if m:
                    code = normalize_code(m.group(1))
                    txt = m.group(2).strip()
                    rows.append((code, txt))
    if not rows:
        raise RuntimeError("No 6-digit codes found in the lexicon PDF")
    df = pd.DataFrame(rows, columns=["code", "text"]).drop_duplicates("code")
    df["major2"] = df["code"].str[:2]
    return df


def apply_rules(code: str, text: str, major2: str) -> Tuple[str, str]:
    code = normalize_code(code)
    t = text or ""

    # 1) Exact override first
    if code in CODE_OVERRIDES:
        return CODE_OVERRIDES[code]

    # 2) Keyword rules
    for pat, (sub, maj) in KEYWORDS:
        if re.search(pat, t, flags=re.I):
            return sub, maj

    # 3) Prefix defaults (major2 groups)
    for pref, (sub, maj) in PREFIX_DEFAULTS:
        if major2 == pref:
            return sub, maj

    # 4) Conservative fallback
    if major2 in MAJOR2_FALLBACK:
        return MAJOR2_FALLBACK[major2]

    # 5) Absolute fallback
    return "OTHER_NON_ELEC", "NON_ELECTRIC"


# ------------------------------- Reporting -----------------------------------

def write_report(df: pd.DataFrame, report_path: Path, strict: bool,
                 strict_threshold: float = 0.20) -> None:
    counts = df["sub_name"].value_counts().sort_values(ascending=False)
    total = int(df.shape[0])

    other_ne = int((df["sub_name"] == "OTHER_NON_ELEC").sum())
    other_elec = int((df["sub_name"] == "OTHER_ELECTRIC").sum())
    ambiguous = 0

    top_other = df[df["sub_name"] == "OTHER_NON_ELEC"][["code", "text"]].head(40)

    lines = []
    lines.append("ATUS Lexicon Mapping Report")
    lines.append("=" * 28)
    lines.append(f"Total rows: {total}\n")
    lines.append("Sub-name distribution:")
    for sub, n in counts.items():
        pct = 100 * n / total if total else 0.0
        lines.append(f"  {sub:<25} {n:>4}  {pct:>5.1f}%")
    lines.append("")
    lines.append(f"OTHER_NON_ELEC: {other_ne:>5} ({other_ne/total:>5.1%})" if total else "OTHER_NON_ELEC: 0 (0.0%)")
    lines.append(f"OTHER_ELECTRIC: {other_elec:>5} ({(other_elec/total if total else 0):>5.1%})")
    lines.append(f"AMBIGUOUS major: {ambiguous:>5} ({0.0:>5.1f}%)")
    lines.append("")
    lines.append("Top 40 codes in OTHER_NON_ELEC (for rule tuning):")
    for _, r in top_other.iterrows():
        lines.append(f"  {r['code']}  {r['text']}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    if strict and total and (other_ne / total) > strict_threshold:
        raise SystemExit(
            f"Strict mode: {other_ne} ({100*other_ne/total:.1f}%) rows mapped to OTHER_NON_ELEC. "
            f"Add more prefix/keyword rules or code overrides."
        )


# --------------------------------- Main --------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to ATUS lexicon PDF (lexiconnoex0323.pdf)")
    ap.add_argument("--out", required=True, help="Path to write CSV mapping")
    ap.add_argument("--report", help="Optional path to write a mapping report")
    ap.add_argument("--strict", action="store_true", help="Fail if OTHER_NON_ELEC share is high")
    ap.add_argument("--strict_threshold", type=float, default=0.20,
                    help="Max OTHER_NON_ELEC share allowed in strict mode (default 0.20)")
    args = ap.parse_args()

    if pdfplumber is None:
        raise SystemExit("pdfplumber not installed. Run: pip install pdfplumber")

    # Assert the key override you asked for:
    assert CODE_OVERRIDES.get("020299") == ("DISHWASHING", "ELECTRICITY_CONSUMING"), \
        "Override 020299 must map to (DISHWASHING, ELECTRICITY_CONSUMING)"

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)

    df = parse_lexicon(pdf_path)

    mapped = df.apply(
        lambda r: apply_rules(r["code"], r["text"], r["major2"]),
        axis=1, result_type="expand"
    )
    mapped.columns = ["sub_name", "major_class"]

    out_df = pd.concat([df[["code", "text", "major2"]], mapped], axis=1)
    out_df["count"] = 0
    out_df = out_df[["code", "text", "major2", "sub_name", "major_class", "count"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✓ Wrote mapping → {out_path}")

    if args.report:
        write_report(out_df, Path(args.report), strict=args.strict, strict_threshold=args.strict_threshold)
        print(f"✓ Wrote report  → {args.report}")


if __name__ == "__main__":
    main()
