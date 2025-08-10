#!/usr/bin/env python3
"""
Build 10-min Markov-ready activity sequences using a compact 14-state taxonomy.

Inputs (defaults relative to repo root):
- data/sequences/grid_10min.npy           # from make_sequences.py
- data/sequences/grid_ids.csv             # TUCASEID order for the grid
- data/sequences/lexiconnoex0323.pdf      # ATUS activity lexicon (2003–2023)


Outputs (same filenames/structure as the original script):
- data/sequences/markov_sequences.parquet  (long, slot-level with state ids)
- data/sequences/states_10min.npy          (N x 144, flat state id per slot)
- data/sequences/state_catalog.json        (14-state catalog + id map)
- data/sequences/transition_counts.csv     (K x K counts)
- data/sequences/transition_probs.csv      (K x K row-normalized)

Major ideas:
- We collapse to 14 states to clarify the electricity signal while keeping temporal dynamics.
- All out-of-home activities are a single OUT_OF_HOME state, unless a specific override keeps them at home.
- We pin important semantics:
  * 020299 “Food & drink prep, presentation, & clean-up, n.e.c.” → DISHWASHING
  * 020101 “Interior cleaning” → CLEANING_ELECTRIC
"""

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# Optional PDF parsing
try:
    import pdfplumber
except Exception:
    pdfplumber = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEQ  = DATA / "sequences"

# ------------------------- 14-STATE TAXONOMY ---------------------------------
# Major classes are coarse energy/location labels; states are the Markov chain.
MAJOR = {
    "ELECTRICITY_CONSUMING": 1,   # at-home electricity-consuming
    "NON_ELECTRIC":          2,   # at-home non-electric
    "OUT_OF_HOME":           3,   # everything away from home
}

# State IDs are STABLE and used in matrices; do not renumber casually.
STATES: List[Tuple[str, str, int]] = [
    # At-home electricity-consuming (7)
    ("COOKING",                   "ELECTRICITY_CONSUMING", 1),
    ("DISHWASHING",               "ELECTRICITY_CONSUMING", 2),
    ("LAUNDRY_IRON",              "ELECTRICITY_CONSUMING", 3),
    ("CLEANING_ELECTRIC",         "ELECTRICITY_CONSUMING", 4),
    ("SCREENS_LEISURE",           "ELECTRICITY_CONSUMING", 5),
    ("ADMIN_ON_DEVICES",          "ELECTRICITY_CONSUMING", 6),
    ("APPLIANCE_HOUSEHOLD_ELEC",  "ELECTRICITY_CONSUMING", 7),

    # At-home non-electric (6)
    ("SLEEP",                     "NON_ELECTRIC",          8),
    ("EAT_DRINK",                 "NON_ELECTRIC",          9),
    ("PERSONAL_CARE",             "NON_ELECTRIC",          10),
    ("CARE_AT_HOME",              "NON_ELECTRIC",          11),
    ("QUIET_SOCIAL",              "NON_ELECTRIC",          12),
    ("EXERCISE_NO_MACHINE",       "NON_ELECTRIC",          13),

    # Out of home (1)
    ("OUT_OF_HOME",               "OUT_OF_HOME",           14),
]
SUB_TO_MAJOR = {name: maj for name, maj, _ in STATES}
SUB_TO_ID    = {name: sid for name, _, sid in STATES}
ID_TO_SUB    = {sid: name for name, _, sid in STATES}

# ---------------------- LEXICON PARSING (PDF) --------------------------------
def parse_lexicon(pdf_path: Path) -> pd.DataFrame:
    """
    Parse ATUS activity lexicon PDF. Returns DataFrame with:
        code (6-digit string), text (full row text), major2 (first two digits)
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Lexicon PDF not found: {pdf_path}")
    if pdfplumber is None:
        raise ImportError("pdfplumber not installed. Try: pip install pdfplumber")

    code_re = re.compile(r"^(\d{6})\s+(.*)$")
    rows: List[Tuple[str, str]] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            for line in txt.splitlines():
                line = line.strip()
                m = code_re.match(line)
                if m:
                    rows.append((m.group(1), m.group(2).strip()))

    if not rows:
        raise RuntimeError("No activity codes parsed from the lexicon; check PDF quality.")

    df = pd.DataFrame(rows, columns=["code", "text"])
    df["major2"] = df["code"].str[:2]
    return df

# -------------------------- MAPPING RULES ------------------------------------
def build_rules() -> Dict:
    """
    Heuristic rules to map ATUS (code, text) → 14-state sub_name.
    Precedence: 1) exact code overrides, 2) precise prefixes, 3) keywords,
    4) major-based fallbacks.
    """
    # -- Exact code overrides where semantics matter for energy/taxonomy --
    code_overrides: Dict[str, str] = {
        # Sleep / personal care
        "010101": "SLEEP", "010102": "SLEEP", "010199": "SLEEP",

        # Eating
        "110101": "EAT_DRINK", "110199": "EAT_DRINK",

        # Dishwashing & kitchen clean-up NEC pinned to cleaning bucket
        "020203": "DISHWASHING",
        "020299": "DISHWASHING",

        # Interior cleaning should not fall to OTHER/non-electric
        "020101": "CLEANING_ELECTRIC",

        # Laundry & ironing family (folding included)
        "020301": "LAUNDRY_IRON",
        "020302": "LAUNDRY_IRON",
        "020303": "LAUNDRY_IRON",

        # Screens/leisure at home
        "120303": "SCREENS_LEISURE",  # TV/movies (non-religious)
        "120304": "SCREENS_LEISURE",  # TV (religious)
        "120305": "SCREENS_LEISURE",  # Radio
        "120306": "SCREENS_LEISURE",  # Listening/playing music (not radio)
        "120307": "SCREENS_LEISURE",  # Games
        "120308": "SCREENS_LEISURE",  # Computer for leisure
        # Telephone calls (we keep with screens/devices signal)
        "160101": "SCREENS_LEISURE",
        "160102": "SCREENS_LEISURE",
        "160103": "SCREENS_LEISURE",
        "160104": "SCREENS_LEISURE",
        "160105": "SCREENS_LEISURE",
        "160106": "SCREENS_LEISURE",
        "160107": "SCREENS_LEISURE",
        "160108": "SCREENS_LEISURE",

        # Household admin on devices
        "020901": "ADMIN_ON_DEVICES",  # Household management
        "020902": "ADMIN_ON_DEVICES",  # Financial management
        "020903": "ADMIN_ON_DEVICES",  # Household/personal org & planning
        "020904": "ADMIN_ON_DEVICES",  # HH & personal email/messages

        # Appliances/tools, quick HVAC/light adjustments
        "020801": "APPLIANCE_HOUSEHOLD_ELEC",
        "020701": "APPLIANCE_HOUSEHOLD_ELEC",  # vehicle repair/maintenance (by self)
        "020703": "APPLIANCE_HOUSEHOLD_ELEC",
        "020303": "APPLIANCE_HOUSEHOLD_ELEC",  # “Heating & cooling” if present

        # Care at home (non-electric): child/adult help/play/reading/etc.
        # (But medical services/obtaining care handled in OUT_OF_HOME section below)
        "030101": "CARE_AT_HOME",
        "030102": "CARE_AT_HOME",
        "030103": "CARE_AT_HOME",
        "030104": "CARE_AT_HOME",
        "030105": "CARE_AT_HOME",
        "030109": "CARE_AT_HOME",
        "030111": "CARE_AT_HOME",
        "030112": "CARE_AT_HOME",
        "030186": "CARE_AT_HOME",
        "030199": "CARE_AT_HOME",
        "030201": "CARE_AT_HOME",
        "030202": "CARE_AT_HOME",
        "030203": "CARE_AT_HOME",
        "030204": "CARE_AT_HOME",
        "030299": "CARE_AT_HOME",
        "030401": "CARE_AT_HOME",
        "030402": "CARE_AT_HOME",
        "030405": "CARE_AT_HOME",
        "030499": "CARE_AT_HOME",
        "030501": "CARE_AT_HOME",
        "030502": "CARE_AT_HOME",
        "030503": "CARE_AT_HOME",
        "030504": "CARE_AT_HOME",
        "030599": "CARE_AT_HOME",
        "039999": "CARE_AT_HOME",

        "040101": "CARE_AT_HOME",
        "040102": "CARE_AT_HOME",
        "040103": "CARE_AT_HOME",
        "040104": "CARE_AT_HOME",
        "040105": "CARE_AT_HOME",
        "040109": "CARE_AT_HOME",
        "040111": "CARE_AT_HOME",
        "040112": "CARE_AT_HOME",
        "040186": "CARE_AT_HOME",
        "040199": "CARE_AT_HOME",
        "040201": "CARE_AT_HOME",
        "040202": "CARE_AT_HOME",
        "040203": "CARE_AT_HOME",
        "040204": "CARE_AT_HOME",
        "040299": "CARE_AT_HOME",
        "040401": "CARE_AT_HOME",
        "040402": "CARE_AT_HOME",
        "040405": "CARE_AT_HOME",
        "040499": "CARE_AT_HOME",
        "040507": "CARE_AT_HOME",
        "040508": "CARE_AT_HOME",
        "040599": "CARE_AT_HOME",
        "049999": "CARE_AT_HOME",

        # Quiet/social at home: reading/writing/crafts, socializing at home, relaxing
        "120101": "QUIET_SOCIAL",
        "120199": "QUIET_SOCIAL",
        "120301": "QUIET_SOCIAL",
        "120302": "QUIET_SOCIAL",
        "120309": "QUIET_SOCIAL",
        "120310": "QUIET_SOCIAL",
        "120311": "QUIET_SOCIAL",
        "120312": "QUIET_SOCIAL",
        "120313": "QUIET_SOCIAL",
        "120399": "QUIET_SOCIAL",

        # Exercise w/o machines that we want at-home non-electric
        "130101": "EXERCISE_NO_MACHINE",  # aerobics
        "130109": "EXERCISE_NO_MACHINE",  # dancing
        "130136": "EXERCISE_NO_MACHINE",  # yoga
    }

    # -- Prefix rules (coarser than exact, but stronger than keywords) --
    prefix_rules: Dict[str, str] = {
        # Cooking family
        "020201": "COOKING",
        "020202": "COOKING",
        "0202":   "COOKING",   # default for prep/presentation (clean-up handled by overrides)

        # Laundry/iron family
        "02030":  "LAUNDRY_IRON",
        "0203":   "LAUNDRY_IRON",

        # Interior cleaning to electric cleaning bucket (vacuum/steam, etc.)
        "020101": "CLEANING_ELECTRIC",

        # Screens/leisure: all 120303–120308 already in overrides; also catch 1601x calls
        "1601":   "SCREENS_LEISURE",

        # Admin on devices: household management cluster
        "02090":  "ADMIN_ON_DEVICES",
        "0209":   "ADMIN_ON_DEVICES",

        # Appliances/tools & quick power uses
        "0208":   "APPLIANCE_HOUSEHOLD_ELEC",
        "0207":   "APPLIANCE_HOUSEHOLD_ELEC",
    }

    # -- Keyword fallbacks (only if no exact/prefix match) --
    kw_to_sub: List[Tuple[str, str]] = [
        (r"\b(cook|stove|oven|microwave|air[-\s]*fryer|grill|bake|saute|boil)\b", "COOKING"),
        (r"\b(dish(wash|washer)|kitchen\s*clean[-\s]*up|clean[-\s]*up)\b",        "DISHWASHING"),
        (r"\b(laundry|washer|dryer|fold|folding|iron|ironing)\b",                "LAUNDRY_IRON"),
        (r"\b(vacuum|steam\s*mop|carpet\s*clean)\b",                              "CLEANING_ELECTRIC"),
        (r"\b(tv|television|video|stream|netflix|youtube|music|radio|podcast|"
         r"console|xbox|playstation|nintendo|game|computer|laptop|desktop|phone|call)\b", "SCREENS_LEISURE"),
        (r"\b(bill|bills|bank|tax|form|paperwork|email|e[-\s]*mail|message|plan|schedule|"
         r"application|register|print|printer|scan|scanner)\b",                   "ADMIN_ON_DEVICES"),
        (r"\b(hvac|thermostat|air\s*condition|heater|power\s*tool|appliance|drill|saw|repair|maintenance)\b",
                                                                                  "APPLIANCE_HOUSEHOLD_ELEC"),
        (r"\b(sleep|nap|sleepless)\b",                                           "SLEEP"),
        (r"\b(eat|drin[k]|meal|breakfast|lunch|dinner)\b",                       "EAT_DRINK"),
        (r"\b(bathe|bath|shower|toilet|groom|makeup|hair|wash(ing)?\b.*self)\b", "PERSONAL_CARE"),
        (r"\b(child\s*care|childcare|babysit|help(ing)?\s*(child|adult)|"
         r"playing\s*with\s*(child|children)|reading\s*to)\b",                   "CARE_AT_HOME"),
        (r"\b(read|book|paper|newspaper|write|journal|craft|knit|board\s*game|"
         r"card\s*game|chat|talk|relax|meditat(e|ion)|tobacco|smok(e|ing))\b",   "QUIET_SOCIAL"),
        (r"\b(yoga|stretch|calisthenic(s)?)\b",                                   "EXERCISE_NO_MACHINE"),
    ]

    # -- Medical care / obtaining services are out of home in this compact model --
    out_of_home_exact = {
        # Travel block handled structurally; here we cover care/medical service codes etc.
        "030301", "030302", "030303",  # children medical care/obtaining/waiting
        "040301", "040302", "040303",  # nonhh children medical care/obtaining/waiting
        "030403", "030404",            # hh adults medical/obtaining
        "040403", "040404",            # nonhh adults medical/obtaining
    }

    # Major prefixes that default to OUT_OF_HOME unless overridden
    out_of_home_majors = {"05", "06", "07", "08", "09", "10", "13", "14", "15", "18"}

    return dict(
        code_overrides=code_overrides,
        prefix_rules=prefix_rules,
        kw_to_sub=kw_to_sub,
        out_of_home_exact=out_of_home_exact,
        out_of_home_majors=out_of_home_majors
    )

def map_code_to_sub(code: str, text: str, rules: Dict) -> Tuple[str, str]:
    """
    Map a (code, text) to (major_class, sub_name) in the 14-state taxonomy.
    Precedence: exact override > prefix > keyword > major fallback.
    """
    code = (code or "").strip()
    text = (text or "").strip()
    if not (len(code) == 6 and code.isdigit()):
        return "OUT_OF_HOME", "OUT_OF_HOME"  # safest default

    major2 = code[:2]

    # 1) Exact overrides
    if code in rules["code_overrides"]:
        sub = rules["code_overrides"][code]
        return SUB_TO_MAJOR[sub], sub

    # 2) Structural OUT_OF_HOME
    if major2 in rules["out_of_home_majors"]:
        # except exact overrides above
        return "OUT_OF_HOME", "OUT_OF_HOME"

    # 3) Specific OUT_OF_HOME exacts (medical/obtaining)
    if code in rules["out_of_home_exact"]:
        return "OUT_OF_HOME", "OUT_OF_HOME"

    # 4) Prefix rules (longest-first)
    pref = rules["prefix_rules"]
    for L in range(6, 1, -1):
        p = code[:L]
        if p in pref:
            sub = pref[p]
            return SUB_TO_MAJOR[sub], sub

    # 5) Keyword fallbacks
    for pat, sub in rules["kw_to_sub"]:
        if re.search(pat, text, flags=re.I):
            return SUB_TO_MAJOR[sub], sub

    # 6) Remaining major heuristics (in-home vs out-of-home)
    if major2 == "01":  # personal care block
        return "NON_ELECTRIC", "PERSONAL_CARE"
    if major2 == "11":  # eating & drinking
        return "NON_ELECTRIC", "EAT_DRINK"
    if major2 == "12":  # relaxing/leisure general
        return "ELECTRICITY_CONSUMING", "SCREENS_LEISURE"
    if major2 == "16":  # phone calls
        return "ELECTRICITY_CONSUMING", "SCREENS_LEISURE"
    if major2 == "02":  # household activities fallback
        return "ELECTRICITY_CONSUMING", "APPLIANCE_HOUSEHOLD_ELEC"

    # Else, safest compact assignment
    return "OUT_OF_HOME", "OUT_OF_HOME"

# ------------------------- AUDIT UTILITY -------------------------------------
def verify_and_export_assignments(lex_df: pd.DataFrame,
                                  rules: Dict,
                                  out_csv: str = str(SEQ / "code_assignments_audit_14state.csv")):
    """
    For every 6-digit code in the lexicon, assign (major, sub, state_id) and
    export a CSV with the triggering path. This is helpful for rule tuning.
    """
    def assign_with_hint(code: str, text: str) -> Tuple[str, str, str]:
        code = (code or "").strip()
        text = (text or "").strip()
        if code in rules["code_overrides"]:
            sub = rules["code_overrides"][code]; return SUB_TO_MAJOR[sub], sub, "OVERRIDE"
        if code[:2] in rules["out_of_home_majors"]:
            return "OUT_OF_HOME", "OUT_OF_HOME", "MAJOR_OUT"
        if code in rules["out_of_home_exact"]:
            return "OUT_OF_HOME", "OUT_OF_HOME", "OUT_EXACT"
        pref = rules["prefix_rules"]
        for L in range(6, 1, -1):
            p = code[:L]
            if p in pref:
                sub = pref[p]; return SUB_TO_MAJOR[sub], sub, f"PREFIX_{L}"
        for pat, sub in rules["kw_to_sub"]:
            if re.search(pat, text, flags=re.I):
                return SUB_TO_MAJOR[sub], sub, "KW"
        # fallbacks
        m, s = map_code_to_sub(code, text, rules)
        return m, s, "FALLBACK"

    rows = []
    for r in lex_df.itertuples(index=False):
        maj, sub, hint = assign_with_hint(r.code, r.text)
        sid = SUB_TO_ID[sub]
        rows.append((r.code, r.text, r.major2, maj, sub, sid, hint))

    audit = pd.DataFrame(rows, columns=["code","text","major2","major","sub","state_id","hint"])
    audit.to_csv(out_csv, index=False)
    print(f"✓ Wrote lexicon audit → {out_csv}")

# ----------------------------- PIPELINE --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, default=str(SEQ / "grid_10min.npy"))
    ap.add_argument("--ids",  type=str, default=str(SEQ / "grid_ids.csv"))
    ap.add_argument("--lex",  type=str, default=str(SEQ / "lexiconnoex0323.pdf"))
    ap.add_argument("--out_parquet", type=str, default=str(SEQ / "markov_sequences.parquet"))
    ap.add_argument("--out_states",  type=str, default=str(SEQ / "states_10min.npy"))
    ap.add_argument("--out_counts",  type=str, default=str(SEQ / "transition_counts.csv"))
    ap.add_argument("--out_probs",   type=str, default=str(SEQ / "transition_probs.csv"))
    ap.add_argument("--catalog",     type=str, default=str(SEQ / "state_catalog.json"))
    ap.add_argument("--audit", action="store_true", help="Export code->assignment audit CSV for verification")
    args = ap.parse_args()

    SEQ.mkdir(parents=True, exist_ok=True)

    # Load grid + ids
    X   = np.load(args.grid)                        # [N, 144] int codes
    ids = pd.read_csv(args.ids)["TUCASEID"].to_numpy()

    if X.shape[0] != len(ids):
        raise ValueError("grid_10min.npy and grid_ids.csv length mismatch")

    # Enforce 10-minute resolution
    if X.shape[1] != 144:
        raise ValueError(
            f"Expected 10-min grids with 144 slots/day, got {X.shape[1]}. "
            "Please regenerate grids at 10-min resolution."
        )

    # Parse lexicon and build rules
    lex_df = parse_lexicon(Path(args.lex))
    rules  = build_rules()
    if args.audit:
        verify_and_export_assignments(lex_df, rules)

    code_to_text = dict(zip(lex_df["code"], lex_df["text"]))

    # Vectorized mapping
    N, S = X.shape
    codes_str = np.char.zfill(X.astype(str), 6)

    df = pd.DataFrame({
        "TUCASEID": np.repeat(ids, S),
        "slot":     np.tile(np.arange(S, dtype=np.int16), N),
        "code":     codes_str.reshape(-1)
    })
    df["text"] = df["code"].map(code_to_text)

    mapped = df[["code","text"]].fillna("").apply(
        lambda r: map_code_to_sub(r["code"], r["text"], rules),
        axis=1
    )
    df[["major_class","sub_name"]] = pd.DataFrame(mapped.tolist(), index=df.index)

    # Attach flat state id; everything is known in 14-state map
    df["state_id"] = df["sub_name"].map(SUB_TO_ID).astype("int16")
    df["major_id"] = df["major_class"].map(MAJOR).astype("int8")

    # Save long table
    out_parquet = Path(args.out_parquet)
    df[["TUCASEID","slot","code","major_class","major_id","sub_name","state_id"]].to_parquet(
        out_parquet, index=False
    )

    # Save N x 144 matrix of flat state IDs
    states = df["state_id"].to_numpy().reshape(N, S)
    np.save(args.out_states, states)

    # Build transition counts/probs (first-order Markov)
    K = max(SUB_TO_ID.values()) + 1  # 15, but we use ids up to 14
    counts = np.zeros((K, K), dtype=np.int64)
    for n in range(N):
        seq = states[n]
        src = seq[:-1]
        dst = seq[1:]
        np.add.at(counts, (src, dst), 1)

    # Row-normalize to probabilities (safe for zero rows)
    probs = counts.astype(float)
    row_sums = probs.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(row_sums > 0, probs / row_sums, 0.0)

    # Save CSV matrices with human-readable headers
    id_order = [i for i in sorted(ID_TO_SUB.keys())]
    col_names = [ID_TO_SUB.get(i, f"STATE_{i}") for i in id_order]
    idx_names = col_names
    pd.DataFrame(counts[id_order][:, id_order], index=idx_names, columns=col_names).to_csv(args.out_counts)
    pd.DataFrame(probs[id_order][:, id_order],  index=idx_names, columns=col_names).to_csv(args.out_probs)

    # Save catalog (majors, subs, mapping)
    catalog = {
        "major": MAJOR,
        "sub_to_major": SUB_TO_MAJOR,
        "sub_to_id": SUB_TO_ID,
        "id_to_sub": ID_TO_SUB,
        "notes": {
            "taxonomy": "14-state compact (7 at-home electric, 6 at-home non-electric, 1 out-of-home)",
            "pin_020299": "DISHWASHING (clean-up n.e.c. stays in cleaning bucket)",
            "pin_020101": "CLEANING_ELECTRIC (interior cleaning not OTHER)",
            "out_of_home": "All away-from-home majors collapsed to OUT_OF_HOME"
        }
    }
    with open(args.catalog, "w") as f:
        json.dump(catalog, f, indent=2)

    print("✓ Wrote:")
    print(f"  {out_parquet}")
    print(f"  {args.out_states}")
    print(f"  {args.out_counts}")
    print(f"  {args.out_probs}")
    print(f"  {args.catalog}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)
