#!/usr/bin/env python3
"""
Make respondent subgroups for ATUS analysis.

Run with NO arguments (defaults) or override via CLI.
- Input (default):   data/processed/respondent.parquet
- Outputs (default): data/processed/subgroups.parquet
                     data/processed/subgroups_summary.parquet
                     data/processed/subgroups_schema.json

Derived subgroup columns (all created here if missing):
  * sex            (from TESEX: 1=Male, 2=Female)
  * employment     (from PRWKSTAT or PEMLR or hours proxies)
  * day_type       (from TUDIARYDAY: weekend vs weekday)
  * hh_size_band   (from TRNUMHOU or HRNUMHOU → {"1","2","3","4plus"})
  * region         (from GEREG: {1,2,3,4} → {"Northeast","Midwest","South","West"})
  * month          (from TUMONTH: 1..12)

Weights:
  * Prefers TUFNWGTP (ATUS final person weight). If absent, falls back to TU20FWGT.
  * Weighted counts in the subgroup summary use this column.

CLI examples
------------
# Run with defaults (recommended first run)
python atus_analysis/scripts/make_subgroups.py

# Override inputs/outputs
python atus_analysis/scripts/make_subgroups.py \
  --respondents atus_analysis/data/processed/respondent.parquet \
  --out_parquet atus_analysis/data/processed/subgroups.parquet \
  --out_summary atus_analysis/data/processed/subgroups_summary.parquet \
  --out_schema  atus_analysis/data/processed/subgroups_schema.json

# Choose different grouping fields (space-separated, thanks to nargs="+")
python atus_analysis/scripts/make_subgroups.py \
  --groupby sex employment region

"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


# ---------- Repo-relative defaults ----------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if (HERE.name.endswith(".py") and (len(HERE.parents) >= 2)) else Path(".").resolve()
DATA = ROOT / "atus_analysis" /"data"
PROCESSED = DATA / "processed"

DEFAULT_IN  = PROCESSED / "respondent.parquet"
DEFAULT_OUT_PARQUET = PROCESSED / "subgroups.parquet"
DEFAULT_OUT_SUMMARY = PROCESSED / "subgroups_summary.parquet"
DEFAULT_OUT_SCHEMA  = PROCESSED / "subgroups_schema.json"

DEFAULT_GROUPBY = ["sex", "employment", "day_type", "hh_size_band", "region", "month"]


# ---------- Helpers to derive subgroup columns ----------

def pick_weight_column(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Return (weight_col, note). Prefers TUFNWGTP; falls back to TU20FWGT.
    """
    if "TUFNWGTP" in df.columns:
        return "TUFNWGTP", "Using ATUS final weight: TUFNWGTP"
    if "TU20FWGT" in df.columns:
        return "TU20FWGT", "Using ATUS 2020+ final weight: TU20FWGT (fallback)"
    # As a last resort, create unit weight
    df["UNIT_WEIGHT"] = 1.0
    return "UNIT_WEIGHT", "No ATUS weight found; using UNIT_WEIGHT=1.0"


def derive_sex(df: pd.DataFrame) -> pd.Series:
    """
    From TESEX: 1=Male, 2=Female; else 'Unknown'
    """
    if "TESEX" in df.columns:
        return df["TESEX"].map({1: "Male", 2: "Female"}).fillna("Unknown")
    return pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")


def derive_day_type(df: pd.DataFrame) -> pd.Series:
    """
    From TUDIARYDAY: 1=Sunday,...,7=Saturday. Weekend if day in {1,7}.
    """
    if "TUDIARYDAY" in df.columns:
        return np.where(df["TUDIARYDAY"].isin([1, 7]), "Weekend", "Weekday")
    return pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")


def derive_month(df: pd.DataFrame) -> pd.Series:
    """
    From TUMONTH (1..12). Keep numeric 1..12 as strings for grouping.
    """
    if "TUMONTH" in df.columns:
        return df["TUMONTH"].where(df["TUMONTH"].between(1, 12)).astype("Int64").astype("string").fillna("Unknown")
    return pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")


def derive_hh_size_band(df: pd.DataFrame) -> pd.Series:
    """
    Household size band from TRNUMHOU (preferred) or HRNUMHOU (CPS core).
    Bands: '1','2','3','4plus'
    """
    source = None
    if "TRNUMHOU" in df.columns:
        source = df["TRNUMHOU"]
    elif "HRNUMHOU" in df.columns:
        source = df["HRNUMHOU"]
    else:
        return pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")

    vals = pd.to_numeric(source, errors="coerce")
    band = np.where(vals == 1, "1",
             np.where(vals == 2, "2",
             np.where(vals == 3, "3",
             np.where(vals >= 4, "4plus", "Unknown"))))
    return pd.Series(band, index=df.index, dtype="object")


def derive_region(df: pd.DataFrame) -> pd.Series:
    """
    From GEREG: 1=Northeast, 2=Midwest, 3=South, 4=West
    """
    if "GEREG" in df.columns:
        return df["GEREG"].map({1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}).fillna("Unknown")
    return pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")


def derive_employment(df: pd.DataFrame) -> pd.Series:
    """
    Employment from CPS status if available (prefer PRWKSTAT, fallback PEMLR).
    Otherwise, use hours proxies (TEHRUSLT/TEHRUSL1/TEHRUSL2).
    Labels: 'Employed', 'Unemployed', 'NotInLF', 'Unknown'

    PRWKSTAT common coding (CPS core):
      1 Full-time, 2 Part-time, 3 Hours vary, 4 Has job, not at work
      5 Unemployed, 6 Not in labor force
    PEMLR (monthly labor force recode):
      1 employed-absent, 2 employed-at work, 3 unemployed (on layoff),
      4 unemployed (looking), 5 NILF
    """
    if "PRWKSTAT" in df.columns:
        return pd.Series(
            np.select(
                [
                    df["PRWKSTAT"].isin([1, 2, 3, 4]),
                    df["PRWKSTAT"].isin([5]),
                    df["PRWKSTAT"].isin([6]),
                ],
                ["Employed", "Unemployed", "NotInLF"],
                default="Unknown",
            ),
            index=df.index,
            dtype="object",
        )

    if "PEMLR" in df.columns:
        return pd.Series(
            np.select(
                [
                    df["PEMLR"].isin([1, 2]),   # employed
                    df["PEMLR"].isin([3, 4]),   # unemployed
                    df["PEMLR"].isin([5]),      # NILF
                ],
                ["Employed", "Unemployed", "NotInLF"],
                default="Unknown",
            ),
            index=df.index,
            dtype="object",
        )

    # Hours proxies (ATUS variables; conservative fallback)
    hrs = (
        df.get("TEHRUSLT", pd.Series(0, index=df.index)).fillna(0).astype("float")
        + df.get("TEHRUSL1", pd.Series(0, index=df.index)).fillna(0).astype("float")
        + df.get("TEHRUSL2", pd.Series(0, index=df.index)).fillna(0).astype("float")
    )
    return pd.Series(np.where(hrs > 0, "Employed", "Unknown"), index=df.index, dtype="object")


def ensure_subgroup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add/overwrite the canonical subgroup columns based on available inputs.
    """
    out = df.copy()
    out["sex"] = derive_sex(out)
    out["employment"] = derive_employment(out)
    out["day_type"] = derive_day_type(out)
    out["hh_size_band"] = derive_hh_size_band(out)
    out["region"] = derive_region(out)
    out["month"] = derive_month(out)
    return out


def summarize_groups(df: pd.DataFrame, groupby: List[str], weight_col: str) -> pd.DataFrame:
    """
    Return summary with raw N and weighted_N per group.
    """
    # Guard: any missing group columns?
    missing = [g for g in groupby if g not in df.columns]
    if missing:
        raise KeyError(f"Missing grouping columns: {missing}")

    # Weighted counts
    g = df.groupby(groupby, dropna=False, as_index=False)
    summary = g.agg(
        n=("TUCASEID", "size"),
        weight_sum=(weight_col, "sum")
    )

    # Sort for readability
    return summary.sort_values(groupby).reset_index(drop=True)


# ---------- Main pipeline ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--respondents", type=str, default=str(DEFAULT_IN),
                    help=f"Respondent parquet (default: {DEFAULT_IN})")
    ap.add_argument("--out_parquet", type=str, default=str(DEFAULT_OUT_PARQUET),
                    help=f"Per-respondent output parquet (default: {DEFAULT_OUT_PARQUET})")
    ap.add_argument("--out_summary", type=str, default=str(DEFAULT_OUT_SUMMARY),
                    help=f"Group summary parquet (default: {DEFAULT_OUT_SUMMARY})")
    ap.add_argument("--out_schema", type=str, default=str(DEFAULT_OUT_SCHEMA),
                    help=f"JSON schema (default: {DEFAULT_OUT_SCHEMA})")
    ap.add_argument("--groupby", nargs="+", default=DEFAULT_GROUPBY,
                    help=f"Grouping columns; space-separated list (default: {' '.join(DEFAULT_GROUPBY)})")
    ap.add_argument("--weight_col", type=str, default=None,
                    help="Weight column to use; if omitted, auto-detect (TUFNWGTP→TU20FWGT→UNIT_WEIGHT)")

    args = ap.parse_args()

    in_path = Path(args.respondents).resolve()
    out_parquet = Path(args.out_parquet).resolve()
    out_summary = Path(args.out_summary).resolve()
    out_schema  = Path(args.out_schema).resolve()

    # Ensure output dirs exist
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_schema.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load respondents
    if not in_path.exists():
        raise FileNotFoundError(f"Respondent parquet not found: {in_path}")
    df = pd.read_parquet(in_path)

    # 2) Derive subgroup columns
    df = ensure_subgroup_columns(df)

    # 3) Pick weight col
    if args.weight_col:
        weight_col = args.weight_col
        if weight_col not in df.columns:
            raise KeyError(f"--weight_col {weight_col} not found in respondents.")
        weight_note = f"Using specified weight: {weight_col}"
    else:
        weight_col, weight_note = pick_weight_column(df)

    # 4) Keep minimal fields for downstream merges
    keep_cols = ["TUCASEID"] + list(set(args.groupby)) + [weight_col]
    # Some files may use string TUCASEID; enforce consistent type
    if "TUCASEID" in df.columns and not np.issubdtype(df["TUCASEID"].dtype, np.number):
        df["TUCASEID"] = df["TUCASEID"].astype(str)

    slim = df.loc[:, [c for c in keep_cols if c in df.columns]].copy()

    # 5) Save per-respondent subgroup labels
    slim.to_parquet(out_parquet, index=False)

    # 6) Summary table (raw + weighted)
    summary = summarize_groups(slim, args.groupby, weight_col)
    summary.to_parquet(out_summary, index=False)

    # 7) Schema/metadata
    schema: Dict = {
        "inputs": {
            "respondents": str(in_path)
        },
        "outputs": {
            "per_respondent": str(out_parquet),
            "summary": str(out_summary),
        },
        "groupby": args.groupby,
        "weight": {
            "column": weight_col,
            "note": weight_note
        },
        "derivations": {
            "sex": "TESEX (1=Male,2=Female)",
            "employment": "PRWKSTAT→PEMLR→TEHRUSLT/L1/L2 fallback",
            "day_type": "TUDIARYDAY weekend={1,7}",
            "hh_size_band": "TRNUMHOU→HRNUMHOU mapped to {'1','2','3','4plus'}",
            "region": "GEREG {1,2,3,4} mapped to US Census regions",
            "month": "TUMONTH 1..12 as string for grouping"
        }
    }
    with open(out_schema, "w") as f:
        json.dump(schema, f, indent=2)

    # 8) Console report
    print("✓ Subgroups built")
    print(f"  Input:     {in_path}")
    print(f"  Weight:    {weight_col} ({weight_note})")
    print(f"  Groupby:   {', '.join(args.groupby)}")
    print(f"  Outputs:")
    print(f"    - {out_parquet}")
    print(f"    - {out_summary}")
    print(f"    - {out_schema}")

    # show a tiny head for sanity
    with pd.option_context("display.width", 120, "display.max_columns", 50):
        print("\nExample rows (per-respondent):")
        print(slim.head(5))
        print("\nExample rows (summary):")
        print(summary.head(10))


if __name__ == "__main__":
    main()
