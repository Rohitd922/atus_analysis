#!/usr/bin/env python3
"""
Build 10-min Markov-ready activity sequences with (major, sub) labels and flat state IDs,
using a precomputed CSV lexicon map (code → text, major2, sub_name, major_class, count).

Inputs (defaults are relative to repo root):
- data/sequences/grid_10min.npy           # from make_sequences.py (N x 144 int ATUS codes)
- data/sequences/grid_ids.csv             # TUCASEID order for grid_10min.npy
- assets/atus_lexicon_map.csv             # generated mapping (code,text,major2,sub_name,major_class,count)

Outputs:
- data/sequences/markov_sequences.parquet  # long, slot-level with state ids
- data/sequences/states_10min.npy          # (N x 144) flat state id per slot
- data/sequences/state_catalog.json        # major/sub catalog + id map
- data/sequences/transition_counts.csv     # (K x K) counts over flat states
- data/sequences/transition_probs.csv      # (K x K) row-normalized probabilities
- data/sequences/unmapped_codes.csv        # any grid codes not found in the CSV map (for QA)
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ------------------- CONSTANTS / CATALOG -------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEQ  = DATA / "sequences"
ASSETS = ROOT / "assets"

# Major classes (stable)
MAJOR = {
    "AMBIGUOUS": 0,
    "ELECTRICITY_CONSUMING": 1,
    "NON_ELECTRIC": 2,
    "OUT_OF_HOME": 3,
}

# Subcategories and their flat Markov state IDs (keep stable across runs)
# NOTE: Make sure these cover all sub_name values in your CSV. Unknown subs fall back safely.
SUB = [
    # ---- Electricity-consuming (anchors first) ----
    ("COOKING",               "ELECTRICITY_CONSUMING", 1),
    ("CLEANING_VACUUM",       "ELECTRICITY_CONSUMING", 2),
    ("ADMIN_AT_HOME",         "ELECTRICITY_CONSUMING", 3),
    ("OUTSIDE_HOUSE",         "OUT_OF_HOME",           4),  # generic outside fallback

    ("DISHWASHING",           "ELECTRICITY_CONSUMING", 5),
    ("LAUNDRY",               "ELECTRICITY_CONSUMING", 6),
    ("IRONING",               "ELECTRICITY_CONSUMING", 7),

    # merged ICT/media/phone/computer/games
    ("WORK_ENTERTAINMENT",    "ELECTRICITY_CONSUMING", 8),

    ("HVAC_ADJUST",           "ELECTRICITY_CONSUMING", 9),
    ("LIGHTING_ONLY",         "ELECTRICITY_CONSUMING", 10),
    ("APPLIANCE_TOOL_USE",    "ELECTRICITY_CONSUMING", 11),
    ("OTHER_ELECTRIC",        "ELECTRICITY_CONSUMING", 12),

    # ---- Non-electric ----
    ("SLEEP",                 "NON_ELECTRIC",          17),
    ("EAT_DRINK",             "NON_ELECTRIC",          18),
    ("PERSONAL_CARE",         "NON_ELECTRIC",          19),
    ("CHILDCARE",             "NON_ELECTRIC",          20),
    ("READ_PAPERWORK_NON_ELEC","NON_ELECTRIC",         21),
    ("SOCIAL_HOME",           "NON_ELECTRIC",          22),
    ("EXERCISE_NO_MACHINE",   "NON_ELECTRIC",          23),
    ("OTHER_NON_ELEC",        "NON_ELECTRIC",          24),

    # ---- Out-of-home detail ----
    ("TRAVEL",                "OUT_OF_HOME",           25),
    ("SHOPPING",              "OUT_OF_HOME",           26),
    ("DINING_OUT",            "OUT_OF_HOME",           27),
    ("WORK_SCHOOL",           "OUT_OF_HOME",           28),
    ("HEALTHCARE_OUT",        "OUT_OF_HOME",           29),
    ("ENTERTAINMENT_OUT",     "OUT_OF_HOME",           30),
    ("OUTDOOR_EXERCISE",      "OUT_OF_HOME",           31),
    ("GOV_SERVICES",          "OUT_OF_HOME",           32),
    ("OTHER_OUT",             "OUT_OF_HOME",           33),
]

SUB_TO_MAJOR: Dict[str, str] = {name: maj for name, maj, _ in SUB}
SUB_TO_ID:    Dict[str, int] = {name: sid for name, _, sid in SUB}
ID_TO_SUB:    Dict[int, str] = {sid: name for name, _, sid in SUB}

# ------------------- IO HELPERS -------------------

def load_grid_and_ids(grid_path: Path, ids_path: Path):
    X = np.load(grid_path)  # N x 144, integer ATUS codes
    ids = pd.read_csv(ids_path, dtype={"TUCASEID": str})["TUCASEID"].to_numpy()
    if X.shape[0] != len(ids):
        raise ValueError("grid_10min.npy and grid_ids.csv length mismatch.")
    if X.shape[1] != 144:
        raise ValueError(f"Expected 144 10-min slots/day, got {X.shape[1]}. Regenerate grids at 10-min resolution.")
    return X, ids

def load_code_map(csv_map_path: Path) -> pd.DataFrame:
    """
    Expect columns: code,text,major2,sub_name,major_class,count
    Ensures 6-digit zero-padded string code.
    """
    df = pd.read_csv(csv_map_path, dtype={"code": str})
    required = {"code", "sub_name", "major_class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mapping CSV missing columns: {missing}")
    df["code"] = df["code"].str.zfill(6)
    return df[["code", "sub_name", "major_class"]].drop_duplicates()

def fallback_state_for_major(major_class: str) -> str:
    """
    If a sub_name in the CSV is not in our SUB catalog, use a consistent fallback
    by major_class so state ids remain stable.
    """
    if major_class == "OUT_OF_HOME":
        return "OUTSIDE_HOUSE"
    if major_class == "NON_ELECTRIC":
        return "OTHER_NON_ELEC"
    if major_class == "ELECTRICITY_CONSUMING":
        return "OTHER_ELECTRIC"
    return "OTHER_OUT"  # very rare / ambiguous

# ------------------- MAIN PIPELINE -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, default=str(SEQ / "grid_10min.npy"))
    ap.add_argument("--ids",  type=str, default=str(SEQ / "grid_ids.csv"))
    ap.add_argument("--map",  type=str, default=str(ASSETS / "atus_lexicon_map.csv"),
                    help="CSV mapping produced by make_lexicon_map.py")
    ap.add_argument("--out_parquet", type=str, default=str(SEQ / "markov_sequences.parquet"))
    ap.add_argument("--out_states",  type=str, default=str(SEQ / "states_10min.npy"))
    ap.add_argument("--out_counts",  type=str, default=str(SEQ / "transition_counts.csv"))
    ap.add_argument("--out_probs",   type=str, default=str(SEQ / "transition_probs.csv"))
    ap.add_argument("--catalog",     type=str, default=str(SEQ / "state_catalog.json"))
    ap.add_argument("--unmapped_out",type=str, default=str(SEQ / "unmapped_codes.csv"))
    args = ap.parse_args()

    # Ensure output dir exists
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    X, ids = load_grid_and_ids(Path(args.grid), Path(args.ids))
    code_map = load_code_map(Path(args.map))

    # Vectorized mapping by merge
    N, S = X.shape
    codes_str = np.char.zfill(X.astype(str), 6)

    df = pd.DataFrame({
        "TUCASEID": np.repeat(ids, S),
        "slot":     np.tile(np.arange(S, dtype=np.int16), N),
        "code":     codes_str.reshape(-1),
    })

    # Merge code → (sub_name, major_class)
    df = df.merge(code_map, on="code", how="left", copy=False)

    # Detect unmapped codes (should be empty if your CSV covers all used codes)
    unmapped = df[df["sub_name"].isna()][["code"]].drop_duplicates().sort_values("code")
    if not unmapped.empty:
        unmapped.to_csv(args.unmapped_out, index=False)
        print(f"Found {len(unmapped)} unmapped codes (wrote {args.unmapped_out}). "
              "They will be assigned by stable fallbacks.")
        # Provide stable fallbacks by major2 structure where sensible
        # Travel (18xxxx) → TRAVEL/OUT_OF_HOME; else by major_class fallback below
        df["major2"] = df["code"].str[:2]
        df.loc[(df["sub_name"].isna()) & (df["major2"] == "18"), ["sub_name", "major_class"]] = ["TRAVEL", "OUT_OF_HOME"]

        # If still NaN after the 18 rule, use generic fallbacks by declared major_class (if any)
        # But we don't have major_class for unmapped rows; assign NON_ELECTRIC by default
        df["major_class"] = df["major_class"].fillna("NON_ELECTRIC")
        # Sub fallback per major
        mask = df["sub_name"].isna()
        df.loc[mask, "sub_name"] = df.loc[mask, "major_class"].map(fallback_state_for_major)

    # Now all rows should have sub_name & major_class
    # Ensure sub_name is in our catalog; otherwise map to major-specific fallback
    df["sub_name"] = df.apply(
        lambda r: r["sub_name"] if r["sub_name"] in SUB_TO_ID
        else fallback_state_for_major(r["major_class"]),
        axis=1
    )

    # Attach ids
    df["state_id"]  = df["sub_name"].map(SUB_TO_ID).astype("int16")
    df["major_id"]  = df["major_class"].map(MAJOR).fillna(MAJOR["AMBIGUOUS"]).astype("int8")

    # Save long table
    out_parquet = Path(args.out_parquet)
    df[["TUCASEID","slot","code","major_class","major_id","sub_name","state_id"]].to_parquet(out_parquet, index=False)

    # Save N x 144 matrix of flat state IDs
    states = df["state_id"].to_numpy().reshape(N, S)
    np.save(args.out_states, states)

    # Build transition counts/probs over all persons (first-order)
    K = 1 + max(ID_TO_SUB.keys())
    counts = np.zeros((K, K), dtype=np.int64)
    for n in range(N):
        seq = states[n]
        src = seq[:-1]
        dst = seq[1:]
        np.add.at(counts, (src, dst), 1)

    probs = counts.astype(float)
    row_sums = probs.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(row_sums > 0, probs / row_sums, 0.0)

    # Save CSV matrices with headers
    id_order = sorted(ID_TO_SUB.keys())
    names = [ID_TO_SUB[i] for i in id_order]
    pd.DataFrame(counts[id_order][:, id_order], index=names, columns=names).to_csv(args.out_counts)
    pd.DataFrame(probs[id_order][:, id_order],  index=names, columns=names).to_csv(args.out_probs)

    # Save a catalog snapshot (useful when loading states later)
    catalog = {
        "major": MAJOR,
        "sub_to_major": SUB_TO_MAJOR,
        "sub_to_id": SUB_TO_ID,
        "id_to_sub": ID_TO_SUB,
        "notes": {
            "anchor_ids": {"COOKING": 1, "CLEANING_VACUUM": 2, "ADMIN_AT_HOME": 3, "OUTSIDE_HOUSE": 4},
            "state_id_is_flat_markov_state": True
        }
    }
    with open(args.catalog, "w") as f:
        json.dump(catalog, f, indent=2)

    # Small mapping QA summary
    summary = (
        df.drop_duplicates(subset=["code"])[["sub_name","major_class"]]
          .value_counts()
          .rename_axis(["sub_name","major_class"])
          .reset_index(name="codes")
          .sort_values(["major_class","sub_name"])
    )
    print("✓ Wrote:")
    print(f"  {out_parquet}")
    print(f"  {args.out_states}")
    print(f"  {args.out_counts}")
    print(f"  {args.out_probs}")
    print(f"  {args.catalog}")
    if not unmapped.empty:
        print(f"  {args.unmapped_out} (unmapped codes)")
    print("\nSub/major coverage (distinct codes):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)
