"""
Create per-respondent test metrics parquet(s) for ATUS models.

Outputs (per rung):
  • test_case_metrics_b1h.parquet (if B1-H model exists or selected)
  • test_case_metrics_b2h.parquet (if B2-H model exists or selected)
  • test_case_metrics.parquet (compatibility write-if-missing)

Each row (per TUCASEID):
  - nll_weighted           = Σ (−log p(true_next | context)) × design_weight
  - top1_correct_weight    = Σ 1[top-1 == true_next] × design_weight
  - weight_total           = Σ design_weight
  - transitions, eval_steps

Features:
  - Uses central fixed split at <models_root>/fixed_split.parquet, fallbacks to <run_dir>/split_assignments.parquet.
  - Supports B1-H (routing) and B2-H (routing + hazard).
  - Prefers 144-slot routing sidecar (b1h_slot_matz.npz) if present (per-group or global); otherwise falls back to block/global matrices.
  - K (#states) inferred from the model (not from test data).

USAGE EXAMPLES
--------------
# Single rung (auto-detect variants present)
python -m atus_analysis.scripts.dump_case_metrics \
  --model_type auto \
  --run_dir atus_analysis/data/models/R1 \
  --sequences atus_analysis/data/sequences/markov_sequences.parquet \
  --subgroups atus_analysis/data/processed/subgroups.parquet \
  --time_blocks "night:0-5,morning:6-11,afternoon:12-17,evening:18-23"

# All rungs under models root (only listed rungs)
python -m atus_analysis.scripts.dump_case_metrics \
  --model_type both \
  --run_dir atus_analysis/data/models \
  --rungs R1 R2 R3 R4 R5 R6 \
  --sequences ... --subgroups ... --time_blocks ...

# Minimal (legacy-compatible): run only B1-H for a single rung
python -m atus_analysis.scripts.dump_case_metrics --model_type b1h --run_dir .../R3 --sequences ... --subgroups ... --time_blocks ...
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .common_hier import (
    parse_time_blocks,
    parse_dwell_bins,
    pool_rare_quarter,
    prepare_long_with_groups,
    compute_runlengths_per_respondent,
    load_json,
)

# ─────────────────────────  NUMERIC SAFETY  ─────────────────────────

def _safe_log(p: float) -> float:
    return math.log(max(float(p), 1e-15))

# ─────────────────────  PER-SLOT SIDEcar LOADER  ────────────────────
# Accepts NPZ with keys:
#   - "global": array [144, K, K] (optional)
#   - "<group_key>": arrays [144, K, K] for group-specific routing (optional)
# Returns dict { "__GLOBAL__": arr, "<group_key>": arr } or None if unavailable.

def _load_slot_mats(npz_path: Path, K: int) -> Optional[Dict[str, np.ndarray]]:
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    store: Dict[str, np.ndarray] = {}

    if "global" in data.files:
        glb = data["global"]
        if isinstance(glb, np.ndarray) and glb.ndim == 3 and glb.shape == (144, K, K):
            store["__GLOBAL__"] = glb

    for k in data.files:
        if k == "global":
            continue
        arr = data[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape == (144, K, K):
            store[k] = arr

    return store or None

# ───────────────────────  ROUTING & HAZARD FETCH  ───────────────────

def _get_row_b1(model_json: dict, gk: str, block: str, i: int,
                slot: Optional[int] = None,
                slot_mats: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """
    Returns routing row (length K) for prev-state i using (in order):
      1) per-slot per-group   (slot_mats[gk][slot, i, :])
      2) per-slot global      (slot_mats["__GLOBAL__"][slot, i, :])
      3) block-level per-group (model_json["matrices"][gk::block])
      4) block-level fallback  (model_json["fallback_block"][block])
      5) global template       (model_json["global"])
    """
    if slot_mats is not None and slot is not None:
        if gk in slot_mats:
            return slot_mats[gk][int(slot), int(i), :]
        if "__GLOBAL__" in slot_mats:
            return slot_mats["__GLOBAL__"][int(slot), int(i), :]

    key = f"{gk}::{block}"
    if key in model_json.get("matrices", {}):
        mat = np.array(model_json["matrices"][key], dtype=float)
    elif str(block) in model_json.get("fallback_block", {}):
        mat = np.array(model_json["fallback_block"][str(block)], dtype=float)
    else:
        mat = np.array(model_json["global"], dtype=float)
    return mat[int(i), :]

def _get_hazard_b2(model_json: dict, gk: str, block: str, i: int, d_bin: int) -> float:
    key = f"{gk}::{block}::s{i}::d{d_bin}"
    if key in model_json.get("hazard", {}):
        return float(model_json["hazard"][key])

    key = f"{block}::s{i}::d{d_bin}"
    if key in model_json.get("hazard_block", {}):
        return float(model_json["hazard_block"][key])

    return float(model_json["hazard_global"][f"s{i}::d{d_bin}"])

# ───────────────────────  CORE EVALUATION LOGIC  ────────────────────

def _determine_K(model_type: str, model_json: dict) -> int:
    if model_type == "b1h":
        return int(np.array(model_json["global"], dtype=float).shape[1])
    # b2h stores embedded routing under "routing_b1h"
    routing = model_json["routing_b1h"]
    return int(np.array(routing["global"], dtype=float).shape[1])

def _load_split_for_run(run_dir: Path) -> pd.DataFrame:
    models_root = run_dir.parent
    central = models_root / "fixed_split.parquet"
    local = run_dir / "split_assignments.parquet"
    if central.exists():
        return pd.read_parquet(central)[["TUCASEID", "set"]]
    if local.exists():
        return pd.read_parquet(local)[["TUCASEID", "set"]]
    raise FileNotFoundError(f"Neither {central} nor {local} found")

def _evaluate_one_variant(
    model_type: str,
    run_dir: Path,
    sequences_path: Path,
    subgroups_path: Path,
    groupby: List[str],
    weight_col: str,
    time_blocks_str: str,
    dwell_bins_str: str,
    min_group_weight: float = 0.0,
) -> Path:
    """Compute metrics for a single rung+variant and write parquet. Returns output path."""
    # Load model JSON
    model_path = run_dir / ("b1h_model.json" if model_type == "b1h" else "b2h_model.json")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    model_json = load_json(model_path)

    # For B2-H we need routing template too
    routing_b1 = model_json["routing_b1h"] if model_type == "b2h" else None

    # Determine K from model
    K = _determine_K(model_type, model_json)

    # Optional per-slot routing sidecar (shared across both variants)
    slot_npz = run_dir / "b1h_slot_matz.npz"
    slot_mats = _load_slot_mats(slot_npz, K)

    # Split and config
    split = _load_split_for_run(run_dir)
    blocks = parse_time_blocks(time_blocks_str)
    dwell_edges = parse_dwell_bins(dwell_bins_str)

    # Load data
    long_df = pd.read_parquet(sequences_path)
    groups_df = pd.read_parquet(subgroups_path)

    # Grouping columns (handle ALL group)
    if not groupby:
        groups_df = groups_df.copy()
        groups_df["__ALL__"] = "ALL"
        groupby = ["__ALL__"]

    # Pool rare quarters and attach group keys + blocks
    groups_df = pool_rare_quarter(groups_df, groupby, weight_col, min_group_weight)
    df = prepare_long_with_groups(long_df, groups_df, groupby, weight_col, blocks)

    # Train/test assignment and previous-step context
    df = df.merge(split, on="TUCASEID", how="left")
    test_df = df[df["set"] == "test"].copy().sort_values(["TUCASEID", "slot"])

    prev = test_df.groupby("TUCASEID").shift(1)
    ok = prev["state_id"].notna()
    test_df = test_df.loc[ok].copy()

    test_df["prev_state"] = prev.loc[ok, "state_id"].astype(int)
    test_df["prev_block"] = prev.loc[ok, "block"]
    test_df["prev_group"] = prev.loc[ok, "group_key"]
    test_df["prev_slot"]  = prev.loc[ok, "slot"].astype(int)

    # Dwell bin for hazard
    if model_type == "b2h":
        test_df = compute_runlengths_per_respondent(test_df)
        prev_run = test_df.groupby("TUCASEID")["runlen"].shift(1).fillna(0).astype(int)

        def _to_bin(d: int) -> int:
            if d == 0:
                return len(dwell_edges)  # treat unknown as overflow bin
            return next((k for k, e in enumerate(dwell_edges) if d <= e), len(dwell_edges))

        test_df["d_bin"] = prev_run.apply(_to_bin)

    # Aggregate per respondent
    rows = []
    for case_id, grp in test_df.groupby("TUCASEID", sort=False):
        nll_w = 0.0
        top_w = 0.0
        w_sum = 0.0
        trans = 0

        for _, r in grp.iterrows():
            i = int(r["prev_state"])
            j = int(r["state_id"])
            if not (0 <= i < K and 0 <= j < K):
                raise IndexError(f"State index out of range (i={i}, j={j}, K={K}) for TUCASEID={case_id}")

            gk = r["prev_group"]
            bl = r["prev_block"]
            sl = int(r["prev_slot"])
            w  = float(r[weight_col])

            if model_type == "b1h":
                prow = _get_row_b1(model_json, gk, bl, i, slot=sl, slot_mats=slot_mats)
                p = float(prow[j])
                top = (j == int(np.argmax(prow)))
            else:
                h = _get_hazard_b2(model_json, gk, bl, i, int(r["d_bin"]))
                prow = _get_row_b1(routing_b1, gk, bl, i, slot=sl, slot_mats=slot_mats)

                denom = float(prow.sum() - prow[i])
                if denom <= 0 or not np.isfinite(denom):
                    leave_to = np.full(K, 1.0 / (K - 1), dtype=float)
                    leave_to[i] = 0.0
                else:
                    leave_to = prow.astype(float).copy()
                    leave_to[i] = 0.0
                    leave_to /= denom

                p_next = leave_to * h
                p_next[i] = 1.0 - h

                p = float(p_next[j])
                top = (j == int(np.argmax(p_next)))

            nll_w += -_safe_log(p) * w
            if top:
                top_w += w
            w_sum += w
            trans += 1

        rows.append({
            "TUCASEID": case_id,
            "nll_weighted": nll_w,
            "top1_correct_weight": top_w,
            "weight_total": w_sum,
            "transitions": trans,
            "eval_steps": trans,
        })

    out = pd.DataFrame(rows)

    # Write outputs (variant-suffixed, plus compat unsuffixed if missing)
    suffixed = run_dir / f"test_case_metrics_{model_type}.parquet"
    out.to_parquet(suffixed, index=False)

    compat = run_dir / "test_case_metrics.parquet"
    if not compat.exists():
        out.to_parquet(compat, index=False)

    print(f"✓ [{run_dir.name}:{model_type}] wrote {suffixed}")
    if suffixed != compat and compat.exists():
        print(f"  (compat) {compat} present")
    return suffixed

# ──────────────────────────────  MAIN  ─────────────────────────────

def _discover_rungs(run_dir: Path, rungs_cli: Optional[List[str]]) -> List[Path]:
    """
    If run_dir looks like a rung (contains b1h_model.json/b2h_model.json), return [run_dir].
    Otherwise, treat it as models root and scan subdirs (R*). Filter by rungs_cli if provided.
    """
    run_dir = run_dir.resolve()
    b1 = run_dir / "b1h_model.json"
    b2 = run_dir / "b2h_model.json"
    if b1.exists() or b2.exists():
        return [run_dir]

    # models root: scan R* subdirs
    choices = []
    for sub in sorted(run_dir.iterdir()):
        if sub.is_dir() and sub.name.upper().startswith("R"):
            if rungs_cli and sub.name not in set(rungs_cli):
                continue
            if (sub / "b1h_model.json").exists() or (sub / "b2h_model.json").exists():
                choices.append(sub)
    if not choices:
        raise FileNotFoundError(f"No rung directories with models under: {run_dir}")
    return choices

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["b1h", "b2h", "both", "auto"], required=True,
                    help="'auto' runs whatever variants exist in each rung; 'both' requires both.")
    ap.add_argument("--run_dir", required=True,
                    help="Either a single rung dir (…/models/Rk) or the models root (…/models).")
    ap.add_argument("--rungs", nargs="*", default=None,
                    help="Optional list of rung names (e.g., R1 R2 R3) when run_dir is a models root.")

    ap.add_argument("--sequences", required=True,
                    help="Long-format sequences parquet")
    ap.add_argument("--subgroups", required=True,
                    help="Per-respondent subgroups parquet")

    ap.add_argument("--groupby",
                    default="employment,day_type,hh_size_band,sex,region,quarter",
                    help="Comma-separated subgroup columns; empty string -> single ALL group")
    ap.add_argument("--weight_col", default="TUFNWGTP")
    ap.add_argument("--time_blocks", required=True,
                    help='e.g. "night:0-5,morning:6-11,afternoon:12-17,evening:18-23"')
    ap.add_argument("--dwell_bins", default="1,2,3,4,6,9,14,20,30")
    ap.add_argument("--min_group_weight", type=float, default=0.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    sequences_path = Path(args.sequences).resolve()
    subgroups_path = Path(args.subgroups).resolve()

    # Parse groupby list (strip blanks)
    groupby = [c.strip() for c in args.groupby.split(",") if c.strip()]

    rungs = _discover_rungs(run_dir, args.rungs)

    for rd in rungs:
        has_b1 = (rd / "b1h_model.json").exists()
        has_b2 = (rd / "b2h_model.json").exists()

        if args.model_type == "b1h":
            if not has_b1:
                print(f"[SKIP {rd.name}] no b1h_model.json")
            else:
                _evaluate_one_variant("b1h", rd, sequences_path, subgroups_path,
                                      groupby, args.weight_col, args.time_blocks, args.dwell_bins,
                                      args.min_group_weight)

        elif args.model_type == "b2h":
            if not has_b2:
                print(f"[SKIP {rd.name}] no b2h_model.json")
            else:
                _evaluate_one_variant("b2h", rd, sequences_path, subgroups_path,
                                      groupby, args.weight_col, args.time_blocks, args.dwell_bins,
                                      args.min_group_weight)

        elif args.model_type == "both":
            if not (has_b1 and has_b2):
                print(f"[SKIP {rd.name}] 'both' requested but missing one of the models (b1h={has_b1}, b2h={has_b2})")
                continue
            _evaluate_one_variant("b1h", rd, sequences_path, subgroups_path,
                                  groupby, args.weight_col, args.time_blocks, args.dwell_bins,
                                  args.min_group_weight)
            _evaluate_one_variant("b2h", rd, sequences_path, subgroups_path,
                                  groupby, args.weight_col, args.time_blocks, args.dwell_bins,
                                  args.min_group_weight)

        else:  # auto
            ran_any = False
            if has_b1:
                _evaluate_one_variant("b1h", rd, sequences_path, subgroups_path,
                                      groupby, args.weight_col, args.time_blocks, args.dwell_bins,
                                      args.min_group_weight)
                ran_any = True
            if has_b2:
                _evaluate_one_variant("b2h", rd, sequences_path, subgroups_path,
                                      groupby, args.weight_col, args.time_blocks, args.dwell_bins,
                                      args.min_group_weight)
                ran_any = True
            if not ran_any:
                print(f"[SKIP {rd.name}] no model jsons found")

if __name__ == "__main__":
    main()
