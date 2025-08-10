"""

Create <run_dir>/test_case_metrics.parquet with per-respondent
   • weighted negative log-likelihood
   • weighted top-1 correctness
   • total design-weight and #transitions

Supports both B1-H (routing only) and B2-H (routing + hazard).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import math
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

# ─────────────────────  PRIVATE SHORTCUTS  ──────────────────────
def _safe_log(p: float) -> float:
    return math.log(max(float(p), 1e-15))

def _get_row_b1(model, gk: str, block: str, i: int) -> np.ndarray:
    key = f"{gk}::{block}"
    if key in model["matrices"]:
        mat = np.array(model["matrices"][key], dtype=float)
    elif str(block) in model["fallback_block"]:
        mat = np.array(model["fallback_block"][str(block)], dtype=float)
    else:
        mat = np.array(model["global"], dtype=float)
    return mat[i, :]

def _get_hazard_b2(model, gk: str, block: str, i: int, d: int) -> float:
    key = f"{gk}::{block}::s{i}::d{d}"
    if key in model["hazard"]:
        return model["hazard"][key]
    key = f"{block}::s{i}::d{d}"
    if key in model["hazard_block"]:
        return model["hazard_block"][key]
    return model["hazard_global"][f"s{i}::d{d}"]

# ───────────────────────────  MAIN  ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["b1h", "b2h"], required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--subgroups", required=True)

    ap.add_argument("--groupby",
                    default="employment,day_type,hh_size_band,sex,region,quarter")
    ap.add_argument("--weight_col", default="TUFNWGTP")
    ap.add_argument("--time_blocks", required=True)
    ap.add_argument("--dwell_bins",  default="1,2,3,4,6,9,14,20,30")
    ap.add_argument("--min_group_weight", type=float, default=0.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    model_file = run_dir / ("b1h_model.json" if args.model_type == "b1h"
                            else "b2h_model.json")
    model = load_json(model_file)
    if args.model_type == "b2h":
        routing_b1 = model["routing_b1h"]

    split = pd.read_parquet(run_dir / "split_assignments.parquet")[["TUCASEID", "set"]]

    # -------------- data with group_key + block -----------------
    blocks = parse_time_blocks(args.time_blocks)
    dwell_edges = parse_dwell_bins(args.dwell_bins)

    long_df   = pd.read_parquet(args.sequences)
    groups_df = pd.read_parquet(args.subgroups)

    cols = [c.strip() for c in args.groupby.split(",") if c.strip()]
    if not cols:
        groups_df = groups_df.copy(); groups_df["__ALL__"] = "ALL"; cols = ["__ALL__"]

    groups_df = pool_rare_quarter(groups_df, cols, args.weight_col, args.min_group_weight)
    df = prepare_long_with_groups(long_df, groups_df, cols, args.weight_col, blocks)
    df = df.merge(split, on="TUCASEID", how="left")

    test_df = df[df["set"] == "test"].copy()
    n_states = int(test_df["state_id"].max()) + 1

    # -------------- previous-state rows -------------------------
    test_df = test_df.sort_values(["TUCASEID", "slot"])
    prev = test_df.groupby("TUCASEID").shift(1)
    ok = prev["state_id"].notna()
    test_df = test_df.loc[ok].copy()

    test_df["prev_state"] = prev.loc[ok, "state_id"].astype(int)
    test_df["prev_block"] = prev.loc[ok, "block"]
    test_df["prev_group"] = prev.loc[ok, "group_key"]

    # for hazard we also need dwell bin at t-1
    if args.model_type == "b2h":
        test_df = compute_runlengths_per_respondent(test_df)
        prev_run = test_df.groupby("TUCASEID").shift(1)["runlen"].astype(int)
        test_df["d_bin"] = prev_run.loc[ok].apply(
            lambda d: len(dwell_edges) if pd.isna(d) else
            next((k for k, e in enumerate(dwell_edges) if d <= e),
                 len(dwell_edges))
        )

    # -------------- aggregate per respondent -------------------
    rows = []
    for case_id, grp in test_df.groupby("TUCASEID", sort=False):
        nll_w = top_w = w_sum = 0.0
        trans = 0
        for _, r in grp.iterrows():
            i = int(r["prev_state"])
            j = int(r["state_id"])
            gk = r["prev_group"]
            bl = r["prev_block"]
            w  = float(r[args.weight_col])

            if args.model_type == "b1h":
                prow = _get_row_b1(model, gk, bl, i)
                p = float(prow[j])
                top = j == int(prow.argmax())
            else:
                d_bin = int(r["d_bin"])
                h = _get_hazard_b2(model, gk, bl, i, d_bin)
                prow = _get_row_b1(routing_b1, gk, bl, i)

                denom = float(prow.sum() - prow[i])
                if denom <= 0:
                    leave_to = np.full(n_states, 1/(n_states-1)); leave_to[i] = 0.0
                else:
                    leave_to = prow.copy(); leave_to[i] = 0.0; leave_to /= denom

                p_next = leave_to * h
                p_next[i] = 1.0 - h

                p = float(p_next[j])
                top = j == int(p_next.argmax())

            nll_w += -_safe_log(p) * w
            top_w += w if top else 0.0
            w_sum += w
            trans += 1

        rows.append({
            "TUCASEID": case_id,
            "nll_weighted": nll_w,
            "top1_correct_weight": top_w,
            "weight_total": w_sum,
            "transitions": trans,
        })

    out = pd.DataFrame(rows)
    out.to_parquet(run_dir / "test_case_metrics.parquet", index=False)
    print("✓ wrote", run_dir / "test_case_metrics.parquet")

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
