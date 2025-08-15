#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
baseline2_hier.py   –   B2-H  (routing + dwell-time hazard, both hierarchical)

Writes into <out_dir>:

    b2h_model.json        # fitted hazard + embedded routing
    eval_b2h.json         # weighted test metrics  (B1-H + B2-H)
    split_assignments.parquet   (local copy, defaults to using central fixed_split.parquet if provided)

If you already have a routing model for the **same split & group-key scheme**,
pass it via --b1h_path to avoid refitting.

Typical call (single rung) – see run_ablation_ladder.py for batch use:

python atus_analysis/scripts/baseline2_hier.py \
    --sequences  atus_analysis/data/sequences/markov_sequences.parquet \
    --subgroups  atus_analysis/data/processed/subgroups.parquet \
    --out_dir    atus_analysis/data/models/R7_full_dwell \
    --groupby    employment,day_type,hh_size_band,sex,region,quarter \
    --time_blocks "night:0-5,morning:6-11,afternoon:12-17,evening:18-23" \
    --dwell_bins "1,2,3,4,6,9,14,20,30" \
    --seed 2025 --test_size 0.2
"""

from __future__ import annotations

import argparse, logging, time
from pathlib import Path

import numpy as np
import pandas as pd

from .common_hier import (
    TIME_BLOCKS_DEFAULT,
    DWELL_BINS_DEFAULT,
    parse_time_blocks,
    parse_dwell_bins,
    pool_rare_quarter,
    prepare_long_with_groups,
    save_json,
    load_json,
    fit_b1_hier,
    fit_b2_hier,
    nll_b1,
    nll_b2,
)

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ─────────────────────────  helper  (shared split)  ──────────────────────────
def _make_or_load_split(
    meta: pd.DataFrame,
    seed: int,
    test_size: float,
    split_path: Path,
) -> pd.DataFrame:
    """
    Respondent-level train/test split stratified by group_key.
    Re-uses the same split if the file already exists.
    """
    if split_path.exists():
        logger.info(f"Loading existing split from {split_path}")
        return pd.read_parquet(split_path)[["TUCASEID", "set"]]

    logger.info(f"Creating new split with {test_size:.1%} test size, seed={seed}")
    rng = np.random.RandomState(seed)
    uniq = meta.drop_duplicates().copy()
    uniq["rand"] = rng.rand(len(uniq))
    uniq["set"] = "train"

    total_test = 0
    for gk, grp in uniq.groupby("group_key"):
        n = len(grp)
        n_test = int(round(test_size * n))
        if n_test:
            take = grp.sort_values("rand").head(n_test).index
            uniq.loc[take, "set"] = "test"
            total_test += n_test

    logger.info(f"Split created: {len(uniq) - total_test} train, {total_test} test respondents")
    uniq[["TUCASEID", "set"]].to_parquet(split_path, index=False)
    return uniq[["TUCASEID", "set"]]

# ─────────────────────────────────  main  ────────────────────────────────────
def main():
    start_time = time.time()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--subgroups", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--groupby",
                    default="employment,day_type,hh_size_band,sex,region,quarter")
    ap.add_argument("--weight_col", default="TUFNWGTP")

    ap.add_argument("--time_blocks", default=TIME_BLOCKS_DEFAULT)
    ap.add_argument("--dwell_bins",  default=DWELL_BINS_DEFAULT)

    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--split_path", default=None)
    ap.add_argument("--min_group_weight", type=float, default=0.0,
                    help="Pool low-weight region×quarter cells to QALL (0 → off)")

    # ── routing (Dirichlet) shrinkage
    ap.add_argument("--tau_block_route", type=float, default=50.0)
    ap.add_argument("--tau_group_route", type=float, default=20.0)
    ap.add_argument("--add_k_route",   type=float, default=1.0)

    # ── hazard (Beta) shrinkage
    ap.add_argument("--tau_block_hazard", type=float, default=200.0)
    ap.add_argument("--tau_group_hazard", type=float, default=50.0)
    ap.add_argument("--k0_global",        type=float, default=1.0,
                    help="Tiny symmetric Beta prior at global level")

    # optional: reuse an existing B1-H model
    ap.add_argument("--b1h_path", default=None)

    args = ap.parse_args()
    
    logger.info("="*60)
    logger.info("STARTING B2-H (HIERARCHICAL ROUTING + HAZARD) MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Sequences file: {args.sequences}")
    logger.info(f"Subgroups file: {args.subgroups}")
    logger.info(f"Output directory: {args.out_dir}")
    logger.info(f"Groupby dimensions: {args.groupby}")
    logger.info(f"Weight column: {args.weight_col}")
    logger.info(f"Time blocks: {args.time_blocks}")
    logger.info(f"Dwell bins: {args.dwell_bins}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Existing B1-H model: {args.b1h_path or 'None (will fit routing from scratch)'}")
    logger.info(f"Shrinkage params - Route: tau_block={args.tau_block_route}, tau_group={args.tau_group_route}, add_k={args.add_k_route}")
    logger.info(f"Shrinkage params - Hazard: tau_block={args.tau_block_hazard}, tau_group={args.tau_group_hazard}, k0_global={args.k0_global}")
    
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ───────────────────  load data & build group_key  ──────────────────────
    logger.info("Parsing time blocks and dwell bins...")
    blocks = parse_time_blocks(args.time_blocks)
    dwell_edges = parse_dwell_bins(args.dwell_bins)
    logger.info(f"Time blocks: {blocks}")
    logger.info(f"Dwell edges: {dwell_edges}")

    logger.info("Loading sequences data...")
    long_df   = pd.read_parquet(args.sequences)
    logger.info(f"Loaded {len(long_df):,} sequence records")
    
    logger.info("Loading subgroups data...")
    groups_df = pd.read_parquet(args.subgroups)
    logger.info(f"Loaded {len(groups_df):,} respondent records")

    cols = [c.strip() for c in args.groupby.split(",") if c.strip()]
    if not cols:
        groups_df = groups_df.copy()
        groups_df["__ALL__"] = "ALL"
        cols = ["__ALL__"]
        logger.info("No groupby columns specified, using single global group")
    else:
        logger.info(f"Grouping by: {cols}")

    logger.info("Pooling rare quarter groups...")
    groups_df = pool_rare_quarter(groups_df, cols, args.weight_col, args.min_group_weight)
    
    logger.info("Preparing data with groups and time blocks...")
    df = prepare_long_with_groups(long_df, groups_df, cols, args.weight_col, blocks)
    n_states = int(df["state_id"].max()) + 1
    logger.info(f"Data prepared: {len(df):,} records, {n_states} states, {len(df['group_key'].unique())} groups")

    # ───────────────────  split  (shared across rungs)  ─────────────────────
    split_path = Path(args.split_path) if args.split_path else out_dir / "split_assignments.parquet"
    logger.info(f"Using split file: {split_path}")
    split = _make_or_load_split(df[["TUCASEID", "group_key"]],
                                args.seed, args.test_size, split_path)
    df = df.merge(split, on="TUCASEID", how="left")
    train_df = df[df["set"] == "train"].copy()
    test_df  = df[df["set"] == "test"].copy()
    logger.info(f"Data split: {len(train_df):,} train records, {len(test_df):,} test records")
    logger.info(f"Train respondents: {train_df['TUCASEID'].nunique():,}, Test respondents: {test_df['TUCASEID'].nunique():,}")

    # ───────────────────  routing model (B1-H)  ────────────────────────────
    if args.b1h_path and Path(args.b1h_path).exists():
        logger.info(f"Loading existing B1-H model from: {args.b1h_path}")
        b1h = load_json(Path(args.b1h_path))
        logger.info("✓ B1-H model loaded successfully")
    else:
        logger.info("Fitting B1-H routing model from scratch...")
        b1h = fit_b1_hier(
            train_df, n_states, args.weight_col,
            tau_block=args.tau_block_route,
            tau_group=args.tau_group_route,
            add_k=args.add_k_route,
        )
        logger.info("✓ B1-H routing model fitted successfully")
        b1h_path = out_dir / "b1h_model.json"
        save_json(b1h, b1h_path)
        logger.info(f"B1-H model saved to: {b1h_path}")

    # ───────────────────  hazard model (B2-H)  ─────────────────────────────
    logger.info("Fitting B2-H hazard model...")
    b2h = fit_b2_hier(
        train_df, n_states, args.weight_col, dwell_edges,
        tau_block=args.tau_block_hazard,
        tau_group=args.tau_group_hazard,
        k0=args.k0_global,
        routing_b1h=b1h,
    )
    logger.info("✓ B2-H hazard model fitted successfully")
    b2h_path = out_dir / "b2h_model.json"
    save_json(b2h, b2h_path)
    logger.info(f"B2-H model saved to: {b2h_path}")

    # ───────────────────  evaluation  ──────────────────────────────────────
    logger.info("Evaluating models on test set...")
    m_b1 = nll_b1(test_df, b1h, n_states, args.weight_col)
    nll_b1_val = m_b1.get('nll', 'N/A')
    logger.info(f"B1-H test NLL: {nll_b1_val:.4f}" if isinstance(nll_b1_val, (int, float)) else f"B1-H test NLL: {nll_b1_val}")
    
    m_b2 = nll_b2(test_df, b2h, dwell_edges, n_states, args.weight_col)
    nll_b2_val = m_b2.get('nll', 'N/A')
    logger.info(f"B2-H test NLL: {nll_b2_val:.4f}" if isinstance(nll_b2_val, (int, float)) else f"B2-H test NLL: {nll_b2_val}")

    eval_data = {
        "b1h": m_b1,
        "b2h": m_b2,
        "notes": {
            "groupby": cols,
            "time_blocks": args.time_blocks,
            "dwell_bins": args.dwell_bins,
            "routing": {
                "tau_block": args.tau_block_route,
                "tau_group": args.tau_group_route,
                "add_k": args.add_k_route,
            },
            "hazard": {
                "tau_block": args.tau_block_hazard,
                "tau_group": args.tau_group_hazard,
                "k0_global": args.k0_global,
            },
            "seed": args.seed,
        },
    }
    eval_path = out_dir / "eval_b2h.json"
    save_json(eval_data, eval_path)
    logger.info(f"Evaluation results saved to: {eval_path}")
    
    total_time = time.time() - start_time
    logger.info("="*60)
    logger.info(f"✓ B2-H training completed in {total_time:.2f} seconds")
    logger.info(f"✓ B2-H written to: {out_dir}")
    logger.info("="*60)
    print("✓ B2-H run complete →", out_dir)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
