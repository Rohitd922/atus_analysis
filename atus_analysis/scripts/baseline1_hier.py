#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
baseline1_hier.py — Dirichlet–multinomial hierarchical routing (B1-H).

Writes into <out_dir>:

    b1h_model.json              # routing model with metadata
    b1h_slot_mats.npz           # per-slot matrices sidecar (arrays g0, g1, ...)
    eval_b1h.json               # weighted test metrics
    split_assignments.parquet   # local split (unless --split_path provided)
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common_hier import (
    TIME_BLOCKS_DEFAULT,
    parse_time_blocks,
    prepare_long_with_groups,
    pool_rare_quarter,
    save_json,
    nll_b1,
    fit_b1_hier,
)

# ───────────────────────── logger ─────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ───────────────────────── helpers ───────────────────────
def _make_or_load_split(meta: pd.DataFrame, seed: int, test_size: float, where: Path) -> pd.DataFrame:
    """
    Create or load a respondent-level train/test split stratified by group_key.
    Expects `meta` to have columns: ["TUCASEID", "group_key"].
    """
    if where.exists():
        logger.info(f"Loading existing split from {where}")
        return pd.read_parquet(where)[["TUCASEID", "set"]]

    logger.info(f"Creating new split with {test_size:.1%} test size, seed={seed}")
    rng = np.random.RandomState(seed)
    uniq = meta.drop_duplicates().copy()
    uniq["rand"] = rng.rand(len(uniq))
    uniq["set"] = "train"

    total_test = 0
    for _, grp in uniq.groupby("group_key"):
        n_test = int(round(test_size * len(grp)))
        if n_test:
            take = grp.sort_values("rand").head(n_test).index
            uniq.loc[take, "set"] = "test"
            total_test += n_test

    logger.info(f"Split created: {len(uniq) - total_test} train, {total_test} test respondents")
    out = uniq[["TUCASEID", "set"]]
    out.to_parquet(where, index=False)
    return out


def _metric_per_weight(metrics: dict):
    """
    Return a numeric NLL per weight from a metrics dict, if present.
    Falls back to nll_weighted / weight_total when needed.
    """
    if "nll_per_weight" in metrics:
        return metrics["nll_per_weight"]
    if "nll" in metrics:  # legacy key
        return metrics["nll"]
    if "nll_weighted" in metrics and "weight_total" in metrics and metrics["weight_total"]:
        return metrics["nll_weighted"] / metrics["weight_total"]
    return "N/A"


# ───────────────────────── main ──────────────────────────
def main():
    start_time = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--subgroups", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--groupby", default="employment,day_type,hh_size_band,sex,region,quarter")
    ap.add_argument("--weight_col", default="TUFNWGTP")
    ap.add_argument("--time_blocks", default=TIME_BLOCKS_DEFAULT)

    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2025)

    ap.add_argument("--min_group_weight", type=float, default=0.0,
                    help="Pool low-weight region×quarter cells to QALL (0 → off)")
    ap.add_argument("--split_path", default=None,
                    help="Optional shared split. If absent, a local split file is created in out_dir.")

    # shrinkage knobs (routing)
    ap.add_argument("--tau_block", type=float, default=50.0)
    ap.add_argument("--tau_group", type=float, default=20.0)
    ap.add_argument("--add_k",     type=float, default=1.0)

    # Always build 144 per-slot matrices; kappa controls pooling to block prior
    ap.add_argument("--kappa_slot", type=float, default=100.0,
                    help="Dirichlet prior strength for 144 per-slot matrices (prior = block posterior mean)")

    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("STARTING B1-H (HIERARCHICAL ROUTING) MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Sequences file: {args.sequences}")
    logger.info(f"Subgroups file: {args.subgroups}")
    logger.info(f"Output directory: {args.out_dir}")
    logger.info(f"Groupby dimensions: {args.groupby}")
    logger.info(f"Weight column: {args.weight_col}")
    logger.info(f"Time blocks: {args.time_blocks}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Shrinkage - tau_block: {args.tau_block}, tau_group: {args.tau_group}, add_k: {args.add_k}")
    logger.info(f"Per-slot pooling - kappa_slot: {args.kappa_slot}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = parse_time_blocks(args.time_blocks)
    logger.info(f"Parsed time blocks: {blocks}")

    # Load data
    logger.info("Loading sequences data...")
    long_df = pd.read_parquet(args.sequences)
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
    n_groups = df["group_key"].nunique()
    logger.info(f"Data prepared: {len(df):,} records, {n_states} states, {n_groups} groups")

    # Split
    split_file = Path(args.split_path) if args.split_path else out_dir / "split_assignments.parquet"
    split = _make_or_load_split(df[["TUCASEID", "group_key"]], args.seed, args.test_size, split_file)
    df = df.merge(split, on="TUCASEID", how="left")

    if df["set"].isna().any():
        missing = df[df["set"].isna()]["TUCASEID"].nunique()
        raise RuntimeError(f"{missing} respondents are missing a train/test assignment in {split_file}")

    train_df = df[df["set"] == "train"].copy()
    test_df  = df[df["set"] == "test"].copy()

    logger.info(f"Data split: {len(train_df):,} train records, {len(test_df):,} test records")
    logger.info(f"Train respondents: {train_df['TUCASEID'].nunique():,}, Test respondents: {test_df['TUCASEID'].nunique():,}")

    # Helpful warning if test has unseen groups
    seen_groups = set(train_df["group_key"].unique())
    unseen_in_test = set(test_df["group_key"].unique()) - seen_groups
    if unseen_in_test:
        logger.warning(f"{len(unseen_in_test)} test groups not seen in training; will back off to parent/global priors")

    # Fit model — ALWAYS compute 144 per-slot matrices
    logger.info("Fitting B1-H hierarchical model (with 144 per-slot matrices)...")
    b1h = fit_b1_hier(
        train_df, n_states, args.weight_col,
        tau_block=args.tau_block,
        tau_group=args.tau_group,
        add_k=args.add_k,
        compute_slot_mats=True,
        kappa_slot=args.kappa_slot,
        time_blocks_spec=args.time_blocks,
    )
    logger.info("✓ Model fitting completed")

    # Save with NPZ sidecar for slot matrices
    model_path = out_dir / "b1h_model.json"
    slot_sidecar = out_dir / "b1h_slot_mats.npz"
    to_save = dict(b1h)
    if "slot_matrices" in to_save:
        slot = to_save.pop("slot_matrices")
        groups_order = to_save.get("meta", {}).get("groups_order", list(slot.keys()))
        arrays = {}
        for gi, gk in enumerate(groups_order):
            arrays[f"g{gi}"] = np.array(slot[gk], dtype=np.float32)
        np.savez_compressed(slot_sidecar, **arrays)
        to_save.setdefault("meta", {})["slot_sidecar"] = str(slot_sidecar)
        to_save["slot_matrices_npz"] = str(slot_sidecar)

    save_json(to_save, model_path)
    logger.info(f"Model saved to: {model_path}")
    if slot_sidecar.exists():
        logger.info(f"Slot matrices sidecar saved to: {slot_sidecar} (size={slot_sidecar.stat().st_size/1024/1024:.1f} MB)")

    # Evaluate model
    logger.info("Evaluating model on test set...")
    m = nll_b1(test_df, to_save, n_states, args.weight_col)  # use saved dict (with sidecar paths)
    nll_val = _metric_per_weight(m)
    if isinstance(nll_val, (int, float, np.floating)):
        logger.info(f"B1-H test NLL (per weight): {nll_val:.4f}")
    else:
        logger.info(f"B1-H test NLL (per weight): {nll_val}")

    eval_data = {
        "b1h": m,
        "notes": {
            "groupby": cols,
            "time_blocks": args.time_blocks,
            "tau_block": args.tau_block,
            "tau_group": args.tau_group,
            "add_k": args.add_k,
            "kappa_slot": args.kappa_slot,
            "seed": args.seed,
            "split_path": str(split_file),
        },
    }
    eval_path = out_dir / "eval_b1h.json"
    save_json(eval_data, eval_path)
    logger.info(f"Evaluation results saved to: {eval_path}")

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f" B1-H training completed in {total_time:.2f} seconds")
    logger.info(f" Written to: {out_dir}")
    logger.info("=" * 60)
    print(" B1-H run complete →", out_dir)


if __name__ == "__main__":
    main()
