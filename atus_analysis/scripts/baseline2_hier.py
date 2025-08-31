#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
baseline2_hier.py — B2-H (routing + dwell-time hazard, both hierarchical)

Writes into <out_dir>:

    b1h_model.json              # routing model (ensured to include slot sidecar/paths)
    b1h_slot_mats.npz           # per-slot matrices sidecar (arrays g0, g1, ...)
    b2h_model.json              # fitted hazard + embedded routing
    eval_b2h.json               # weighted test metrics (B1-H + B2-H)
    split_assignments.parquet   # local split (unless --split_path provided)

If you already have a routing model for the **same split & group-key scheme**,
pass it via --b1h_path. If it lacks per-slot matrices **or** an NPZ sidecar, this
script will refit or re-save so that 144 per-slot matrices are included and saved.
"""
from __future__ import annotations

import argparse
import logging
import time
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
def _make_or_load_split(meta: pd.DataFrame, seed: int, test_size: float, split_path: Path) -> pd.DataFrame:
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
    for _, grp in uniq.groupby("group_key"):
        n_test = int(round(test_size * len(grp)))
        if n_test:
            take = grp.sort_values("rand").head(n_test).index
            uniq.loc[take, "set"] = "test"
            total_test += n_test

    logger.info(f"Split created: {len(uniq) - total_test} train, {total_test} test respondents")
    out = uniq[["TUCASEID", "set"]]
    out.to_parquet(split_path, index=False)
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


def _ensure_b1h_has_slot_sidecar(b1h: dict, out_dir: Path) -> dict:
    """
    Ensures the given B1-H dict references a local NPZ sidecar with per-slot matrices.
    - If inline matrices exist, write them to NPZ and strip from JSON.
    - If an NPZ path exists, load and re-save into out_dir for portability.
    Returns a dictionary ready to be saved/used with updated paths.
    """
    model = dict(b1h)  # shallow copy
    slot_sidecar = out_dir / "b1h_slot_mats.npz"

    # Case A: already has sidecar path AND file exists → copy into out_dir
    npz_path = model.get("slot_matrices_npz") or model.get("meta", {}).get("slot_sidecar")
    if npz_path and Path(npz_path).exists():
        with np.load(npz_path) as npz:
            arrays = {k: np.array(npz[k], dtype=np.float32) for k in npz.files}
        np.savez_compressed(slot_sidecar, **arrays)
        model.setdefault("meta", {})["slot_sidecar"] = str(slot_sidecar)
        model["slot_matrices_npz"] = str(slot_sidecar)
        # strip any inline matrices if present
        if "slot_matrices" in model:
            model.pop("slot_matrices", None)
        return model

    # Case B: has inline 'slot_matrices' → write them to NPZ
    if "slot_matrices" in model:
        slot = model.pop("slot_matrices")
        groups_order = model.get("meta", {}).get("groups_order", list(slot.keys()))
        arrays = {f"g{gi}": np.array(slot[gk], dtype=np.float32) for gi, gk in enumerate(groups_order)}
        np.savez_compressed(slot_sidecar, **arrays)
        model.setdefault("meta", {})["slot_sidecar"] = str(slot_sidecar)
        model["slot_matrices_npz"] = str(slot_sidecar)
        return model

    # Case C: neither sidecar nor inline → return as-is (caller may decide to refit)
    return model


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
    ap.add_argument("--dwell_bins", default=DWELL_BINS_DEFAULT)

    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--split_path", default=None)
    ap.add_argument("--min_group_weight", type=float, default=0.0,
                    help="Pool low-weight region×quarter cells to QALL (0 → off)")

    # routing (Dirichlet) shrinkage
    ap.add_argument("--tau_block_route", type=float, default=50.0)
    ap.add_argument("--tau_group_route", type=float, default=20.0)
    ap.add_argument("--add_k_route",     type=float, default=1.0)
    # Always build 144 per-slot matrices; kappa controls pooling to block prior
    ap.add_argument("--kappa_slot", type=float, default=100.0,
                    help="Dirichlet prior strength for 144 per-slot matrices (prior = block posterior mean)")

    # hazard (Beta) shrinkage
    ap.add_argument("--tau_block_hazard", type=float, default=200.0)
    ap.add_argument("--tau_group_hazard", type=float, default=50.0)
    ap.add_argument("--k0_global",        type=float, default=1.0,
                    help="Tiny symmetric Beta prior at global level")

    # optional: reuse an existing B1-H model
    ap.add_argument("--b1h_path", default=None)

    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("STARTING B2-H (HIERARCHICAL ROUTING + HAZARD) MODEL TRAINING")
    logger.info("=" * 60)
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
    logger.info(f"Shrinkage params - Route: tau_block={args.tau_block_route}, tau_group={args.tau_group_route}, add_k={args.add_k_route}, kappa_slot={args.kappa_slot}")
    logger.info(f"Shrinkage params - Hazard: tau_block={args.tau_block_hazard}, tau_group={args.tau_group_hazard}, k0_global={args.k0_global}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # parse blocks/bins
    logger.info("Parsing time blocks and dwell bins...")
    blocks = parse_time_blocks(args.time_blocks)
    dwell_edges = parse_dwell_bins(args.dwell_bins)
    logger.info(f"Time blocks: {blocks}")
    logger.info(f"Dwell edges: {dwell_edges}")

    # load data
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

    # split (shared across rungs if provided)
    split_path = Path(args.split_path) if args.split_path else out_dir / "split_assignments.parquet"
    logger.info(f"Using split file: {split_path}")
    split = _make_or_load_split(df[["TUCASEID", "group_key"]], args.seed, args.test_size, split_path)
    df = df.merge(split, on="TUCASEID", how="left")

    if df["set"].isna().any():
        missing = df[df["set"].isna()]["TUCASEID"].nunique()
        raise RuntimeError(f"{missing} respondents are missing a train/test assignment in the split file {split_path}")

    train_df = df[df["set"] == "train"].copy()
    test_df  = df[df["set"] == "test"].copy()
    logger.info(f"Data split: {len(train_df):,} train records, {len(test_df):,} test records")
    logger.info(f"Train respondents: {train_df['TUCASEID'].nunique():,}, Test respondents: {test_df['TUCASEID'].nunique():,}")

    # helpful warning if test has unseen groups
    seen_groups = set(train_df["group_key"].unique())
    unseen_in_test = set(test_df["group_key"].unique()) - seen_groups
    if unseen_in_test:
        logger.warning(f"{len(unseen_in_test)} test groups not seen in training; these will back off to parent/global priors")

    # ── B1-H routing model: ensure 144 per-slot matrices + NPZ sidecar ──
    b1h_path_effective = out_dir / "b1h_model.json"  # always save a copy next to B2 outputs
    if args.b1h_path and Path(args.b1h_path).exists():
        logger.info(f"Loading existing B1-H model from: {args.b1h_path}")
        b1h_loaded = load_json(Path(args.b1h_path))

        has_inline = "slot_matrices" in b1h_loaded
        has_npz    = ("slot_matrices_npz" in b1h_loaded) or ("slot_sidecar" in b1h_loaded.get("meta", {}))

        if not (has_inline or has_npz):
            logger.info("Existing B1-H lacks per-slot matrices; refitting routing with per-slot pooling...")
            b1h = fit_b1_hier(
                train_df, n_states, args.weight_col,
                tau_block=args.tau_block_route,
                tau_group=args.tau_group_route,
                add_k=args.add_k_route,
                compute_slot_mats=True,
                kappa_slot=args.kappa_slot,
                time_blocks_spec=args.time_blocks,
            )
        else:
            # Ensure NPZ sidecar saved into out_dir and JSON points to it
            b1h = _ensure_b1h_has_slot_sidecar(b1h_loaded, out_dir)

        # Save/overwrite the colocated copy
        save_json(b1h, b1h_path_effective)
        logger.info(f"B1-H (ensured with per-slot matrices & sidecar) saved to: {b1h_path_effective}")

        # Log sidecar info if present
        npz_path = b1h.get("slot_matrices_npz") or b1h.get("meta", {}).get("slot_sidecar")
        if npz_path and Path(npz_path).exists():
            logger.info(f"Slot matrices sidecar present: {npz_path} (size={Path(npz_path).stat().st_size/1024/1024:.1f} MB)")
        else:
            logger.warning("Warning: no slot sidecar found after ensuring. Per-slot inference may fall back to block/global.")
    else:
        logger.info("Fitting B1-H routing model from scratch (with 144 per-slot matrices)...")
        b1h = fit_b1_hier(
            train_df, n_states, args.weight_col,
            tau_block=args.tau_block_route,
            tau_group=args.tau_group_route,
            add_k=args.add_k_route,
            compute_slot_mats=True,
            kappa_slot=args.kappa_slot,
            time_blocks_spec=args.time_blocks,
        )
        # Save with NPZ sidecar (colocated)
        to_save = dict(b1h)
        slot_sidecar = out_dir / "b1h_slot_mats.npz"
        if "slot_matrices" in to_save:
            slot = to_save.pop("slot_matrices")
            groups_order = to_save.get("meta", {}).get("groups_order", list(slot.keys()))
            arrays = {f"g{gi}": np.array(slot[gk], dtype=np.float32) for gi, gk in enumerate(groups_order)}
            np.savez_compressed(slot_sidecar, **arrays)
            to_save.setdefault("meta", {})["slot_sidecar"] = str(slot_sidecar)
            to_save["slot_matrices_npz"] = str(slot_sidecar)
        save_json(to_save, b1h_path_effective)
        b1h = to_save
        logger.info(f"B1-H model saved to: {b1h_path_effective}")
        if slot_sidecar.exists():
            logger.info(f"Slot matrices sidecar saved to: {slot_sidecar} (size={slot_sidecar.stat().st_size/1024/1024:.1f} MB)")

    # ── B2-H hazard model ──
    logger.info("Fitting B2-H hazard model...")
    b2h = fit_b2_hier(
        train_df, n_states, args.weight_col, dwell_edges,
        tau_block=args.tau_block_hazard,
        tau_group=args.tau_group_hazard,
        k0=args.k0_global,
        routing_b1h=b1h,
    )
    b2h_path = out_dir / "b2h_model.json"
    save_json(b2h, b2h_path)
    logger.info(f"B2-H hazard model saved to: {b2h_path}")

    # ── Evaluation ──
    logger.info("Evaluating models on test set...")
    # Use the persisted b1h dict (with sidecar references) for evaluation
    m_b1 = nll_b1(test_df, b1h, n_states, args.weight_col)
    b1_val = _metric_per_weight(m_b1)
    if isinstance(b1_val, (int, float, np.floating)):
        logger.info(f"B1-H test NLL (per weight): {b1_val:.4f}")
    else:
        logger.info(f"B1-H test NLL (per weight): {b1_val}")

    m_b2 = nll_b2(test_df, b2h, dwell_edges, n_states, args.weight_col)
    b2_val = _metric_per_weight(m_b2)
    if isinstance(b2_val, (int, float, np.floating)):
        logger.info(f"B2-H test NLL (per weight): {b2_val:.4f}")
    else:
        logger.info(f"B2-H test NLL (per weight): {b2_val}")

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
                "kappa_slot": args.kappa_slot,
            },
            "hazard": {
                "tau_block": args.tau_block_hazard,
                "tau_group": args.tau_group_hazard,
                "k0_global": args.k0_global,
            },
            "seed": args.seed,
            "split_path": str(split_path),
            "b1h_model_path": str(b1h_path_effective),
        },
    }
    eval_path = out_dir / "eval_b2h.json"
    save_json(eval_data, eval_path)
    logger.info(f"Evaluation results saved to: {eval_path}")

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f" B2-H training completed in {total_time:.2f} seconds")
    logger.info(f" B2-H written to: {out_dir}")
    logger.info("=" * 60)
    print(" B2-H run complete →", out_dir)


if __name__ == "__main__":
    main()
