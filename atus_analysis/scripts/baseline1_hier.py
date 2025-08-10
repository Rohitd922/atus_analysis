
"""
Dirichlet–multinomial hierarchical routing (B1-H).
Writes:
    b1h_model.json
    eval_b1h.json
    split_assignments.parquet   (first time only)
"""
import argparse, logging, time
from pathlib import Path
import numpy as np
import pandas as pd
from .common_hier import (
    TIME_BLOCKS_DEFAULT, parse_time_blocks,
    prepare_long_with_groups, pool_rare_quarter,
    save_json, nll_b1, fit_b1_hier,
)

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def _make_or_load_split(meta, seed, pct_test, where: Path):
    """Create or load train/test split, ensuring consistent splits across runs."""
    if where.exists():
        logger.info(f"Loading existing split from {where}")
        return pd.read_parquet(where)[["TUCASEID", "set"]]
    
    logger.info(f"Creating new split with {pct_test:.1%} test size, seed={seed}")
    rng = np.random.RandomState(seed)
    uniq = meta.drop_duplicates().copy()
    uniq["rand"] = rng.rand(len(uniq))
    uniq["set"]  = "train"
    
    # Split by group to ensure balanced representation
    total_test = 0
    for gk, grp in uniq.groupby("group_key"):
        take = int(round(pct_test * len(grp)))
        if take:
            idx = grp.sort_values("rand").head(take).index
            uniq.loc[idx, "set"] = "test"
            total_test += take
    
    logger.info(f"Split created: {len(uniq) - total_test} train, {total_test} test respondents")
    uniq[["TUCASEID", "set"]].to_parquet(where, index=False)
    return uniq[["TUCASEID", "set"]]

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
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--min_group_weight", type=float, default=0.0)
# shared respondent split file (optional; set by run_ladders.py)
    ap.add_argument("--split_path", default=None)
    # shrinkage knobs
    ap.add_argument("--tau_block", type=float, default=50.0)
    ap.add_argument("--tau_group", type=float, default=20.0)
    ap.add_argument("--add_k",   type=float, default=1.0)
    args = ap.parse_args()

    logger.info("="*60)
    logger.info("STARTING B1-H (HIERARCHICAL ROUTING) MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Sequences file: {args.sequences}")
    logger.info(f"Subgroups file: {args.subgroups}")
    logger.info(f"Output directory: {args.out_dir}")
    logger.info(f"Groupby dimensions: {args.groupby}")
    logger.info(f"Weight column: {args.weight_col}")
    logger.info(f"Time blocks: {args.time_blocks}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Shrinkage - tau_block: {args.tau_block}, tau_group: {args.tau_group}, add_k: {args.add_k}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    blocks = parse_time_blocks(args.time_blocks)
    logger.info(f"Parsed time blocks: {blocks}")

    # Load data
    logger.info("Loading sequences data...")
    long = pd.read_parquet(args.sequences)
    logger.info(f"Loaded {len(long):,} sequence records")
    
    logger.info("Loading subgroups data...")
    groups = pd.read_parquet(args.subgroups)
    logger.info(f"Loaded {len(groups):,} respondent records")

    cols = [c.strip() for c in args.groupby.split(",") if c.strip()]
    if not cols:
        groups = groups.copy(); groups["__ALL__"] = "ALL"; cols = ["__ALL__"]
        logger.info("No groupby columns specified, using single global group")
    else:
        logger.info(f"Grouping by: {cols}")

    logger.info("Pooling rare quarter groups...")
    groups = pool_rare_quarter(groups, cols, args.weight_col, args.min_group_weight)
    
    logger.info("Preparing data with groups and time blocks...")
    df = prepare_long_with_groups(long, groups, cols, args.weight_col, blocks)
    n_states = int(df["state_id"].max()) + 1
    logger.info(f"Data prepared: {len(df):,} records, {n_states} states, {len(df['group_key'].unique())} groups")

    # Create or load split
    split_file = Path(args.split_path) if args.split_path else out / "split_assignments.parquet"
    split = _make_or_load_split(df[["TUCASEID", "group_key"]], args.seed,
                                args.test_size, split_file)
    df = df.merge(split, on="TUCASEID", how="left")
    train, test = df[df.set == "train"], df[df.set == "test"]
    
    logger.info(f"Data split: {len(train):,} train records, {len(test):,} test records")
    logger.info(f"Train respondents: {train['TUCASEID'].nunique():,}, Test respondents: {test['TUCASEID'].nunique():,}")

    # Fit model
    logger.info("Fitting B1-H hierarchical model...")
    model = fit_b1_hier(train, n_states, args.weight_col,
                        tau_block=args.tau_block,
                        tau_group=args.tau_group,
                        add_k=args.add_k)
    logger.info("✓ Model fitting completed")
    
    model_path = out / "b1h_model.json"
    save_json(model, model_path)
    logger.info(f"Model saved to: {model_path}")

    # Evaluate model
    logger.info("Evaluating model on test set...")
    metrics = nll_b1(test, model, n_states, args.weight_col)
    logger.info(f"Test NLL: {metrics.get('nll', 'N/A'):.4f}")
    
    eval_data = {"b1h": metrics,
               "notes": {"groupby": cols,
                         "time_blocks": args.time_blocks,
                         "tau_block": args.tau_block,
                         "tau_group": args.tau_group,
                         "add_k": args.add_k,
                         "seed": args.seed}}
    eval_path = out / "eval_b1h.json"
    save_json(eval_data, eval_path)
    logger.info(f"Evaluation results saved to: {eval_path}")
    
    total_time = time.time() - start_time
    logger.info("="*60)
    logger.info(f"✓ B1-H training completed in {total_time:.2f} seconds")
    logger.info(f"✓ B1-H written to: {out}")
    logger.info("="*60)
    print("✓ B1-H written to:", out)

if __name__ == "__main__":
    main()
