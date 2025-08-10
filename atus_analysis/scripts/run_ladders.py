#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_ladders.py
==============

One-touch orchestrator for all hierarchical ATUS baselines.

Presets
-------
• full      → R1 … R7   (routing ladder + hazard rung + bootstrap R6 vs R7)
• routing   → R1 … R6   (routing ladder only)
• single    → run ONE rung (choose with --rung  R1 … R7; add --also_b2 for hazard)

Defaults assume the standard repo layout:
    data/sequences/markov_sequences.parquet
    data/processed/subgroups.parquet
    data/models/               (each rung is a sub-dir)

Override paths with the --seq, --sub, --out flags if your
files live elsewhere.

Examples
--------
# Full paper experiment, defaults:
python atus_analysis/scripts/run_ladders.py full

# Same but custom split seed
python atus_analysis/scripts/run_ladders.py full --seed 999

# R3 only, routing + hazard:
python atus_analysis/scripts/run_ladders.py single --rung R3 --also_b2
"""

from __future__ import annotations
import argparse, subprocess, json, sys, time, logging, gc
from pathlib import Path
import os

# Set UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ─────────────────────────  rung definitions  ────────────────────────────
SPECS = {
    "R1": "region",
    "R2": "region,sex",
    "R3": "region,employment",
    "R4": "region,day_type",
    "R5": "region,hh_size_band",
    "R6": "employment,day_type,hh_size_band,sex,region,quarter",
    "R7": "employment,day_type,hh_size_band,sex,region,quarter",   # + hazard
}

# ───────────────────────────── helpers ───────────────────────────────────
def call(cmd: list[str]):
    """Run a subprocess, stdout passthrough, abort on non-zero."""
    start_time = time.time()
    logging.info(f"Starting command: {' '.join(cmd)}")
    print(">", " ".join(cmd), flush=True)  # Changed from ▶ to > for Windows compatibility
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        logging.info(f"Command completed successfully in {elapsed:.2f} seconds")
        return result
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logging.error(f"Command failed after {elapsed:.2f} seconds with return code {e.returncode}")
        raise

def run_rung(rung: str, args, split_path: Path, also_b2: bool):
    """Fit routing (B1-H) and optionally hazard (B2-H) for a single rung."""
    logging.info(f"=== Starting rung {rung} ===")
    groupby = SPECS[rung]
    out_dir = args.out / rung
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Rung {rung} configuration:")
    logging.info(f"  - Groupby: {groupby}")
    logging.info(f"  - Output directory: {out_dir}")
    logging.info(f"  - Will run hazard model: {(rung == 'R7') or also_b2}")

    # ---------- routing ----------
    logging.info(f"Starting B1-H (routing) model for {rung}...")
    call([
        sys.executable, "-m", "atus_analysis.scripts.baseline1_hier",
        "--sequences", args.seq, "--subgroups", args.sub, "--out_dir", str(out_dir),
        "--groupby", groupby, "--time_blocks", args.time_blocks,
        "--seed", str(args.seed), "--test_size", str(args.test_size),
        "--split_path", str(split_path),
    ])
    logging.info(f"✓ B1-H model completed for {rung}")

    # ---------- hazard ----------
    if (rung == "R7") or also_b2:
        logging.info(f"Starting B2-H (hazard) model for {rung}...")
        call([
            sys.executable, "-m", "atus_analysis.scripts.baseline2_hier",
            "--sequences", args.seq, "--subgroups", args.sub, "--out_dir", str(out_dir),
            "--groupby", groupby, "--time_blocks", args.time_blocks,
            "--dwell_bins", args.dwell_bins, "--seed", str(args.seed),
            "--test_size", str(args.test_size), "--split_path", str(split_path),
            "--b1h_path", str(out_dir / "b1h_model.json"),
        ])
        logging.info(f"✓ B2-H model completed for {rung}")
    
    logging.info(f"=== Completed rung {rung} ===")
    
    # Memory management
    if args.gc:
        logging.info("Running garbage collection...")
        gc.collect()
    
    if args.delay > 0:
        logging.info(f"Waiting {args.delay} seconds before next operation...")
        time.sleep(args.delay)

def bootstrap_r6_vs_r7(args):
    """Runs respondent-level ΔNLL bootstrap between R6 and R7."""
    logging.info("=== Starting bootstrap comparison between R6 and R7 ===")
    runA = args.out / "R6"
    runB = args.out / "R7"
    
    logging.info("Dumping case metrics for R6...")
    call([
        sys.executable, "-m", "atus_analysis.scripts.dump_case_metrics",
        "--model_type", "b1h", "--run_dir", str(runA),
        "--sequences", args.seq, "--subgroups", args.sub,
        "--groupby", SPECS["R6"], "--time_blocks", args.time_blocks,
    ])
    logging.info("✓ R6 case metrics completed")
    
    logging.info("Dumping case metrics for R7...")
    call([
        sys.executable, "-m", "atus_analysis.scripts.dump_case_metrics",
        "--model_type", "b2h", "--run_dir", str(runB),
        "--sequences", args.seq, "--subgroups", args.sub,
        "--groupby", SPECS["R7"], "--time_blocks", args.time_blocks,
        "--dwell_bins", args.dwell_bins,
    ])
    logging.info("✓ R7 case metrics completed")
    
    logging.info(f"Running bootstrap comparison with {args.B} bootstrap samples...")
    call([
        sys.executable, "-m", "atus_analysis.scripts.run_bootstrap_ci",
        "--run_a", str(runA), "--run_b", str(runB),
        "--B", str(args.B), "--seed", str(args.seed),
    ])
    logging.info("✓ Bootstrap comparison completed")
    logging.info("=== Bootstrap analysis finished ===")

# ─────────────────────────────  main  ────────────────────────────────────
def main():
    start_time = time.time()
    
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--seq",  default="atus_analysis/data/sequences/markov_sequences.parquet",
                      help="long Markov sequences parquet")
    base.add_argument("--sub",  default="atus_analysis/data/processed/subgroups.parquet",
                      help="subgroups parquet")
    base.add_argument("--out",  type=Path, default=Path("atus_analysis/data/models"),
                      help="root directory for model outputs")
    base.add_argument("--time_blocks",
                      default="night:0-5,morning:6-11,afternoon:12-17,evening:18-23")
    base.add_argument("--dwell_bins",
                      default="1,2,3,4,6,9,14,20,30")
    base.add_argument("--seed", type=int, default=2025)
    base.add_argument("--test_size", type=float, default=0.2)
    
    # Memory management options
    base.add_argument("--delay", type=int, default=0,
                      help="Delay in seconds between rungs to reduce system load")
    base.add_argument("--gc", action="store_true",
                      help="Force garbage collection between rungs")

    parent = argparse.ArgumentParser()
    sub = parent.add_subparsers(dest="cmd", required=True)

    # full ladder
    p_full = sub.add_parser("full", parents=[base],
                            help="R1-R7 and bootstrap R6 vs R7")
    p_full.add_argument("--B", type=int, default=1000,
                        help="# bootstrap draws for ΔNLL")

    # routing only
    sub.add_parser("routing", parents=[base],
                   help="R1-R6 routing ladder (no hazard)")

    # single rung
    p_single = sub.add_parser("single", parents=[base],
                              help="run ONE rung")
    p_single.add_argument("--rung", choices=SPECS.keys(), required=True)
    p_single.add_argument("--also_b2", action="store_true",
                          help="add hazard model even if rung≠R7")

    args = parent.parse_args()

    # Log experiment configuration
    logging.info("="*60)
    logging.info("STARTING ATUS HIERARCHICAL BASELINE EXPERIMENTS")
    logging.info("="*60)
    logging.info(f"Command: {args.cmd}")
    logging.info(f"Sequences file: {args.seq}")
    logging.info(f"Subgroups file: {args.sub}")
    logging.info(f"Output directory: {args.out}")
    logging.info(f"Random seed: {args.seed}")
    logging.info(f"Test split size: {args.test_size}")
    logging.info(f"Time blocks: {args.time_blocks}")
    logging.info(f"Dwell bins: {args.dwell_bins}")
    
    # shared split file (under root output)
    args.out.mkdir(parents=True, exist_ok=True)
    split_path = args.out / "fixed_split.parquet"
    logging.info(f"Split file will be saved/loaded from: {split_path}")

    if args.cmd == "routing":
        logging.info("Running ROUTING LADDER (R1-R6, B1-H models only)")
        rungs = ("R1", "R2", "R3", "R4", "R5", "R6")
        for i, rung in enumerate(rungs, 1):
            logging.info(f"Progress: {i}/{len(rungs)} rungs")
            run_rung(rung, args, split_path, also_b2=False)

    elif args.cmd == "full":
        logging.info("Running FULL LADDER (R1-R7 + Bootstrap)")
        if hasattr(args, 'B'):
            logging.info(f"Bootstrap samples: {args.B}")
        rungs = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")
        for i, rung in enumerate(rungs, 1):
            logging.info(f"Progress: {i}/{len(rungs)} rungs")
            run_rung(rung, args, split_path, also_b2=False)
        bootstrap_r6_vs_r7(args)

    elif args.cmd == "single":
        logging.info(f"Running SINGLE RUNG: {args.rung}")
        if args.also_b2:
            logging.info("Will also run B2-H hazard model")
        run_rung(args.rung, args, split_path, also_b2=args.also_b2)

    total_time = time.time() - start_time
    logging.info("="*60)
    logging.info(f"✓ All requested jobs finished in {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    logging.info("="*60)
    print("\n✓ All requested jobs finished.")

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
