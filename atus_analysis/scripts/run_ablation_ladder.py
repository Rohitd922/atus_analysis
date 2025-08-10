#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the 7-rung ablation ladder on a **single respondent split**:

R1 – R6  → routing-only (B1-H) with gradually richer subgroup sets
R7        → adds dwell-time hazard (B2-H) on top of R6’s routing

Optional: `--also_b2` builds a B2-H twin for EVERY rung (handy for
stress-tests, but slow).

Usage (from repo root)
----------------------
python atus_analysis/scripts/run_ablation_ladder.py \
    --seq  atus_analysis/data/sequences/markov_sequences.parquet \
    --sub  atus_analysis/data/processed/subgroups.parquet \
    --time_blocks "night:0-5,morning:6-11,afternoon:12-17,evening:18-23" \
    --out  atus_analysis/data/models \
    --seed 2025 \
    --also_b2      # ← optional
"""
from __future__ import annotations
import subprocess, argparse, json, time, logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ───────────────────────── rung specs ─────────────────────────
SPECS = {
    # name     groupby dimensions (comma-separated string)
    "R1_region"         : "region",
    "R2_region_sex"     : "region,sex",
    "R3_region_employ"  : "region,employment",
    "R4_region_daytype" : "region,day_type",
    "R5_region_hhsize"  : "region,hh_size_band",
    "R6_full"           : "employment,day_type,hh_size_band,sex,region,quarter",
    "R7_full_dwell"     : "employment,day_type,hh_size_band,sex,region,quarter",  # same as R6
}

def call(cmd: list[str]):
    """Run a subprocess with detailed logging."""
    start_time = time.time()
    logging.info(f"Starting command: {' '.join(cmd)}")
    print("▶", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        logging.info(f"Command completed successfully in {elapsed:.2f} seconds")
        return result
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logging.error(f"Command failed after {elapsed:.2f} seconds with return code {e.returncode}")
        raise

def main():
    start_time = time.time()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq",  required=True, help="parquet with long Markov seqs")
    ap.add_argument("--sub",  required=True, help="parquet with subgroup columns")
    ap.add_argument("--time_blocks", default="night:0-5,morning:6-11,afternoon:12-17,evening:18-23")
    ap.add_argument("--dwell_bins",  default="1,2,3,4,6,9,14,20,30")
    ap.add_argument("--out",  required=True, help="root output dir  (…/models)")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--also_b2", action="store_true",
                    help="if set, train a B2-H model for EVERY rung, not just R7")
    args = ap.parse_args()

    # Log experiment configuration
    logging.info("="*60)
    logging.info("STARTING ATUS ABLATION LADDER EXPERIMENTS")
    logging.info("="*60)
    logging.info(f"Sequences file: {args.seq}")
    logging.info(f"Subgroups file: {args.sub}")
    logging.info(f"Output directory: {args.out}")
    logging.info(f"Random seed: {args.seed}")
    logging.info(f"Test split size: {args.test_size}")
    logging.info(f"Time blocks: {args.time_blocks}")
    logging.info(f"Dwell bins: {args.dwell_bins}")
    logging.info(f"Run B2-H for all rungs: {args.also_b2}")

    root_out = Path(args.out).resolve()
    root_out.mkdir(parents=True, exist_ok=True)

    # One fixed split file shared by all rungs
    split_path = root_out / "fixed_split.parquet"
    logging.info(f"Split file will be saved/loaded from: {split_path}")

    summary_rows = []
    total_rungs = len(SPECS)
    
    logging.info(f"Will process {total_rungs} rungs: {list(SPECS.keys())}")

    for rung_idx, (rung, groupby) in enumerate(SPECS.items(), 1):
        logging.info("="*40)
        logging.info(f"Processing rung {rung_idx}/{total_rungs}: {rung}")
        logging.info(f"Groupby dimensions: {groupby}")
        
        is_r7 = (rung == "R7_full_dwell")
        out_dir = root_out / rung
        out_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Output directory for {rung}: {out_dir}")
        will_run_b2h = is_r7 or args.also_b2
        logging.info(f"Will run B2-H (hazard) model: {will_run_b2h}")

        # -------------------- B1-H --------------------
        logging.info(f"Starting B1-H (routing) model for {rung}...")
        cmd = [
            "python", "-m", "atus_analysis.scripts.baseline1_hier",
            "--sequences", args.seq,
            "--subgroups", args.sub,
            "--out_dir", str(out_dir),
            "--groupby", groupby,
            "--time_blocks", args.time_blocks,
            "--seed", str(args.seed),
            "--test_size", str(args.test_size),
            "--split_path", str(split_path),
        ]
        call(cmd)
        logging.info(f"✓ B1-H model completed for {rung}")
        summary_rows.append({"rung": rung,
                             "model": "B1H",
                             "eval_file": str(out_dir / "eval_b1h.json")})

        # -------------------- B2-H (if requested) --------------------
        if will_run_b2h:
            logging.info(f"Starting B2-H (hazard) model for {rung}...")
            cmd = [
                "python", "-m", "atus_analysis.scripts.baseline2_hier",
                "--sequences", args.seq,
                "--subgroups", args.sub,
                "--out_dir", str(out_dir),
                "--groupby", groupby,
                "--time_blocks", args.time_blocks,
                "--dwell_bins", args.dwell_bins,
                "--seed", str(args.seed),
                "--test_size", str(args.test_size),
                "--split_path", str(split_path),

                # reuse routing from the same rung
                "--b1h_path", str(out_dir / "b1h_model.json"),
            ]
            call(cmd)
            logging.info(f"✓ B2-H model completed for {rung}")
            summary_rows.append({"rung": rung,
                                 "model": "B2H",
                                 "eval_file": str(out_dir / "eval_b2h.json")})
        
        logging.info(f"✓ Completed rung {rung} ({rung_idx}/{total_rungs})")

    # ------------ tiny manifest for convenience ------------
    manifest = root_out / "ablation_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    
    total_time = time.time() - start_time
    logging.info("="*60)
    logging.info(f"✓ Ablation ladder finished in {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    logging.info(f"Manifest saved to: {manifest}")
    logging.info(f"Total models trained: {len(summary_rows)}")
    logging.info("="*60)
    print("\n✓ Ladder finished.  Manifest →", manifest)

if __name__ == "__main__":
    main()
