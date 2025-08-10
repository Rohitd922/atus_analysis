"""
Bootstrap the difference in weighted NLL between two runs A and B
(e.g. B1-H vs B2-H).  Expects each run directory to contain
test_case_metrics.parquet produced by dump_case_metrics.py.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# ────────────────────  BOOTSTRAP FUNCTION  ─────────────────────
def bootstrap_ci(dfA: pd.DataFrame, dfB: pd.DataFrame,
                 B: int, seed: int = 2025):
    """
    Returns (samples_df, summary_dict)
    """
    # align on TUCASEID
    df = dfA.merge(dfB, on="TUCASEID", suffixes=("_A", "_B"))
    rng = np.random.RandomState(seed)

    n = len(df)
    draws = []

    for _ in range(B):
        idx = rng.randint(0, n, n)   # simple bootstrap, no stratification
        sample = df.iloc[idx]

        WA = sample["weight_total_A"].sum()
        WB = sample["weight_total_B"].sum()  # same but keep notation

        nllA = sample["nll_weighted_A"].sum() / WA
        nllB = sample["nll_weighted_B"].sum() / WB
        topA = sample["top1_correct_weight_A"].sum() / WA
        topB = sample["top1_correct_weight_B"].sum() / WB

        draws.append({
            "nllA": nllA, "nllB": nllB, "delta": nllA - nllB,
            "topA": topA, "topB": topB, "delta_top": topA - topB,
        })

    samples = pd.DataFrame(draws)

    # summaries
    d_mean = float(samples["delta"].mean())
    lo, hi = np.percentile(samples["delta"], [2.5, 97.5])
    nllA_m = float(samples["nllA"].mean())
    nllB_m = float(samples["nllB"].mean())
    rel = (-d_mean / nllA_m) * 100.0

    d_top = float(samples["delta_top"].mean())
    lo_t, hi_t = np.percentile(samples["delta_top"], [2.5, 97.5])

    summary = {
        "B": B,
        "delta_nll_mean": d_mean,
        "delta_nll_ci95": [float(lo), float(hi)],
        "nllA_mean": nllA_m,
        "nllB_mean": nllB_m,
        "relative_improvement_percent": rel,
        "delta_top1_mean": d_top,
        "delta_top1_ci95": [float(lo_t), float(hi_t)],
        "note": "Negative delta favours run B (lower NLL).",
    }
    return samples, summary

# ───────────────────────────  MAIN  ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", required=True,
                    help="directory with test_case_metrics.parquet (baseline)")
    ap.add_argument("--run_b", required=True,
                    help="directory with test_case_metrics.parquet (candidate)")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    runA = Path(args.run_a)
    runB = Path(args.run_b)

    dfA = pd.read_parquet(runA / "test_case_metrics.parquet")
    dfB = pd.read_parquet(runB / "test_case_metrics.parquet")

    samples, summary = bootstrap_ci(dfA, dfB, args.B, args.seed)

    outdir = runB / f"bootstrap_vs_{runA.name}"
    outdir.mkdir(parents=True, exist_ok=True)

    samples.to_csv(outdir / "bootstrap_samples.csv", index=False)
    with open(outdir / "bootstrap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("✓ bootstrap results →", outdir)

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()