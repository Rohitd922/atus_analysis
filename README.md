# ATUS Analysis

## Overview
This project implements a complete pipeline for modeling daily activity sequences from the American Time Use Survey (ATUS). It converts raw survey files into processed tables, builds minute-level sequences, and trains hierarchical routing and hazard models through a series of experiments.

## Repository structure
```
atus_analysis/
├── assets/          # figures, tables, lexicon map
├── scripts/         # command-line data & modeling utilities
├── utils/           # shared helpers
└── notebooks/       # exploratory notebooks
```
Data files are expected under `atus_analysis/data/` (not versioned):
- `data/raw/` – raw ATUS `.dat` files
- `data/processed/` – cleaned respondent/activity tables and subgroup labels
- `data/sequences/` – generated grids, Markov sequences, and transition matrices
- `data/models/` – output models for rungs

## Data preparation
1. **Respondent & activity tables** – `scripts/build_dataset.py` reads raw ATUS files and writes `respondent.parquet` and `activity.parquet` to `data/processed`.
2. **Sequence grids** – `scripts/make_sequences.py` converts `activity.parquet` into 10‑minute grids (`grid_10min.npy`/`grid_ids.csv`) and ragged sequences (`ragged_sequences.parquet`) stored in `data/sequences`.
3. **Lexicon map** – `scripts/make_lexicon_map.py` parses the official ATUS lexicon PDF and outputs `assets/atus_lexicon_map.csv` mapping activity codes to taxonomy.
4. **Markov-ready sequences** – `scripts/make_markov_sequences.py` merges the grids with the lexicon to produce state‑labeled sequences and transition matrices in `data/sequences`.
5. **Subgroup derivation** – `scripts/make_subgroups.py` adds demographic/day-type labels and saves `subgroups.parquet`, `subgroups_summary.parquet`, and a schema JSON under `data/processed`.

## Experiment families

We now support three experiment families:

### A) Global baseline
- **R0** – Global model with **no grouping** (single population).

### B) Single-covariate rungs (new)
Used for covariate importance relative to **R0**. Each rung groups on exactly one field:

| Code | Grouping variable |
|------|-------------------|
| S1   | sex |
| S2   | employment |
| S3   | day_type |
| S4   | hh_size_band |
| S5   | region |
| S6   | quarter |

These runs share the same split file as the ladder (see below) so metrics are directly comparable to R0.

### C) Hierarchical ladder (existing)
Progressively richer routing; R7 adds dwell-time hazard:

| Rung | Grouping variables |
|------|--------------------|
| R1 | region |
| R2 | region, sex |
| R3 | region, employment |
| R4 | region, day_type |
| R5 | region, hh_size_band |
| R6 | employment, day_type, hh_size_band, sex, region, quarter |
| R7 | employment, day_type, hh_size_band, sex, region, quarter (**adds hazard**) |

### Required inputs
All runs require:
- `data/sequences/markov_sequences.parquet`
- `data/processed/subgroups.parquet`
- optional `data/models/fixed_split.parquet` to share a common train/test split

### Artifacts written per run
- **Routing (B1-H)** – `baseline1_hier.py` writes:
  - `b1h_model.json` (plus `b1h_slot_mats.npz` for 144 per-slot matrices)
  - `eval_b1h.json`
  - `split_assignments.parquet` (or reuses a shared split)
- **Hazard (B2-H)** – `baseline2_hier.py` (when enabled) adds:
  - `b2h_model.json`
  - `eval_b2h.json`

## Evaluation
`dump_case_metrics.py` evaluates a trained model on held‑out respondents, producing `test_case_metrics.parquet` with per‑case negative log-likelihood and top‑1 accuracy. Aggregate metrics are also stored in each rung’s `eval_*.json`.

---

## Example workflows

### 1) Prepare data (one-time)
```bash
python atus_analysis/scripts/build_dataset.py
python atus_analysis/scripts/make_sequences.py
python atus_analysis/scripts/make_lexicon_map.py --pdf <lexicon.pdf> --out atus_analysis/assets/atus_lexicon_map.csv
python atus_analysis/scripts/make_markov_sequences.py
python atus_analysis/scripts/make_subgroups.py
```

### 2) Run the full ladder (R1–R7)
```bash
python atus_analysis/scripts/run_ladders.py --mode full  # runs R1–R7
```

### 3) Run single-covariate rungs (S1–S6) — two options

**Option A (if available):** via the orchestrator
```bash
python atus_analysis/scripts/run_ladders.py --mode singles  # runs S1–S6
```

**Option B:** direct calls to the baseline scripts

Routing only (B1-H):
```bash
# Example: S3 (day_type)
python -m atus_analysis.scripts.baseline1_hier   --sequences atus_analysis/data/sequences/markov_sequences.parquet   --subgroups atus_analysis/data/processed/subgroups.parquet   --out_dir atus_analysis/data/models/S3   --groupby day_type   --time_blocks "night:0-5,morning:6-11,afternoon:12-17,evening:18-23"   --seed 2025   --test_size 0.2   --split_path atus_analysis/data/models/fixed_split.parquet
```

Routing + Hazard (B2-H) when you want dwell-time modeling on a single covariate:
```bash
# Example: S5 (region)
python -m atus_analysis.scripts.baseline2_hier   --sequences atus_analysis/data/sequences/markov_sequences.parquet   --subgroups atus_analysis/data/processed/subgroups.parquet   --out_dir atus_analysis/data/models/S5   --groupby region   --time_blocks "night:0-5,morning:6-11,afternoon:12-17,evening:18-23"   --dwell_bins "1,2,3,4,6,9,14,20,30"   --seed 2025   --test_size 0.2   --split_path atus_analysis/data/models/fixed_split.parquet
```

> **Tip:** Keep using `data/models/fixed_split.parquet` for all S* and R* runs to ensure **paired** comparisons (bootstrap or per-case deltas).

### 4) Evaluate any run
```bash
python atus_analysis/scripts/dump_case_metrics.py   --model_type b1h   --run_dir atus_analysis/data/models/S3   --sequences atus_analysis/data/sequences/markov_sequences.parquet   --subgroups atus_analysis/data/processed/subgroups.parquet   --time_blocks "night:0-5,morning:6-11,afternoon:12-17,evening:18-23"
```

---

## Reproducibility & tips
- Use the same `fixed_split.parquet` for all model families (R0, S*, R1–R7) to support paired bootstrap comparisons.
- For ablations that combine multiple covariates (best 2, 3, 4), pass multiple fields to `--groupby` (e.g., `--groupby sex,employment`).
- The B1-H router can optionally emit **144 per-slot** transition matrices; B2-H overlays a **semi‑Markov hazard** (dwell-time) on top of the same routing backbone.

## Utilities and notebooks
- `utils/atus_io.py` provides helpers for loading raw `.dat` files.
- `notebooks/` contains validation and figure-generation notebooks.

## Requirements
Dependencies live in `requirements.txt` and the conda `environment.yml`.

---

### Changelog
- **Added:** Single-covariate rung family (S1–S6) for covariate-importance analysis against R0.
- **Clarified:** Workflow examples for running single-covariate experiments and evaluating with paired splits.
