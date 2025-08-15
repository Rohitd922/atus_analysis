# ATUS Analysis

## Overview
This project implements a complete pipeline for modeling daily activity sequences from the American Time Use Survey (ATUS).  It converts raw survey files into processed tables, builds minute-level sequences, and trains hierarchical routing and hazard models through a series of experiments (R1–R7).

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
- `data/models/` – output models for rungs R1–R7

## Data preparation
1. **Respondent & activity tables** – `scripts/build_dataset.py` reads raw ATUS files and writes `respondent.parquet` and `activity.parquet` to `data/processed`【F:atus_analysis/scripts/build_dataset.py†L46-L47】【F:atus_analysis/scripts/build_dataset.py†L114-L115】
2. **Sequence grids** – `scripts/make_sequences.py` converts `activity.parquet` into 10‑minute grids (`grid_10min.npy`/`grid_ids.csv`) and ragged sequences (`ragged_sequences.parquet`) stored in `data/sequences`【F:atus_analysis/scripts/make_sequences.py†L6-L10】【F:atus_analysis/scripts/make_sequences.py†L44-L60】
3. **Lexicon map** – `scripts/make_lexicon_map.py` parses the official ATUS lexicon PDF and outputs `assets/atus_lexicon_map.csv` mapping activity codes to taxonomy【F:atus_analysis/scripts/make_lexicon_map.py†L1-L27】
4. **Markov-ready sequences** – `scripts/make_markov_sequences.py` merges the grids with the lexicon to produce state‑labeled sequences and transition matrices in `data/sequences`【F:atus_analysis/scripts/make_markov_sequences.py†L1-L17】【F:atus_analysis/scripts/make_markov_sequences.py†L129-L140】【F:atus_analysis/scripts/make_markov_sequences.py†L215-L248】
5. **Subgroup derivation** – `scripts/make_subgroups.py` adds demographic/day-type labels and saves `subgroups.parquet`, `subgroups_summary.parquet`, and a schema JSON under `data/processed`【F:atus_analysis/scripts/make_subgroups.py†L1-L9】【F:atus_analysis/scripts/make_subgroups.py†L300-L338】

## Experiments R1–R7
`scripts/run_ladders.py` orchestrates a ladder of hierarchical experiments.  Each rung trains a Dirichlet–multinomial routing model (B1‑H) with progressively richer subgroup sets; R7 additionally fits a dwell‑time hazard model (B2‑H).  Rung definitions are:

| Rung | Grouping variables |
|------|-------------------|
| R1 | region |
| R2 | region, sex |
| R3 | region, employment |
| R4 | region, day_type |
| R5 | region, hh_size_band |
| R6 | employment, day_type, hh_size_band, sex, region, quarter |
| R7 | employment, day_type, hh_size_band, sex, region, quarter (adds hazard) |

These mappings are defined in `run_ladders.py`【F:atus_analysis/scripts/run_ladders.py†L51-L60】

### Required inputs
All rungs require:
- `data/sequences/markov_sequences.parquet` (from step 4)
- `data/processed/subgroups.parquet` (from step 5)
- optional `data/models/fixed_split.parquet` to share train/test splits

### Produced files per rung
`baseline1_hier.py` writes the routing model and evaluation metrics (`b1h_model.json`, `eval_b1h.json`, `split_assignments.parquet`) in each rung's `data/models/R#/` directory【F:atus_analysis/scripts/baseline1_hier.py†L3-L7】【F:atus_analysis/scripts/baseline1_hier.py†L128-L158】

When hazard modeling is enabled (rung R7 or `--also_b2`), `baseline2_hier.py` adds `b2h_model.json` and `eval_b2h.json` alongside the routing outputs【F:atus_analysis/scripts/baseline2_hier.py†L4-L10】【F:atus_analysis/scripts/baseline2_hier.py†L211-L257】

## Evaluation
`dump_case_metrics.py` evaluates a trained model on held‑out respondents, producing `test_case_metrics.parquet` with per‑case negative log-likelihood and top‑1 accuracy【F:atus_analysis/scripts/dump_case_metrics.py†L1-L11】【F:atus_analysis/scripts/dump_case_metrics.py†L166-L176】.  Aggregate metrics are already stored in each rung's `eval_*.json` from the training scripts.

## Example workflow
```
python atus_analysis/scripts/build_dataset.py
python atus_analysis/scripts/make_sequences.py
python atus_analysis/scripts/make_lexicon_map.py --pdf <lexicon.pdf> --out atus_analysis/assets/atus_lexicon_map.csv
python atus_analysis/scripts/make_markov_sequences.py
python atus_analysis/scripts/make_subgroups.py
python atus_analysis/scripts/run_ladders.py --mode full  # runs R1–R7
```
Evaluation for a rung:
```
python atus_analysis/scripts/dump_case_metrics.py \
  --model_type b1h \
  --run_dir atus_analysis/data/models/R1 \
  --sequences atus_analysis/data/sequences/markov_sequences.parquet \
  --subgroups atus_analysis/data/processed/subgroups.parquet \
  --time_blocks "night:0-5,morning:6-11,afternoon:12-17,evening:18-23"
```

## Utilities and notebooks
- `utils/atus_io.py` supplies helpers for loading raw `.dat` files【F:atus_analysis/utils/atus_io.py†L1-L12】
- `notebooks/` contains validation and figure-generation notebooks.

## Requirements
Dependencies are listed in `requirements.txt` and the conda `environment.yml`.
