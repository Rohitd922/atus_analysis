#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
common_hier.py
==============

Helpers for the hierarchical baselines used in the ATUS
ablation ladder.

• B1-H  (routing)  : Dirichlet–multinomial shrinkage
• B2-H  (hazard)   : Beta–binomial   shrinkage

Both share the same utilities for time-block parsing, subgroup pooling,
design-weighted losses, and respondent-level run-length calculation.

No external Bayesian library is required; posterior means are computed
in closed form.

"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────  CONSTANTS  ──────────────────────────────
TIME_BLOCKS_DEFAULT = "night:0-5,morning:6-11,afternoon:12-17,evening:18-23"
DWELL_BINS_DEFAULT  = "1,2,3,4,6,9,14,20,30"

# ───────────────────────  TIME-BLOCK / DWELL PARSING  ────────────────────
def parse_time_blocks(spec: str) -> List[Tuple[str, int, int]]:
    """
    "night:0-5,morning:6-11"  →  [("night",0,5), ("morning",6,11)]
    """
    out: List[Tuple[str, int, int]] = []
    for token in spec.split(","):
        name, rng = token.split(":")
        lo, hi = map(int, rng.split("-"))
        out.append((name.strip(), lo, hi))
    return out


def assign_time_block(hour: int, blocks: List[Tuple[str, int, int]]) -> str:
    for name, lo, hi in blocks:
        if lo <= hour <= hi:
            return name
    return "all"


def parse_dwell_bins(spec: str) -> List[int]:
    """
    "1,2,3,4,6,9,14,20,30" → [1,2,3,4,6,9,14,20,30]
    """
    return sorted(int(x) for x in spec.split(",") if x.strip())


def dwell_bin_index(runlen: int, edges: List[int]) -> int:
    """
    Returns the 0-based bin index for a run-length.
    """
    for idx, edge in enumerate(edges):
        if runlen <= edge:
            return idx
    return len(edges)  # overflow bin

# ───────────────────────────────  I/O  ───────────────────────────────────
def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────  GROUP-KEY & DATA PREPARATION  ────────────────────
def pool_rare_quarter(
    groups: pd.DataFrame,
    cols: List[str],
    w_col: str,
    threshold: float,
) -> pd.DataFrame:
    """
    If 'quarter' ∈ cols and a (cols) key has total design-weight < threshold,
    collapse that key’s quarter to 'QALL'.
    """
    if "quarter" not in cols or threshold <= 0:
        return groups

    key_cols = cols.copy()
    tmp = groups[key_cols + [w_col, "TUCASEID"]].copy()
    tot = (
        tmp.groupby(key_cols, as_index=False)[w_col]
        .sum()
        .rename(columns={w_col: "__tot__"})
    )
    tmp = tmp.merge(tot, on=key_cols, how="left")

    rare = tmp["__tot__"] < float(threshold)
    tmp.loc[rare, "quarter"] = "QALL"
    tmp = tmp.drop(columns="__tot__")

    out = groups.drop(columns=["quarter"]).merge(
        tmp[["TUCASEID", "quarter"]], on="TUCASEID", how="left"
    )
    return out


def prepare_long_with_groups(
    long: pd.DataFrame,
    groups: pd.DataFrame,
    cols: List[str],
    w_col: str,
    blocks: List[Tuple[str, int, int]],
) -> pd.DataFrame:
    """
    Adds block + group_key columns and merges design-weights.
    """
    g = groups[["TUCASEID"] + cols + [w_col]].copy()
    df = long.merge(g, on="TUCASEID", how="inner").copy()

    df["hour"] = (df["slot"] // 6).astype(int)
    df["block"] = df["hour"].apply(lambda h: assign_time_block(h, blocks))
    for c in cols:
        df[c] = df[c].astype("string")

    df["group_key"] = df[cols].agg("|".join, axis=1)
    return df


def compute_runlengths_per_respondent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'runlen' column: consecutive repeats of the same state within each TUCASEID.
    """
    df = df.sort_values(["TUCASEID", "slot"]).copy()
    run, last_case, last_state = 0, None, None
    runlens: List[int] = []
    for case, st in zip(df["TUCASEID"].to_numpy(), df["state_id"].to_numpy()):
        if case != last_case:
            run = 1
        else:
            run = run + 1 if st == last_state else 1
        runlens.append(run)
        last_case, last_state = case, st
    df["runlen"] = runlens
    return df

# ─────────────────────────────  SAFETY  ──────────────────────────────────
def _safe_log(p: float) -> float:
    """log with floor to avoid −inf."""
    return math.log(max(float(p), 1e-15))

# ═════════════════════════  B1-H (ROUTING)  ══════════════════════════════
def fit_b1_hier(
    train: pd.DataFrame,
    n_states: int,
    w_col: str,
    *,
    tau_block: float = 50.0,
    tau_group: float = 20.0,
    add_k: float = 1.0,
) -> Dict:
    """
    Two-level Dirichlet–multinomial posterior mean for each prev-state row.
    """
    # --- build training matrix with previous state, block, group_key ----------
    tr = train.sort_values(["TUCASEID", "slot"])
    prev = tr.groupby("TUCASEID").shift(1)
    ok = prev["state_id"].notna()
    tr = tr.loc[ok].copy()

    tr["prev_state"] = prev.loc[ok, "state_id"].astype(int)
    tr["prev_block"] = prev.loc[ok, "block"]
    tr["prev_group"] = prev.loc[ok, "group_key"]

    # ---------- GLOBAL COUNTS -----------------------------------------------
    C_global = np.zeros((n_states, n_states), dtype=np.float64)
    np.add.at(
        C_global,
        (tr["prev_state"].to_numpy(), tr["state_id"].to_numpy()),
        tr[w_col].to_numpy(),
    )
    P_global = C_global + add_k
    row_sum = P_global.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    P_global /= row_sum

    # ---------- BLOCK TEMPLATES ---------------------------------------------
    block_keys = sorted(tr["prev_block"].unique().tolist())
    C_block = {b: np.zeros_like(C_global) for b in block_keys}
    for b, grp in tr.groupby("prev_block", sort=False):
        np.add.at(
            C_block[b],
            (grp["prev_state"].to_numpy(), grp["state_id"].to_numpy()),
            grp[w_col].to_numpy(),
        )

    P_block: Dict[str, np.ndarray] = {}
    for b in block_keys:
        mat = C_block[b] + tau_block * P_global
        denom = (C_block[b].sum(axis=1, keepdims=True) + tau_block)
        denom[denom == 0] = 1.0
        P_block[b] = mat / denom

    # ---------- GROUP × BLOCK MATRICES --------------------------------------
    matrices: Dict[str, List[List[float]]] = {}
    for (gk, b), grp in tr.groupby(["prev_group", "prev_block"], sort=False):
        M = np.zeros_like(C_global)
        np.add.at(
            M,
            (grp["prev_state"].to_numpy(), grp["state_id"].to_numpy()),
            grp[w_col].to_numpy(),
        )
        M = M + tau_group * P_block[b]
        denom = (M.sum(axis=1, keepdims=True))  # already includes tau_group counts
        denom[denom == 0] = 1.0
        matrices[f"{gk}::{b}"] = (M / denom).tolist()

    fallback_block = {str(b): P_block[b].tolist() for b in block_keys}

    return {
        "n_states": int(n_states),
        "matrices": matrices,
        "fallback_block": fallback_block,
        "global": P_global.tolist(),
        "meta": {
            "type": "B1-H",
            "tau_block": tau_block,
            "tau_group": tau_group,
            "add_k": add_k,
        },
    }


def _get_row_b1(model: Dict, gk: str, block: str, i: int) -> np.ndarray:
    """
    Retrieve the correct row with block → global fallbacks.
    """
    key = f"{gk}::{block}"
    if key in model["matrices"]:
        mat = np.array(model["matrices"][key], dtype=np.float64)
    elif str(block) in model["fallback_block"]:
        mat = np.array(model["fallback_block"][str(block)], dtype=np.float64)
    else:
        mat = np.array(model["global"], dtype=np.float64)
    return mat[i, :]


def nll_b1(
    test: pd.DataFrame,
    model: Dict,
    n_states: int,
    w_col: str,
) -> Dict[str, float]:
    """
    Weighted test NLL + top-1 accuracy for a B1-H model.
    """
    te = test.sort_values(["TUCASEID", "slot"])
    prev = te.groupby("TUCASEID").shift(1)
    ok = prev["state_id"].notna()
    te = te.loc[ok].copy()

    te["prev_state"] = prev.loc[ok, "state_id"].astype(int)
    te["prev_block"] = prev.loc[ok, "block"]
    te["prev_group"] = prev.loc[ok, "group_key"]

    total_w = total_nll = top1_w = 0.0
    for _, row in te.iterrows():
        i = int(row["prev_state"])
        j = int(row["state_id"])
        prow = _get_row_b1(model, row["prev_group"], row["prev_block"], i)

        p = float(prow[j])
        w = float(row[w_col])

        total_nll += -_safe_log(p) * w
        top1_w += w if j == int(prow.argmax()) else 0.0
        total_w += w

    return {
        "nll_per_weight": total_nll / max(total_w, 1e-9),
        "top1_acc_weighted": top1_w / max(total_w, 1e-9),
        "total_weight": total_w,
    }

# ═════════════════════════  B2-H (HAZARD)  ═══════════════════════════════
def fit_b2_hier(
    train: pd.DataFrame,
    n_states: int,
    w_col: str,
    dwell_edges: List[int],
    *,
    tau_block: float = 200.0,
    tau_group: float = 50.0,
    k0: float = 1.0,
    routing_b1h: Dict,
) -> Dict:
    """
    Two-level Beta–binomial posterior mean for leave-hazard, plus embedded routing.
    """
    tr = train.sort_values(["TUCASEID", "slot"])
    tr = compute_runlengths_per_respondent(tr)
    prev = tr.groupby("TUCASEID").shift(1)
    ok = prev["state_id"].notna()
    tr = tr.loc[ok].copy()

    tr["prev_state"] = prev.loc[ok, "state_id"].astype(int)
    tr["prev_block"] = prev.loc[ok, "block"]
    tr["prev_group"] = prev.loc[ok, "group_key"]
    tr["leave"] = (tr["state_id"] != tr["prev_state"]).astype(int)
    tr["d_bin"] = tr["runlen"].apply(lambda d: dwell_bin_index(int(d), dwell_edges))

    n_bins = len(dwell_edges) + 1

    # ---------- GLOBAL ---------------------------------------------
    leave0 = np.zeros((n_states, n_bins))
    tot0   = np.zeros((n_states, n_bins))

    for (i, d), grp in tr.groupby(["prev_state", "d_bin"]):
        w = grp[w_col].to_numpy()
        l = grp["leave"].to_numpy()
        leave0[i, d] += float((w * l).sum())
        tot0[i, d]   += float(w.sum())

    h0 = (leave0 + k0) / (tot0 + 2.0 * k0)

    # ---------- BLOCK LEVEL ----------------------------------------
    blocks = sorted(tr["prev_block"].unique().tolist())
    leaveB = {b: np.zeros_like(leave0) for b in blocks}
    totB   = {b: np.zeros_like(tot0)   for b in blocks}

    for (b, i, d), grp in tr.groupby(["prev_block", "prev_state", "d_bin"]):
        w = grp[w_col].to_numpy()
        l = grp["leave"].to_numpy()
        leaveB[b][i, d] += float((w * l).sum())
        totB[b][i, d]   += float(w.sum())

    hB = {
        b: (leaveB[b] + tau_block * h0) / (totB[b] + tau_block)
        for b in blocks
    }

    # ---------- GROUP × BLOCK -------------------------------------
    hazard: Dict[str, float] = {}
    for (gk, b, i, d), grp in tr.groupby(
        ["prev_group", "prev_block", "prev_state", "d_bin"], sort=False
    ):
        leave = float((grp[w_col] * grp["leave"]).sum())
        tot   = float(grp[w_col].sum())
        h = (leave + tau_group * hB[b][i, d]) / (tot + tau_group)
        hazard[f"{gk}::{b}::s{i}::d{d}"] = float(h)

    hazard_block = {
        f"{b}::s{i}::d{d}": float(hB[b][i, d])
        for b in blocks for i in range(n_states) for d in range(n_bins)
    }
    hazard_global = {
        f"s{i}::d{d}": float(h0[i, d])
        for i in range(n_states) for d in range(n_bins)
    }

    return {
        "n_states": int(n_states),
        "dwell_edges": dwell_edges,
        "hazard": hazard,
        "hazard_block": hazard_block,
        "hazard_global": hazard_global,
        "routing_b1h": routing_b1h,
        "meta": {
            "type": "B2-H",
            "tau_block": tau_block,
            "tau_group": tau_group,
            "k0": k0,
        },
    }


def _get_hazard_b2(model: Dict, gk: str, block: str, i: int, d_bin: int) -> float:
    """
    Retrieve hazard with fallbacks (group×block → block → global).
    """
    key = f"{gk}::{block}::s{i}::d{d_bin}"
    if key in model["hazard"]:
        return model["hazard"][key]
    key = f"{block}::s{i}::d{d_bin}"
    if key in model["hazard_block"]:
        return model["hazard_block"][key]
    return model["hazard_global"][f"s{i}::d{d_bin}"]


def nll_b2(
    test: pd.DataFrame,
    model: Dict,
    dwell_edges: List[int],
    n_states: int,
    w_col: str,
) -> Dict[str, float]:
    """
    Weighted test NLL + top-1 accuracy for a B2-H model.
    """
    b1 = model["routing_b1h"]
    te = test.sort_values(["TUCASEID", "slot"])
    te = compute_runlengths_per_respondent(te)
    prev = te.groupby("TUCASEID").shift(1)
    ok = prev["state_id"].notna()
    te = te.loc[ok].copy()

    te["prev_state"] = prev.loc[ok, "state_id"].astype(int)
    te["prev_block"] = prev.loc[ok, "block"]
    te["prev_group"] = prev.loc[ok, "group_key"]
    te["d_bin"] = prev.loc[ok, "runlen"].astype(int).apply(
        lambda d: dwell_bin_index(int(d), dwell_edges)
    )

    total_w = total_nll = top1_w = 0.0
    for _, row in te.iterrows():
        i = int(row["prev_state"])
        j = int(row["state_id"])
        gk = row["prev_group"]
        bl = row["prev_block"]
        d_bin = int(row["d_bin"])
        w = float(row[w_col])

        h = _get_hazard_b2(model, gk, bl, i, d_bin)
        prow = _get_row_b1(b1, gk, bl, i)

        denom = float(prow.sum() - prow[i])
        if denom <= 0:
            leave_to = np.full(n_states, 1.0 / (n_states - 1))
            leave_to[i] = 0.0
        else:
            leave_to = prow.copy()
            leave_to[i] = 0.0
            leave_to /= denom

        p_next = leave_to * h
        p_next[i] = 1.0 - h

        p = float(p_next[j])

        total_nll += -_safe_log(p) * w
        top1_w += w if j == int(p_next.argmax()) else 0.0
        total_w += w

    return {
        "nll_per_weight": total_nll / max(total_w, 1e-9),
        "top1_acc_weighted": top1_w / max(total_w, 1e-9),
        "total_weight": total_w,
    }
