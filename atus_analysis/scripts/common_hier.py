#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
common_hier.py
==============

Helpers for the hierarchical baselines used in the ATUS ablation ladder.

• B1-H  (routing)  : Dirichlet–multinomial shrinkage
• B2-H  (hazard)   : Beta–binomial   shrinkage

Vectorized (NumPy) implementations for fast counting and posterior means.
Optional generation of 144 per-slot transition matrices pooled from blocks.

No external Bayesian library is required; posterior means are computed
in closed form.

This version ensures:
- Block indices follow the SPEC order (not data-driven factorization).
- Slot→block mapping is guaranteed to align with the model’s block axis.
- Inference prefers group×slot posteriors and falls back to group×block/global.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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


def _row_normalize_last(a: np.ndarray) -> np.ndarray:
    """
    Row-normalize over the last axis (destination state).
    Works for 2D (K,K), 3D (G,K,K or B,K,K or S,K,K), or 4D (G,B,K,K) arrays.
    """
    s = a.sum(axis=-1, keepdims=True)
    s[s == 0.0] = 1.0
    return a / s


def block_of_slot_vec(spec: str = TIME_BLOCKS_DEFAULT) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      block_idx: np.ndarray[int16] of length 144 with the block id for each 10-min slot
      block_names: list of block names in SPEC order
    """
    blocks = parse_time_blocks(spec)
    names = [b[0] for b in blocks]  # SPEC order
    # Map each hour 0..23 to block id in SPEC order
    hour2bid = np.empty(24, dtype=np.int16)
    for bid, (_, lo, hi) in enumerate(blocks):
        hour2bid[lo:hi+1] = bid
    # Expand to 144 slots (10-min slots; hour = slot//6)
    hours = np.arange(144, dtype=np.int16) // 6
    return hour2bid[hours], names


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
    # >>> NEW: build 144 per-slot matrices pooled from block with strength kappa_slot
    compute_slot_mats: bool = False,
    kappa_slot: float = 0.0,
    time_blocks_spec: str = TIME_BLOCKS_DEFAULT,
) -> Dict:
    """
    Two-level Dirichlet–multinomial posterior means for each prev-state row,
    vectorized. Optionally also produces 144 per-slot matrices per group.

    If compute_slot_mats=True:
      For each slot t, prior = block posterior mean for block b(t) with strength kappa_slot.
      Posterior per-slot rows = (counts + kappa_slot * prior_row) / (row_sum + kappa_slot).
    """
    # --- Sort, build previous-row features --------------------------------
    tr = train.sort_values(["TUCASEID", "slot"])
    prev = tr.groupby("TUCASEID").shift(1)
    ok = prev["state_id"].notna()
    tr = tr.loc[ok].copy()

    tr["prev_state"] = prev.loc[ok, "state_id"].astype(int)
    tr["prev_block"] = prev.loc[ok, "block"]
    tr["prev_group"] = prev.loc[ok, "group_key"]
    tr["prev_slot"]  = prev.loc[ok, "slot"].astype(int)

    # --- Factorize groups for fast tensor indexing -------------------------
    g_codes, g_uniques = pd.factorize(tr["prev_group"], sort=True)
    G = int(g_uniques.size)
    K = int(n_states)

    # --- Build block axis from SPEC (not from data) ------------------------
    blocks_spec = parse_time_blocks(time_blocks_spec)
    spec_names  = [nm for (nm, _, _) in blocks_spec]       # SPEC order
    bmap = {nm: bi for bi, nm in enumerate(spec_names)}    # name → index (SPEC)
    # Map prev_block strings to SPEC indices; drop rows whose block isn't in SPEC (shouldn't happen)
    b_raw = tr["prev_block"].astype(str)
    b = b_raw.map(lambda x: bmap.get(x, -1)).to_numpy(np.int32)
    keep = b >= 0

    # Slice all arrays to kept rows
    i = tr.loc[keep, "prev_state"].to_numpy(np.int32)
    j = tr.loc[keep, "state_id"].to_numpy(np.int32)
    g = g_codes[keep].astype(np.int32, copy=False)
    b = b[keep]
    w = tr.loc[keep, w_col].to_numpy(np.float64)
    t = tr.loc[keep, "prev_slot"].to_numpy(np.int32)

    B = len(spec_names)

    # --- GLOBAL COUNTS (K×K) ----------------------------------------------
    C_global = np.zeros((K, K), dtype=np.float64)
    np.add.at(C_global, (i, j), w)

    # Posterior mean with tiny additive smoothing (uniform over columns)
    P_global = _row_normalize_last(C_global + add_k)

    # --- BLOCK COUNTS (B×K×K), in SPEC order -------------------------------
    C_block = np.zeros((B, K, K), dtype=np.float64)
    np.add.at(C_block, (b, i, j), w)

    # Block posterior means: shrink toward global with strength tau_block
    row_block = C_block.sum(axis=2, keepdims=True)  # (B,K,1)
    denom_block = row_block + tau_block
    denom_block[denom_block == 0.0] = 1.0
    P_block = (C_block + tau_block * P_global[None, :, :]) / denom_block  # (B,K,K)

    # --- GROUP×BLOCK COUNTS (G×B×K×K) -------------------------------------
    C_gb = np.zeros((G, B, K, K), dtype=np.float64)
    np.add.at(C_gb, (g, b, i, j), w)

    # Group×Block posterior means: shrink toward block templates with tau_group
    row_gb = C_gb.sum(axis=3, keepdims=True)  # (G,B,K,1)
    denom_gb = row_gb + tau_group
    denom_gb[denom_gb == 0.0] = 1.0
    P_gb = (C_gb + tau_group * P_block[None, :, :, :]) / denom_gb  # (G,B,K,K)

    # --- OPTIONAL: SLOT-LEVEL COUNTS (G×144×K×K) and posterior means ------
    slot_mats_by_group: Optional[Dict[str, List[List[List[float]]]]] = None
    if compute_slot_mats:
        # counts per group×slot×i×j from origin slot t
        C_gs = np.zeros((G, 144, K, K), dtype=np.float64)
        np.add.at(C_gs, (g, t, i, j), w)

        # SPEC-aligned slot → block-id mapping
        slot_block_ids = np.empty(144, dtype=np.int32)
        spec_name_by_hour = np.empty(24, dtype=object)
        for nm, lo, hi in blocks_spec:
            spec_name_by_hour[lo:hi+1] = nm
        for s in range(144):
            slot_block_ids[s] = bmap[spec_name_by_hour[s // 6]]

        # Posterior per slot; rows with zero exposure fall back to that slot's block prior
        P_gs = np.zeros_like(C_gs)
        for s in range(143):  # handle last slot after loop
            pb = P_block[slot_block_ids[s]]  # (K,K)
            post = C_gs[:, s, :, :] + kappa_slot * pb[None, :, :]  # (G,K,K)
            rs = post.sum(axis=-1, keepdims=True)                  # (G,K,1)
            zero_row = (rs == 0.0)
            # Safe division to get row-stochastic
            post = np.divide(post, rs, out=np.zeros_like(post), where=~zero_row)
            # Copy block prior rows where exposure is zero
            if zero_row.any():
                zr = zero_row[..., 0]  # (G,K)
                for gidx in range(G):
                    if zr[gidx].any():
                        post[gidx, zr[gidx], :] = pb[zr[gidx], :]
            P_gs[:, s, :, :] = post

        # For slot 143 (no outgoing), copy its block prior for convenience
        pb_last = P_block[slot_block_ids[143]]
        P_gs[:, 143, :, :] = pb_last[None, :, :]

        # Convert to JSON-friendly dict: { group_key: [144 matrices] }
        slot_mats_by_group = {}
        for g_idx, gk in enumerate(g_uniques):
            mats_144 = []
            for s in range(144):
                mats_144.append(P_gs[g_idx, s].tolist())
            slot_mats_by_group[str(gk)] = mats_144

    # ---------- Assemble JSON-friendly outputs (keep backward compatibility)
    # matrices: { "group_key::block_name" : KxK list } with SPEC block names
    matrices: Dict[str, List[List[float]]] = {}
    for g_idx, gk in enumerate(g_uniques):
        for b_idx, bk in enumerate(spec_names):
            matrices[f"{gk}::{bk}"] = P_gb[g_idx, b_idx].tolist()

    fallback_block = {str(bk): P_block[b_idx].tolist() for b_idx, bk in enumerate(spec_names)}

    out = {
        "n_states": int(n_states),
        "matrices": matrices,
        "fallback_block": fallback_block,
        "global": P_global.tolist(),
        "meta": {
            "type": "B1-H",
            "tau_block": float(tau_block),
            "tau_group": float(tau_group),
            "add_k": float(add_k),
            # per-slot metadata
            "compute_slot_mats": bool(compute_slot_mats),
            "kappa_slot": float(kappa_slot),
            "time_blocks_spec": str(time_blocks_spec),
            "blocks_order": [str(x) for x in spec_names],   # SPEC order guaranteed
            "groups_order": [str(x) for x in g_uniques],
        },
    }

    # include per-slot matrices if requested
    if compute_slot_mats and slot_mats_by_group is not None:
        out["slot_matrices"] = slot_mats_by_group  # { group_key: [144 x (KxK)] }

    return out


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


def _get_row_b1_slot(model: Dict, gk: str, slot: int, i: int) -> Optional[np.ndarray]:
    """
    Return p(i→·) from 144 per-slot matrices if available for this group+slot.
    Falls back to None if sidecar missing or group not present.
    Caches loaded arrays under model['_slot_cache'].
    """
    # 1) Inline JSON path (rare; large)
    if "slot_matrices" in model:
        gmats = model["slot_matrices"].get(gk)
        if gmats is not None and 0 <= slot < len(gmats):
            return np.asarray(gmats[slot], dtype=np.float64)[i, :]

    # 2) NPZ sidecar
    meta = model.get("meta", {}) or {}
    npz_path = model.get("slot_matrices_npz") or meta.get("slot_sidecar")
    groups_order = meta.get("groups_order")
    if not npz_path or not groups_order:
        return None

    try:
        gi = groups_order.index(gk)
    except ValueError:
        return None  # unseen group → let caller fall back to block/global

    key = f"g{gi}"
    cache = model.setdefault("_slot_cache", {})
    mats = cache.get(key)
    if mats is None:
        with np.load(npz_path) as npz:
            mats = np.array(npz[key], dtype=np.float64)  # (144,K,K)
        cache[key] = mats

    if not (0 <= slot < mats.shape[0]):
        return None
    return mats[slot, i, :]


def _get_row_b1_any(model: Dict, gk: str, block: str, slot: int, i: int) -> np.ndarray:
    """Try group×slot first; if not available, fall back to group×block/global."""
    prow = _get_row_b1_slot(model, gk, slot, i)
    if prow is not None:
        return prow
    return _get_row_b1(model, gk, block, i)


def nll_b1(test: pd.DataFrame, model: Dict, n_states: int, w_col: str) -> Dict[str, float]:
    te = test.sort_values(["TUCASEID", "slot"])
    prev = te.groupby("TUCASEID").shift(1)
    ok = prev["state_id"].notna()
    te = te.loc[ok].copy()

    te["prev_state"] = prev.loc[ok, "state_id"].astype(int)
    te["prev_block"] = prev.loc[ok, "block"]
    te["prev_group"] = prev.loc[ok, "group_key"]
    te["prev_slot"]  = prev.loc[ok, "slot"].astype(int)

    total_w = total_nll = top1_w = 0.0
    for _, row in te.iterrows():
        i = int(row["prev_state"])
        j = int(row["state_id"])
        prow = _get_row_b1_any(model, row["prev_group"], row["prev_block"], int(row["prev_slot"]), i)

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
    tr["d_bin"] = prev.loc[ok, "runlen"].astype(int).apply(
        lambda d: dwell_bin_index(int(d), dwell_edges)
    )

    n_bins = len(dwell_edges) + 1

    # ---------- GLOBAL ---------------------------------------------
    leave0 = np.zeros((n_states, n_bins))
    tot0   = np.zeros((n_states, n_bins))

    ii = tr["prev_state"].to_numpy(np.int32)
    dd = tr["d_bin"].to_numpy(np.int32)
    ww = tr[w_col].to_numpy(np.float64)
    ll = tr["leave"].to_numpy(np.float64)

    np.add.at(leave0, (ii, dd), ww * ll)
    np.add.at(tot0,   (ii, dd), ww)

    h0 = (leave0 + k0) / (tot0 + 2.0 * k0)

    # ---------- BLOCK LEVEL (SPEC order) ----------------------------
    # read block spec from routing meta to guarantee alignment
    meta_route = routing_b1h.get("meta", {}) or {}
    time_blocks_spec = meta_route.get("time_blocks_spec", TIME_BLOCKS_DEFAULT)
    blocks_spec = parse_time_blocks(time_blocks_spec)
    spec_names  = [nm for (nm, _, _) in blocks_spec]
    bmap = {nm: bi for bi, nm in enumerate(spec_names)}

    b_raw = tr["prev_block"].astype(str)
    b = b_raw.map(lambda x: bmap.get(x, -1)).to_numpy(np.int32)
    keep = b >= 0

    ii_k = ii[keep]
    dd_k = dd[keep]
    ww_k = ww[keep]
    ll_k = ll[keep]
    b_k  = b[keep]

    B = len(spec_names)
    leaveB = np.zeros((B, n_states, n_bins))
    totB   = np.zeros((B, n_states, n_bins))
    np.add.at(leaveB, (b_k, ii_k, dd_k), ww_k * ll_k)
    np.add.at(totB,   (b_k, ii_k, dd_k), ww_k)

    hB = (leaveB + tau_block * h0[None, :, :]) / (totB + tau_block)

    # ---------- GROUP × BLOCK -------------------------------------
    g_codes, g_uniques = pd.factorize(tr["prev_group"], sort=True)
    hazard: Dict[str, float] = {}

    # Iterate grouped keys to apply tau_group pooling to block template
    for (gk, bname, i, d), grp in tr.groupby(
        ["prev_group", "prev_block", "prev_state", "d_bin"], sort=False
    ):
        if str(bname) not in bmap:
            continue  # skip rows whose block isn't in spec (shouldn't happen)
        b_idx = bmap[str(bname)]
        wsum = float(grp[w_col].sum())
        lsum = float((grp[w_col] * grp["leave"]).sum())
        hb = float(hB[b_idx, i, d])
        h = (lsum + tau_group * hb) / (wsum + tau_group)
        hazard[f"{gk}::{bname}::s{i}::d{d}"] = h

    hazard_block = {
        f"{bname}::s{i}::d{d}": float(hB[b_idx, i, d])
        for b_idx, bname in enumerate(spec_names)
        for i in range(n_states) for d in range(n_bins)
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
            "tau_block": float(tau_block),
            "tau_group": float(tau_group),
            "k0": float(k0),
            "blocks_order": [str(x) for x in spec_names],  # alignment aid
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


def nll_b2(test: pd.DataFrame, model: Dict, dwell_edges: List[int], n_states: int, w_col: str) -> Dict[str, float]:
    b1 = model["routing_b1h"]
    te = test.sort_values(["TUCASEID", "slot"])
    te = compute_runlengths_per_respondent(te)
    prev = te.groupby("TUCASEID").shift(1)
    ok = prev["state_id"].notna()
    te = te.loc[ok].copy()

    te["prev_state"] = prev.loc[ok, "state_id"].astype(int)
    te["prev_block"] = prev.loc[ok, "block"]
    te["prev_group"] = prev.loc[ok, "group_key"]
    te["prev_slot"]  = prev.loc[ok, "slot"].astype(int)
    te["d_bin"] = prev.loc[ok, "runlen"].astype(int).apply(
        lambda d: dwell_bin_index(int(d), dwell_edges)
    )

    total_w = total_nll = top1_w = 0.0
    for _, row in te.iterrows():
        i = int(row["prev_state"]); j = int(row["state_id"])
        gk = row["prev_group"]; bl = row["prev_block"]; slot = int(row["prev_slot"])
        d_bin = int(row["d_bin"]); w = float(row[w_col])

        h = _get_hazard_b2(model, gk, bl, i, d_bin)
        prow = _get_row_b1_any(b1, gk, bl, slot, i)  # prefer slot posterior, fallback to block/global

        denom = float(prow.sum() - prow[i])
        if denom <= 0.0:
            # degenerate row: spread leave mass uniformly over j≠i
            leave_to = np.full(n_states, 1.0 / max(n_states - 1, 1))
            if n_states > 1:
                leave_to[i] = 0.0
        else:
            leave_to = prow.copy(); leave_to[i] = 0.0; leave_to /= denom

        p_next = leave_to * h
        p_next[i] = 1.0 - h

        total_nll += -_safe_log(float(p_next[j])) * w
        top1_w += w if j == int(p_next.argmax()) else 0.0
        total_w += w

    return {
        "nll_per_weight": total_nll / max(total_w, 1e-9),
        "top1_acc_weighted": top1_w / max(total_w, 1e-9),
        "total_weight": total_w,
    }
