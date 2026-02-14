from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class OnlineGraphState:
    prior_base: float
    alpha: float
    # (key_cols) -> (global_cnt, global_pos)
    global_stats: dict[tuple[str, ...], tuple[float, float]]
    # (key_cols) -> dict[key_tuple] = [cnt, pos, last_ts]
    states: dict[tuple[str, ...], dict[tuple[object, ...], list[float]]]


def _safe_val(v: object) -> object:
    if pd.isna(v):
        return "__NA__"
    return v


def _feature_prefix_name(key_cols: tuple[str, ...]) -> str:
    return "ogr_" + "__".join(key_cols)


def _weighted_prior(
    df: pd.DataFrame, label_col: str, sample_weight: pd.Series | None
) -> float:
    labeled_mask = df[label_col] >= 0
    if labeled_mask.sum() == 0:
        return 0.5
    y = (df.loc[labeled_mask, label_col] == 1).astype(np.float64).to_numpy()
    if sample_weight is None:
        return float(y.mean())
    w = sample_weight.loc[labeled_mask].to_numpy(dtype=np.float64)
    denom = float(w.sum())
    if denom <= 0:
        return 0.5
    return float((y * w).sum() / denom)


def build_online_graph_features(
    df: pd.DataFrame,
    keys: Iterable[tuple[str, ...]],
    label_col: str,
    time_col: str,
    event_id_col: str,
    mode: str,
    prior: float,
    alpha: float,
    init_state: OnlineGraphState | None = None,
    update_labeled_only: bool = True,
    include_risk: bool = True,
    include_count: bool = True,
) -> tuple[pd.DataFrame, OnlineGraphState]:
    if mode not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported mode={mode}")

    key_list = [tuple(k) for k in keys]
    sort_df = df[[time_col, event_id_col]].copy()
    sort_df["_idx"] = np.arange(len(df), dtype=np.int64)
    sort_df = sort_df.sort_values([time_col, event_id_col], kind="mergesort")
    order = sort_df["_idx"].to_numpy(dtype=np.int64)

    n = len(df)
    out = df.copy()
    ts = out[time_col].to_numpy(dtype=np.float64)
    labels = out[label_col].to_numpy(dtype=np.int32) if label_col in out.columns else None

    if init_state is None:
        states: dict[tuple[str, ...], dict[tuple[object, ...], list[float]]] = {
            k: {} for k in key_list
        }
        state_prior_base = prior
        state_alpha = alpha
        global_stats: dict[tuple[str, ...], tuple[float, float]] = {
            k: (0.0, 0.0) for k in key_list
        }
    else:
        states = {k: dict(v) for k, v in init_state.states.items()}
        for k in key_list:
            states.setdefault(k, {})
        state_prior_base = init_state.prior_base
        state_alpha = init_state.alpha
        global_stats = dict(init_state.global_stats)
        for k in key_list:
            global_stats.setdefault(k, (0.0, 0.0))

    for key_cols in key_list:
        pfx = _feature_prefix_name(key_cols)
        cnt_prev = np.zeros(n, dtype=np.float32)
        pos_prev = np.zeros(n, dtype=np.float32)
        risk = np.zeros(n, dtype=np.float32)
        tslast = np.full(n, -1.0, dtype=np.float32)
        is_new = np.ones(n, dtype=np.int8)

        key_values = [out[c].to_numpy() for c in key_cols]
        d = states[key_cols]
        global_cnt, global_pos = global_stats[key_cols]

        for i in order:
            key = tuple(_safe_val(v[i]) for v in key_values)
            rec = d.get(key)
            if rec is None:
                c = 0.0
                p = 0.0
                last = None
            else:
                c, p, last = rec[0], rec[1], rec[2]

            prior_cur = (
                global_pos / global_cnt if global_cnt > 0.0 else state_prior_base
            )
            cnt_prev[i] = c
            pos_prev[i] = p
            risk[i] = (p + state_alpha * prior_cur) / (c + state_alpha)
            if last is not None:
                tslast[i] = float(ts[i] - last)
                is_new[i] = 0

            if mode != "train":
                continue

            do_update = True
            if update_labeled_only and labels is not None:
                do_update = labels[i] >= 0
            if not do_update:
                continue

            c_new = c + 1.0
            p_new = p + (1.0 if labels is not None and labels[i] == 1 else 0.0)
            d[key] = [c_new, p_new, float(ts[i])]
            global_cnt += 1.0
            global_pos += 1.0 if labels is not None and labels[i] == 1 else 0.0

        global_stats[key_cols] = (global_cnt, global_pos)

        if include_count:
            out[f"{pfx}_cnt_prev"] = cnt_prev
            out[f"{pfx}_is_new"] = is_new
            out[f"{pfx}_tslast_sec"] = tslast
        if include_risk:
            out[f"{pfx}_pos_prev"] = pos_prev
            out[f"{pfx}_risk_smooth"] = risk

    return out, OnlineGraphState(
        prior_base=state_prior_base,
        alpha=state_alpha,
        global_stats=global_stats,
        states=states,
    )


def fit_graph_prior(
    df: pd.DataFrame, label_col: str, sample_weight: pd.Series | None = None
) -> float:
    return _weighted_prior(df, label_col=label_col, sample_weight=sample_weight)
