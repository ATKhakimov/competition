#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from competition.datasets import build_train_frame
from competition.graph_risk import build_online_graph_features
from competition.pipeline_config import load_config

GRAPH_KEYS = [
    ("customer_id", "mcc_code"),
    ("customer_id", "channel_indicator_sub_type"),
    ("customer_id", "event_type_nm"),
]


def _build_features(df: pd.DataFrame, prior: float, alpha: float) -> pd.DataFrame:
    feat, _ = build_online_graph_features(
        df=df.copy(),
        keys=GRAPH_KEYS,
        label_col="target",
        time_col="event_ts",
        event_id_col="event_id",
        mode="train",
        prior=prior,
        alpha=alpha,
        init_state=None,
        update_labeled_only=True,
        include_risk=True,
        include_count=True,
    )
    return feat


def _main() -> int:
    parser = argparse.ArgumentParser(description="Smoke checks for online graph features.")
    parser.add_argument("--config", default="conf/pipeline.yaml")
    parser.add_argument("--max-labeled-rows", type=int, default=5000)
    parser.add_argument("--max-unlabeled-rows", type=int, default=5000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    graph_cfg = cfg.get("graph", {})
    prior = float(graph_cfg.get("prior_base", 0.5))
    alpha = float(graph_cfg.get("alpha", 20.0))
    train_pl = build_train_frame(
        cfg,
        max_labeled_rows=args.max_labeled_rows,
        max_unlabeled_rows=args.max_unlabeled_rows,
    )
    df = train_pl.to_pandas(use_pyarrow_extension_array=False)
    df = df.sort_values(["event_ts", "event_id"], kind="mergesort").reset_index(drop=True)

    feat_full = _build_features(df, prior=prior, alpha=alpha)

    # 1) Causality check: remove future events, current-row features must stay equal.
    i = min(len(df) // 2, len(df) - 1)
    e_i = df.loc[i, "event_id"]
    df_prefix = df.iloc[: i + 1].copy().reset_index(drop=True)
    feat_prefix = _build_features(df_prefix, prior=prior, alpha=alpha)
    i_prefix = i

    feature_cols = [c for c in feat_full.columns if c.startswith("ogr_")]
    v1 = feat_full.loc[i, feature_cols].to_numpy(dtype=np.float64)
    v2 = feat_prefix.loc[i_prefix, feature_cols].to_numpy(dtype=np.float64)
    if not np.allclose(v1, v2, equal_nan=True):
        print("FAIL: causality check")
        return 1
    print("PASS: causality check")

    # 2) No-self-leak: changing own label should not change own-row features.
    if df.loc[i, "target"] >= 0:
        df_flip = df.copy()
        df_flip.loc[i, "target"] = 1 - int(df_flip.loc[i, "target"] == 1)
        feat_flip = _build_features(df_flip, prior=prior, alpha=alpha)
        vf = feat_flip.loc[i, feature_cols].to_numpy(dtype=np.float64)
        if not np.allclose(v1, vf, equal_nan=True):
            print("FAIL: no-self-leak check")
            return 1
        print("PASS: no-self-leak check")
    else:
        print("SKIP: no-self-leak check (picked unlabeled row)")

    # 3) Monotonic sanity checks.
    cnt_cols = [c for c in feature_cols if c.endswith("_cnt_prev")]
    risk_cols = [c for c in feature_cols if c.endswith("_risk_smooth")]
    if cnt_cols and (feat_full[cnt_cols].to_numpy() < 0).any():
        print("FAIL: monotonic sanity (negative cnt_prev)")
        return 1
    if risk_cols:
        risk_vals = feat_full[risk_cols].to_numpy(dtype=np.float64)
        if ((risk_vals < 0) | (risk_vals > 1)).any():
            print("FAIL: monotonic sanity (risk outside [0,1])")
            return 1
    print("PASS: monotonic sanity")

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
