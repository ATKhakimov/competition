#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    index_path = Path("artifacts/runs/runs_index.csv")
    if not index_path.exists():
        print("No runs yet: artifacts/runs/runs_index.csv not found")
        return
    df = pd.read_csv(index_path)
    if "model_family" in df.columns:
        df["model_family"] = df["model_family"].fillna("xgb")
    cols = [
        "run_name",
        "timestamp_utc",
        "model_family",
        "cv_ap_all_events_mean",
        "cv_ap_all_events_sampled_mean",
        "cv_ap_labeled_mean",
        "cv_ap_proxy_mean",
        "cv_ap_proxy_lastday_mean",
        "cv_ap_seen_mean",
        "cv_ap_cold_mean",
        "cv_valid_pos_rate_primary_mean",
        "graph_risk_mode",
        "use_sequence",
        "include_unlabeled",
        "full_week_eval_unsampled",
        "runtime_device",
        "disable_graph_risk",
        "train_rows",
        "submission_path",
    ]
    cols = [c for c in cols if c in df.columns]
    sort_col = "cv_ap_all_events_mean" if "cv_ap_all_events_mean" in df.columns else "timestamp_utc"
    print(df.sort_values(sort_col, ascending=False)[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
