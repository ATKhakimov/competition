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
    cols = [
        "run_name",
        "timestamp_utc",
        "cv_ap_labeled_mean",
        "cv_ap_proxy_mean",
        "cv_ap_proxy_lastday_mean",
        "runtime_device",
        "disable_graph_risk",
        "train_rows",
        "submission_path",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
