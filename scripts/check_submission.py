#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def main() -> int:
    parser = argparse.ArgumentParser(description="Submission sanity checks.")
    parser.add_argument("--submission", required=True, help="Path to submission csv")
    parser.add_argument("--test-file", default="db/test.parquet")
    args = parser.parse_args()

    sub_path = Path(args.submission)
    if not sub_path.exists():
        print(f"FAIL: submission not found: {sub_path}")
        return 1

    sub = pd.read_csv(sub_path)
    test_ids = pl.read_parquet(args.test_file).select("event_id").to_pandas()["event_id"]

    ok = True
    if len(sub) != len(test_ids):
        print(f"FAIL: row count mismatch: submission={len(sub)} test={len(test_ids)}")
        ok = False
    else:
        print(f"PASS: row count = {len(sub)}")

    dup = int(sub["event_id"].duplicated().sum())
    if dup > 0:
        print(f"FAIL: duplicated event_id rows = {dup}")
        ok = False
    else:
        print("PASS: no duplicated event_id")

    missing = int((~test_ids.isin(sub["event_id"])).sum())
    extra = int((~sub["event_id"].isin(test_ids)).sum())
    if missing > 0 or extra > 0:
        print(f"FAIL: id set mismatch: missing={missing} extra={extra}")
        ok = False
    else:
        print("PASS: event_id set matches test")

    same_order = bool(np.array_equal(sub["event_id"].to_numpy(), test_ids.to_numpy()))
    print(f"INFO: same order as test = {same_order}")

    score = sub["predict"].astype(float)
    finite_ok = bool(np.isfinite(score.to_numpy()).all())
    if not finite_ok:
        non_finite = int((~np.isfinite(score.to_numpy())).sum())
        print(f"FAIL: non-finite predict values = {non_finite}")
        ok = False
    else:
        print("PASS: all predict values are finite")

    uniq = int(score.nunique(dropna=False))
    if uniq <= 1:
        print("FAIL: predict is constant (single unique value)")
        ok = False
    else:
        print(f"PASS: predict unique values = {uniq}")

    q = score.quantile([0.5, 0.9, 0.99, 0.999]).to_dict()
    score_std = float(score.std())
    print(
        "score stats: "
        f"mean={score.mean():.6f} std={score_std:.6f} "
        f"q50={q.get(0.5):.6f} q90={q.get(0.9):.6f} "
        f"q99={q.get(0.99):.6f} q999={q.get(0.999):.6f}"
    )
    if score_std < 1e-8:
        print("WARN: score distribution is extremely narrow (std < 1e-8)")

    if ok:
        print("Submission sanity: PASS")
        return 0
    print("Submission sanity: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
