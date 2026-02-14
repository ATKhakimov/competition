from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


PROFILE_COLS = [
    "customer_id",
    "pre_profile_events",
    "pre_profile_active_days",
    "pre_profile_amt_mean",
    "pre_profile_amt_std",
    "pre_profile_unique_event_type",
    "pre_profile_unique_event_desc",
    "pre_profile_unique_channel_type",
    "pre_profile_unique_channel_sub_type",
    "pre_profile_unique_mcc",
    "pre_profile_last_ts",
]


def _cfg_path(cfg: dict, key: str, default: str) -> str:
    return str(cfg["paths"].get(key, default))


def load_or_build_pretrain_profile(cfg: dict) -> pd.DataFrame:
    data_dir = Path(cfg["paths"]["data_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    cache_dir = artifacts_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "customer_profile_pretrain.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    pretrain_glob = _cfg_path(cfg, "pretrain_glob", "pretrain_part_*.parquet")
    lf = pl.scan_parquet(str(data_dir / pretrain_glob)).select(
        [
            "customer_id",
            "event_dttm",
            "operaton_amt",
            "event_type_nm",
            "event_desc",
            "channel_indicator_type",
            "channel_indicator_sub_type",
            "mcc_code",
        ]
    )
    dt = pl.col("event_dttm").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
    profile = (
        lf.with_columns(
            dt.alias("event_dt"),
            dt.dt.epoch("s").alias("event_ts"),
            dt.dt.date().alias("event_date"),
        )
        .group_by("customer_id")
        .agg(
            pl.len().alias("pre_profile_events"),
            pl.col("event_date").n_unique().alias("pre_profile_active_days"),
            pl.col("operaton_amt").fill_null(0.0).mean().alias("pre_profile_amt_mean"),
            pl.col("operaton_amt").fill_null(0.0).std().fill_null(0.0).alias("pre_profile_amt_std"),
            pl.col("event_type_nm").n_unique().alias("pre_profile_unique_event_type"),
            pl.col("event_desc").n_unique().alias("pre_profile_unique_event_desc"),
            pl.col("channel_indicator_type").n_unique().alias("pre_profile_unique_channel_type"),
            pl.col("channel_indicator_sub_type")
            .n_unique()
            .alias("pre_profile_unique_channel_sub_type"),
            pl.col("mcc_code").n_unique().alias("pre_profile_unique_mcc"),
            pl.col("event_ts").max().alias("pre_profile_last_ts"),
        )
        .collect(streaming=True)
    )
    profile.write_parquet(cache_path)
    return profile.to_pandas()


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df

    out = df.copy()
    out = out.sort_values(["customer_id", "event_ts", "event_id"], kind="mergesort")
    n = len(out)

    seq_idx = np.zeros(n, dtype=np.int32)
    dt_prev = np.zeros(n, dtype=np.float32)
    roll_1h = np.zeros(n, dtype=np.int32)
    roll_1d = np.zeros(n, dtype=np.int32)
    roll_7d = np.zeros(n, dtype=np.int32)
    amt_over_prev_mean = np.zeros(n, dtype=np.float32)
    is_new_event_type = np.zeros(n, dtype=np.int8)
    is_new_channel_sub_type = np.zeros(n, dtype=np.int8)
    is_new_mcc = np.zeros(n, dtype=np.int8)

    cust = out["customer_id"].to_numpy()
    ts = out["event_ts"].to_numpy(dtype=np.int64)
    amt = out["operaton_amt"].fillna(0.0).to_numpy(dtype=np.float64)
    ev = out["event_type_nm"].to_numpy()
    ch_sub = out["channel_indicator_sub_type"].to_numpy()
    mcc = out["mcc_code"].astype("string").fillna("<NULL>").to_numpy()

    boundaries = np.r_[0, np.flatnonzero(cust[1:] != cust[:-1]) + 1, n]
    for bi in range(len(boundaries) - 1):
        s, e = int(boundaries[bi]), int(boundaries[bi + 1])
        t = ts[s:e]
        a = amt[s:e]

        idx = np.arange(e - s, dtype=np.int32)
        seq_idx[s:e] = idx

        dp = np.zeros(e - s, dtype=np.float32)
        if e - s > 1:
            dp[1:] = (t[1:] - t[:-1]).astype(np.float32)
        dt_prev[s:e] = dp

        left_1h = np.searchsorted(t, t - 3600, side="left")
        left_1d = np.searchsorted(t, t - 86400, side="left")
        left_7d = np.searchsorted(t, t - 86400 * 7, side="left")
        roll_1h[s:e] = idx - left_1h + 1
        roll_1d[s:e] = idx - left_1d + 1
        roll_7d[s:e] = idx - left_7d + 1

        prev_mean = np.zeros(e - s, dtype=np.float64)
        if e - s > 1:
            prev_mean[1:] = np.cumsum(a[:-1]) / np.arange(1, e - s)
        amt_over_prev_mean[s:e] = (a / (prev_mean + 1.0)).astype(np.float32)

        seen_ev: set[object] = set()
        seen_ch: set[object] = set()
        seen_mcc: set[object] = set()
        for j in range(s, e):
            if ev[j] not in seen_ev:
                is_new_event_type[j] = 1
                seen_ev.add(ev[j])
            if ch_sub[j] not in seen_ch:
                is_new_channel_sub_type[j] = 1
                seen_ch.add(ch_sub[j])
            if mcc[j] not in seen_mcc:
                is_new_mcc[j] = 1
                seen_mcc.add(mcc[j])

    out["seq_customer_idx"] = seq_idx
    out["dt_since_prev_sec"] = dt_prev
    out["rolling_count_1h"] = roll_1h
    out["rolling_count_1d"] = roll_1d
    out["rolling_count_7d"] = roll_7d
    out["burst_1h_over_1d"] = (roll_1h.astype(np.float32) / (roll_1d.astype(np.float32) + 1e-6))
    out["amt_over_prev_mean"] = amt_over_prev_mean
    out["is_new_event_type"] = is_new_event_type
    out["is_new_channel_sub_type"] = is_new_channel_sub_type
    out["is_new_mcc"] = is_new_mcc

    return out.sort_index()

