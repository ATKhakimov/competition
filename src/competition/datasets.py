from __future__ import annotations

from pathlib import Path
from datetime import datetime

import polars as pl

from .features import FEATURE_COLUMNS, with_time_and_amount_features


def _paths(cfg: dict) -> tuple[Path, str, Path, Path]:
    data_dir = Path(cfg["paths"]["data_dir"])
    train_glob = cfg["paths"]["train_glob"]
    labels_path = data_dir / cfg["paths"]["labels_file"]
    test_path = data_dir / cfg["paths"]["test_file"]
    return data_dir, train_glob, labels_path, test_path


def build_train_frame(
    cfg: dict,
    max_labeled_rows: int | None = None,
    max_unlabeled_rows: int | None = None,
) -> pl.DataFrame:
    data_dir, train_glob, labels_path, _ = _paths(cfg)

    train_events = pl.scan_parquet(str(data_dir / train_glob)).select(FEATURE_COLUMNS)
    labels = pl.scan_parquet(str(labels_path)).select(["event_id", "target"])

    labeled = labels.join(train_events, on="event_id", how="left")
    final_cols = ["event_id", "target"] + [c for c in FEATURE_COLUMNS if c != "event_id"]
    labeled = labeled.select(final_cols)
    if max_labeled_rows is not None:
        labeled = labeled.limit(max_labeled_rows)

    include_unlabeled = bool(cfg["dataset"]["include_unlabeled"])
    if not include_unlabeled:
        return with_time_and_amount_features(labeled).collect(streaming=True)

    modulo = int(cfg["dataset"]["unlabeled_modulo"])
    sampled_train = train_events.filter((pl.col("event_id") % modulo) == 0)
    unlabeled = (
        sampled_train.join(labels.select(["event_id"]), on="event_id", how="anti")
        .with_columns(pl.lit(-1).cast(pl.Int32).alias("target"))
        .select(final_cols)
    )
    if max_unlabeled_rows is not None:
        unlabeled = unlabeled.limit(max_unlabeled_rows)

    combined = pl.concat([labeled, unlabeled], how="vertical_relaxed")
    return with_time_and_amount_features(combined).collect(streaming=True)


def build_test_frame(cfg: dict, max_rows: int | None = None) -> pl.DataFrame:
    data_dir, _, _, test_path = _paths(cfg)
    test_lf = pl.scan_parquet(str(test_path)).select(FEATURE_COLUMNS)
    if max_rows is not None:
        test_lf = test_lf.limit(max_rows)
    return with_time_and_amount_features(test_lf).collect(streaming=True)


def build_train_week_frame_full(
    cfg: dict,
    week_start: datetime,
    max_rows: int | None = None,
) -> pl.DataFrame:
    data_dir, train_glob, labels_path, _ = _paths(cfg)

    train_events = pl.scan_parquet(str(data_dir / train_glob)).select(FEATURE_COLUMNS)
    labels = pl.scan_parquet(str(labels_path)).select(["event_id", "target"])
    final_cols = ["event_id", "target"] + [c for c in FEATURE_COLUMNS if c != "event_id"]

    full = (
        train_events.join(labels, on="event_id", how="left")
        .with_columns(pl.col("target").fill_null(-1).cast(pl.Int32))
        .select(final_cols)
    )
    full = with_time_and_amount_features(full).filter(pl.col("week_start") == pl.lit(week_start))
    if max_rows is not None:
        full = full.limit(max_rows)
    return full.collect(streaming=True)
