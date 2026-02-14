from __future__ import annotations

from typing import Iterator

import pandas as pd


def rolling_week_folds(
    df: pd.DataFrame, n_folds: int, min_train_weeks: int
) -> Iterator[tuple[pd.Timestamp, pd.Series, pd.Series]]:
    weeks = sorted(pd.to_datetime(df["week_start"]).dropna().unique())
    if len(weeks) < (n_folds + min_train_weeks):
        raise ValueError(
            f"Not enough weekly buckets for CV. Have={len(weeks)}, "
            f"required>={n_folds + min_train_weeks}."
        )

    val_weeks = weeks[-n_folds:]
    week_col = pd.to_datetime(df["week_start"])
    for val_week in val_weeks:
        train_mask = week_col < val_week
        valid_mask = week_col == val_week
        yield pd.Timestamp(val_week), train_mask, valid_mask

