from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class RiskTable:
    prior: float
    token_risk: dict[str, dict[object, float]]
    token_count: dict[str, dict[object, float]]
    pair_risk: dict[tuple[str, str], dict[str, float]]
    pair_count: dict[tuple[str, str], dict[str, float]]


def _fit_token_table(
    df: pd.DataFrame,
    y: pd.Series,
    col: str,
    alpha: float,
    sample_weight: pd.Series | None,
    prior: float,
) -> tuple[dict[object, float], dict[object, float]]:
    tmp = pd.DataFrame({"key": df[col], "y": y})
    if sample_weight is None:
        tmp["w"] = 1.0
    else:
        tmp["w"] = sample_weight.values
    tmp["yw"] = tmp["y"] * tmp["w"]
    grouped = tmp.groupby("key", dropna=False, observed=False).agg(
        y_sum=("yw", "sum"), w_sum=("w", "sum")
    )
    risk = (grouped["y_sum"] + alpha * prior) / (grouped["w_sum"] + alpha)
    return risk.to_dict(), grouped["w_sum"].to_dict()


def _pair_key(df: pd.DataFrame, left: str, right: str) -> pd.Series:
    return df[left].astype("string").fillna("<NULL>") + "|" + df[right].astype("string").fillna(
        "<NULL>"
    )


def fit_risk_tables(
    labeled_df: pd.DataFrame,
    y: pd.Series,
    token_cols: Iterable[str],
    pair_cols: Iterable[tuple[str, str]],
    alpha: float = 20.0,
    sample_weight: pd.Series | None = None,
) -> RiskTable:
    prior = float(np.average(y, weights=sample_weight))
    token_risk: dict[str, dict[object, float]] = {}
    token_count: dict[str, dict[object, float]] = {}
    for col in token_cols:
        r, c = _fit_token_table(labeled_df, y, col, alpha, sample_weight, prior)
        token_risk[col] = r
        token_count[col] = c

    pair_risk: dict[tuple[str, str], dict[str, float]] = {}
    pair_count: dict[tuple[str, str], dict[str, float]] = {}
    for left, right in pair_cols:
        key = _pair_key(labeled_df, left, right)
        tmp = pd.DataFrame({"k": key, "y": y})
        if sample_weight is None:
            tmp["w"] = 1.0
        else:
            tmp["w"] = sample_weight.values
        tmp["yw"] = tmp["y"] * tmp["w"]
        grouped = tmp.groupby("k", observed=False).agg(y_sum=("yw", "sum"), w_sum=("w", "sum"))
        risk = (grouped["y_sum"] + alpha * prior) / (grouped["w_sum"] + alpha)
        pair_risk[(left, right)] = risk.to_dict()
        pair_count[(left, right)] = grouped["w_sum"].to_dict()

    return RiskTable(
        prior=prior,
        token_risk=token_risk,
        token_count=token_count,
        pair_risk=pair_risk,
        pair_count=pair_count,
    )


def apply_risk_tables(
    df: pd.DataFrame,
    tables: RiskTable,
    token_cols: Iterable[str],
    pair_cols: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    out = df.copy()
    for col in token_cols:
        mapped = out[col].astype("object").map(tables.token_risk[col])
        out[f"risk_{col}"] = mapped.fillna(tables.prior).astype(float)
        out[f"log_cnt_{col}"] = (
            np.log1p(out[col].astype("object").map(tables.token_count[col]).fillna(0.0).astype(float))
        )

    for left, right in pair_cols:
        key = _pair_key(out, left, right)
        out[f"risk_{left}_{right}"] = key.map(tables.pair_risk[(left, right)]).fillna(
            tables.prior
        )
        out[f"log_cnt_{left}_{right}"] = np.log1p(
            key.map(tables.pair_count[(left, right)]).fillna(0.0).astype(float)
        )
    return out
