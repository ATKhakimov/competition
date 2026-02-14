#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import xgboost as xgb
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from competition.behavioral_features import load_or_build_pretrain_profile
from competition.datasets import build_test_frame, build_train_frame
from competition.pipeline_config import load_config
from competition.sequence_first_features import add_sequence_first_features
from competition.splits import rolling_week_folds


DROP_COLUMNS = {"target", "event_dttm", "event_dt", "week_start"}


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _to_jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


def _cuda_available() -> bool:
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True)
        return "GPU" in proc.stdout
    except Exception:
        return False


def _resolve_device(requested_device: str) -> tuple[str, bool]:
    has_cuda = _cuda_available()
    if requested_device == "auto":
        return ("cuda" if has_cuda else "cpu"), has_cuda
    if requested_device == "cuda":
        return "cuda", has_cuda
    return "cpu", has_cuda


def _lastday_mask(df: pd.DataFrame) -> pd.Series:
    event_day = pd.to_datetime(df["event_dt"], errors="coerce").dt.normalize()
    last_day_per_customer = event_day.groupby(df["customer_id"]).transform("max")
    return event_day.eq(last_day_per_customer) & event_day.notna()


def _average_precision_safe(y_true: pd.Series, pred: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    return float(average_precision_score(y_true, pred))


def _recall_at_k(y_true: pd.Series, pred: np.ndarray, k: int) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    k_eff = min(k, len(pred))
    top_idx = np.argsort(pred)[::-1][:k_eff]
    tp = int(y_true.iloc[top_idx].sum())
    return float(tp / positives)


def _fit_category_maps(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for c in cols:
        s = df[c].astype("string").fillna("<NULL>")
        uniq = pd.Index(s.unique())
        maps[c] = {str(v): int(i) for i, v in enumerate(uniq)}
    return maps


def _prepare_categories(df: pd.DataFrame, cols: list[str], maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = out[c].astype("string").fillna("<NULL>")
        out[c] = s.map(maps[c]).fillna(-1).astype(np.int32)
    return out


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    all_cols = [c for c in df.columns if c not in DROP_COLUMNS and c not in {"event_id"}]
    cat_cols: list[str] = []
    for c in all_cols:
        dt = df[c].dtype
        if str(dt).startswith(("object", "string", "category")):
            cat_cols.append(c)
    return all_cols, cat_cols


def _build_model(cfg: dict, device: str) -> XGBClassifier:
    params = cfg["model"]
    return XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device=device,
        enable_categorical=False,
        random_state=int(params["random_state"]),
        n_jobs=(-1 if device == "cpu" else 0),
    )


def _fit_with_fallback(
    cfg: dict,
    device: str,
    allow_fallback: bool,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
) -> tuple[XGBClassifier, str]:
    model = _build_model(cfg, device)
    try:
        model.fit(x_train, y_train, sample_weight=w_train, verbose=False)
        return model, device
    except xgb.core.XGBoostError as exc:
        if device == "cuda" and allow_fallback:
            print(f"[warn] CUDA training failed, fallback to CPU: {exc}")
            m = _build_model(cfg, "cpu")
            m.fit(x_train, y_train, sample_weight=w_train, verbose=False)
            return m, "cpu"
        raise


def _predict_scores(model: XGBClassifier, x: pd.DataFrame, runtime_device: str) -> np.ndarray:
    if runtime_device == "cuda":
        model.get_booster().set_param({"device": "cpu"})
    return model.predict(x, output_margin=True)


def _merge_profile(df: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(profile, on="customer_id", how="left")
    num_cols = [c for c in profile.columns if c != "customer_id"]
    for c in num_cols:
        out[c] = out[c].fillna(0.0)
    if "pre_profile_last_ts" in out.columns:
        out["gap_from_pretrain_last_sec"] = (
            out["event_ts"].astype(np.float64) - out["pre_profile_last_ts"].astype(np.float64)
        ).fillna(-1.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequence-first training without graph features.")
    parser.add_argument("--config", default="conf/sequence_first.yaml", type=str)
    parser.add_argument("--run-name", default=None, type=str)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--max-labeled-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--disable-pretrain-profile", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Sequence-first: train on labeled only by default.
    cfg["dataset"]["include_unlabeled"] = False

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or _run_id()
    runs_dir = Path(cfg["tracking"]["runs_dir"])
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    requested_device = args.device or str(cfg["model"].get("device", "auto")).lower()
    runtime_device, has_cuda = _resolve_device(requested_device)
    allow_fallback = bool(cfg["model"].get("gpu_fallback_to_cpu", True)) and requested_device == "auto"
    use_profile = not args.disable_pretrain_profile

    print(
        f"[device] requested={requested_device} resolved={runtime_device} "
        f"cuda_available={has_cuda}"
    )
    print(f"[setup] sequence-first labeled_only=True pretrain_profile={use_profile}")

    print("[1/4] Building labeled train frame...")
    train_pl = build_train_frame(cfg, max_labeled_rows=args.max_labeled_rows, max_unlabeled_rows=None)
    train_df = train_pl.to_pandas(use_pyarrow_extension_array=False)
    print(f"train rows={len(train_df):,}")

    train_df = add_sequence_first_features(train_df)
    if use_profile:
        profile = load_or_build_pretrain_profile(cfg)
        train_df = _merge_profile(train_df, profile)
    else:
        profile = None

    y = (train_df["target"] == 1).astype(np.int8)
    w = np.where(train_df["target"] == 1, 1.0, float(cfg["dataset"].get("reviewed_negative_weight", 1.0)))
    w = pd.Series(w, index=train_df.index)
    week_col = pd.to_datetime(train_df["week_start"])
    topk_list = [int(v) for v in cfg["validation"].get("topk", [100, 500, 1000])]

    feature_cols_raw, cat_cols = _feature_columns(train_df)
    cat_maps = _fit_category_maps(train_df[feature_cols_raw], cat_cols)
    train_df = _prepare_categories(train_df, cat_cols, cat_maps)
    feature_cols, _ = _feature_columns(train_df)

    print("[2/4] Temporal CV...")
    folds = rolling_week_folds(
        train_df.copy(),
        n_folds=int(cfg["validation"]["n_folds"]),
        min_train_weeks=int(cfg["validation"]["min_train_weeks"]),
    )
    cv_scores = []
    cv_proxy_scores = []
    cv_lastday_scores = []
    fold_rows = []

    for i, (val_week, _, valid_mask) in enumerate(folds, start=1):
        full_train_mask = week_col < val_week
        val_idx = train_df.index[valid_mask]

        x_train = train_df.loc[full_train_mask, feature_cols]
        y_train = y.loc[full_train_mask]
        w_train = w.loc[full_train_mask]
        x_valid = train_df.loc[val_idx, feature_cols]
        y_valid = y.loc[val_idx]

        model, used_device = _fit_with_fallback(
            cfg, runtime_device, allow_fallback, x_train, y_train, w_train
        )
        if used_device != runtime_device:
            runtime_device = used_device
            allow_fallback = False
            print(f"[device] switched to {runtime_device}")

        pred_valid = _predict_scores(model, x_valid, runtime_device)
        ap_labeled = _average_precision_safe(y_valid, pred_valid)
        cv_scores.append(ap_labeled)

        valid_week_mask = week_col == val_week
        proxy_part = train_df.loc[valid_week_mask]
        x_proxy = proxy_part[feature_cols]
        y_proxy = y.loc[valid_week_mask]
        pred_proxy = _predict_scores(model, x_proxy, runtime_device)
        ap_proxy = _average_precision_safe(y_proxy, pred_proxy)
        cv_proxy_scores.append(ap_proxy)

        lastday_mask = _lastday_mask(proxy_part)
        y_last = y_proxy.loc[lastday_mask]
        pred_last = pred_proxy[lastday_mask.to_numpy()]
        ap_last = _average_precision_safe(y_last, pred_last)
        cv_lastday_scores.append(ap_last)

        row = {
            "fold": i,
            "val_week": pd.Timestamp(val_week).date().isoformat(),
            "ap_labeled": float(ap_labeled),
            "ap_proxy_week": float(ap_proxy),
            "ap_proxy_lastday": float(ap_last),
            "train_rows": int(full_train_mask.sum()),
            "valid_rows": int(len(val_idx)),
            "valid_lastday_rows": int(lastday_mask.sum()),
        }
        for k in topk_list:
            row[f"recall_proxy_top{k}"] = _recall_at_k(y_proxy, pred_proxy, k)
            row[f"recall_lastday_top{k}"] = _recall_at_k(y_last, pred_last, k)
        fold_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics_live.csv", index=False)
        print(
            f"  fold={i} week={val_week.date()} AP_labeled={ap_labeled:.6f} "
            f"AP_proxy={ap_proxy:.6f} AP_lastday={ap_last:.6f}"
        )

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    cv_proxy_mean = float(np.mean(cv_proxy_scores))
    cv_proxy_std = float(np.std(cv_proxy_scores))
    cv_last_mean = float(np.mean(cv_lastday_scores))
    cv_last_std = float(np.std(cv_lastday_scores))
    print(f"CV AP_labeled mean={cv_mean:.6f} std={cv_std:.6f}")
    print(f"CV AP_proxy   mean={cv_proxy_mean:.6f} std={cv_proxy_std:.6f}")
    print(f"CV AP_lastday mean={cv_last_mean:.6f} std={cv_last_std:.6f}")

    print("[3/4] Building test frame...")
    test_pl = build_test_frame(cfg, max_rows=args.max_test_rows)
    test_df = test_pl.to_pandas(use_pyarrow_extension_array=False)
    test_df = add_sequence_first_features(test_df)
    if use_profile and profile is not None:
        test_df = _merge_profile(test_df, profile)
    test_df = _prepare_categories(test_df, cat_cols, cat_maps)

    model, used_device = _fit_with_fallback(
        cfg, runtime_device, allow_fallback, train_df[feature_cols], y, w
    )
    if used_device != runtime_device:
        runtime_device = used_device
        print(f"[device] final model switched to {runtime_device}")

    print("[4/4] Writing submission...")
    pred = _predict_scores(model, test_df[feature_cols], runtime_device)
    submission = pd.DataFrame({"event_id": test_df["event_id"], "predict": pred})
    submission_path = artifacts_dir / cfg["inference"]["submission_name"]
    submission.to_csv(submission_path, index=False)
    print(f"Submission written: {submission_path}")

    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False)
    fi = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    fi.sort_values("importance", ascending=False).to_csv(run_dir / "feature_importance.csv", index=False)

    summary = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "submission_path": str(submission_path),
        "config_path": args.config,
        "pipeline": "sequence_first_no_graph",
        "requested_device": requested_device,
        "runtime_device": runtime_device,
        "cuda_available": has_cuda,
        "use_pretrain_profile": use_profile,
        "train_rows": int(len(train_df)),
        "cv_ap_labeled_mean": cv_mean,
        "cv_ap_labeled_std": cv_std,
        "cv_ap_proxy_mean": cv_proxy_mean,
        "cv_ap_proxy_std": cv_proxy_std,
        "cv_ap_proxy_lastday_mean": cv_last_mean,
        "cv_ap_proxy_lastday_std": cv_last_std,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, ensure_ascii=False, indent=2)

    index_path = runs_dir / "runs_index.csv"
    row = pd.DataFrame(
        [
            {
                "run_name": run_name,
                "timestamp_utc": summary["timestamp_utc"],
                "pipeline": summary["pipeline"],
                "runtime_device": runtime_device,
                "train_rows": summary["train_rows"],
                "cv_ap_labeled_mean": cv_mean,
                "cv_ap_proxy_mean": cv_proxy_mean,
                "cv_ap_proxy_lastday_mean": cv_last_mean,
                "submission_path": str(submission_path),
                "run_dir": str(run_dir),
            }
        ]
    )
    if index_path.exists():
        prev = pd.read_csv(index_path)
        pd.concat([prev, row], ignore_index=True).to_csv(index_path, index=False)
    else:
        row.to_csv(index_path, index=False)
    print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()

