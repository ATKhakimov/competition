#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import xgboost as xgb
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from competition.datasets import build_test_frame, build_train_frame
from competition.behavioral_features import add_sequence_features, load_or_build_pretrain_profile
from competition.graph_risk import apply_risk_tables, fit_risk_tables
from competition.pipeline_config import load_config
from competition.splits import rolling_week_folds


DROP_COLUMNS = {
    "target",
    "event_dttm",
    "event_dt",
    "week_start",
}

GRAPH_TOKEN_COLS = [
    "mcc_code",
    "event_type_nm",
    "channel_indicator_sub_type",
    "channel_indicator_type",
    "timezone",
    "operating_system_type",
    "device_system_version",
]
GRAPH_PAIR_COLS = [
    ("customer_id", "mcc_code"),
    ("customer_id", "channel_indicator_sub_type"),
    ("customer_id", "event_type_nm"),
]


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


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _recall_at_k(y_true: pd.Series, pred: np.ndarray, k: int) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    k_eff = min(k, len(pred))
    top_idx = np.argsort(pred)[::-1][:k_eff]
    tp = int(y_true.iloc[top_idx].sum())
    return float(tp / positives)


def _average_precision_safe(y_true: pd.Series, pred: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    return float(average_precision_score(y_true, pred))


def _lastday_mask(df: pd.DataFrame) -> pd.Series:
    event_day = pd.to_datetime(df["event_dt"], errors="coerce").dt.normalize()
    last_day_per_customer = event_day.groupby(df["customer_id"]).transform("max")
    mask = event_day.eq(last_day_per_customer) & event_day.notna()
    return mask


PROFILE_NUM_COLS = [
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


def _merge_pretrain_profile(df: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(profile, on="customer_id", how="left")
    for col in PROFILE_NUM_COLS:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)
    if "pre_profile_last_ts" in out.columns:
        out["gap_from_pretrain_last_sec"] = (
            out["event_ts"].astype(np.float64) - out["pre_profile_last_ts"].astype(np.float64)
        ).fillna(-1.0)
    return out


def _build_test_sequence_features(cfg: dict, test_df: pd.DataFrame) -> pd.DataFrame:
    _ = cfg
    return add_sequence_features(test_df)


SEQ_COLS = [
    "seq_customer_idx",
    "dt_since_prev_sec",
    "rolling_count_1h",
    "rolling_count_1d",
    "rolling_count_7d",
    "burst_1h_over_1d",
    "amt_over_prev_mean",
    "is_new_event_type",
    "is_new_channel_sub_type",
    "is_new_mcc",
]


def _cuda_available() -> bool:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        )
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


def _model_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_cols = [
        c for c in df.columns if c not in DROP_COLUMNS and c not in {"event_id"}
    ]
    categorical = [
        "event_type_nm",
        "event_desc",
        "channel_indicator_type",
        "channel_indicator_sub_type",
        "currency_iso_cd",
        "mcc_code",
        "pos_cd",
        "accept_language",
        "browser_language",
        "timezone",
        "session_id",
        "operating_system_type",
        "battery",
        "device_system_version",
        "screen_size",
        "developer_tools",
        "phone_voip_call_state",
        "web_rdp_connection",
        "compromised",
    ]
    cat_cols = [c for c in categorical if c in feature_cols]
    return feature_cols, cat_cols


def _fit_category_maps(df: pd.DataFrame, cat_cols: list[str]) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for col in cat_cols:
        s = df[col].astype("string").fillna("<NULL>")
        uniq = pd.Index(s.unique())
        maps[col] = {str(v): int(i) for i, v in enumerate(uniq)}
    return maps


def _prepare_pandas(
    df: pd.DataFrame, cat_cols: list[str], cat_maps: dict[str, dict[str, int]]
) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        s = out[col].astype("string").fillna("<NULL>")
        mapped = s.map(cat_maps[col])
        out[col] = mapped.fillna(-1).astype(np.int32)
    for col in out.columns:
        if col in cat_cols:
            continue
        if out[col].dtype == "bool":
            out[col] = out[col].astype(np.int8)
    return out


def _target_and_weight(
    df: pd.DataFrame, reviewed_negative_weight: float, unlabeled_weight: float
) -> tuple[pd.Series, pd.Series]:
    y = (df["target"] == 1).astype(np.int8)
    w = np.where(df["target"] == 1, 1.0, reviewed_negative_weight)
    w = np.where(df["target"] == -1, unlabeled_weight, w)
    return pd.Series(y, index=df.index), pd.Series(w, index=df.index)


def _build_model(cfg: dict, device: str) -> XGBClassifier:
    params = cfg["model"]
    model = XGBClassifier(
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
    return model


def _fit_with_fallback(
    cfg: dict,
    device: str,
    allow_fallback: bool,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
) -> tuple[XGBClassifier, str]:
    model = _build_model(cfg, device=device)
    try:
        model.fit(x_train, y_train, sample_weight=w_train, verbose=False)
        return model, device
    except xgb.core.XGBoostError as exc:
        if device == "cuda" and allow_fallback:
            print(f"[warn] CUDA training failed, fallback to CPU: {exc}")
            cpu_model = _build_model(cfg, device="cpu")
            cpu_model.fit(x_train, y_train, sample_weight=w_train, verbose=False)
            return cpu_model, "cpu"
        raise


def _predict_proba(
    model: XGBClassifier, x: pd.DataFrame, runtime_device: str
) -> np.ndarray:
    if runtime_device == "cuda":
        model.get_booster().set_param({"device": "cpu"})
    return model.predict_proba(x)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal baseline and create submission.")
    parser.add_argument("--config", default="conf/pipeline.yaml", type=str)
    parser.add_argument("--run-name", default=None, type=str)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--max-labeled-rows", type=int, default=None)
    parser.add_argument("--max-unlabeled-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--disable-graph-risk", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or _run_id()
    runs_dir = Path(cfg["tracking"]["runs_dir"])
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_device = args.device or str(cfg["model"].get("device", "auto")).lower()
    runtime_device, has_cuda = _resolve_device(requested_device)
    allow_fallback = bool(cfg["model"].get("gpu_fallback_to_cpu", True)) and requested_device == "auto"
    print(
        f"[device] requested={requested_device} resolved={runtime_device} "
        f"cuda_available={has_cuda}"
    )

    print("[1/4] Building train frame...")
    train_pl = build_train_frame(
        cfg,
        max_labeled_rows=args.max_labeled_rows,
        max_unlabeled_rows=args.max_unlabeled_rows,
    )
    train_df = train_pl.to_pandas(use_pyarrow_extension_array=False)
    print(f"train rows={len(train_df):,}")
    train_df = add_sequence_features(train_df)
    pretrain_profile = load_or_build_pretrain_profile(cfg)
    train_df = _merge_pretrain_profile(train_df, pretrain_profile)

    _, cat_cols = _model_features(train_df)
    cat_maps = _fit_category_maps(train_df, cat_cols)
    train_df = _prepare_pandas(train_df, cat_cols, cat_maps)
    y, w = _target_and_weight(
        train_df,
        reviewed_negative_weight=float(cfg["dataset"]["reviewed_negative_weight"]),
        unlabeled_weight=float(cfg["dataset"]["unlabeled_weight"]),
    )
    is_labeled = train_df["target"] >= 0
    week_col = pd.to_datetime(train_df["week_start"])
    topk_list = [int(v) for v in cfg["validation"].get("topk", [100, 500, 1000])]

    print("[2/4] Temporal CV on labeled validation weeks...")
    folds = rolling_week_folds(
        train_df.loc[is_labeled].copy(),
        n_folds=int(cfg["validation"]["n_folds"]),
        min_train_weeks=int(cfg["validation"]["min_train_weeks"]),
    )
    cv_scores = []
    cv_proxy_scores = []
    cv_proxy_lastday_scores = []
    fold_rows = []
    oof_frames = []
    oof_lastday_frames = []
    for i, (val_week, _, valid_mask_labeled) in enumerate(folds, start=1):
        full_train_mask = week_col < val_week
        labeled_slice = train_df.loc[is_labeled].copy()
        val_idx = labeled_slice.index[valid_mask_labeled]

        train_part = train_df.loc[full_train_mask].copy()
        valid_part = train_df.loc[val_idx].copy()

        if not args.disable_graph_risk:
            labeled_in_train = train_part.loc[train_part["target"] >= 0]
            y_labeled_in_train = (labeled_in_train["target"] == 1).astype(np.int8)
            w_labeled_in_train = w.loc[labeled_in_train.index]
            risk_tables = fit_risk_tables(
                labeled_in_train,
                y_labeled_in_train,
                token_cols=GRAPH_TOKEN_COLS,
                pair_cols=GRAPH_PAIR_COLS,
                alpha=20.0,
                sample_weight=w_labeled_in_train,
            )
            train_part = apply_risk_tables(
                train_part,
                risk_tables,
                token_cols=GRAPH_TOKEN_COLS,
                pair_cols=GRAPH_PAIR_COLS,
            )
            valid_part = apply_risk_tables(
                valid_part,
                risk_tables,
                token_cols=GRAPH_TOKEN_COLS,
                pair_cols=GRAPH_PAIR_COLS,
            )

        fold_feature_cols, _ = _model_features(train_part)
        x_train = train_part[fold_feature_cols]
        y_train = y.loc[full_train_mask]
        w_train = w.loc[full_train_mask]
        x_valid = valid_part[fold_feature_cols]
        y_valid = y.loc[val_idx]

        model, used_device = _fit_with_fallback(
            cfg=cfg,
            device=runtime_device,
            allow_fallback=allow_fallback,
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
        )
        if used_device != runtime_device:
            runtime_device = used_device
            allow_fallback = False
            print(f"[device] switched to {runtime_device} for remaining stages")
        pred_valid = _predict_proba(model, x_valid, runtime_device=runtime_device)
        ap_labeled = _average_precision_safe(y_valid, pred_valid)
        cv_scores.append(ap_labeled)

        valid_week_mask = week_col == val_week
        proxy_part = train_df.loc[valid_week_mask].copy()
        if not args.disable_graph_risk:
            proxy_part = apply_risk_tables(
                proxy_part,
                risk_tables,
                token_cols=GRAPH_TOKEN_COLS,
                pair_cols=GRAPH_PAIR_COLS,
            )
        x_proxy = proxy_part[fold_feature_cols]
        y_proxy = y.loc[valid_week_mask]
        pred_proxy = _predict_proba(model, x_proxy, runtime_device=runtime_device)
        ap_proxy = _average_precision_safe(y_proxy, pred_proxy)
        cv_proxy_scores.append(ap_proxy)

        lastday_mask = _lastday_mask(proxy_part)
        x_proxy_lastday = x_proxy.loc[lastday_mask]
        y_proxy_lastday = y_proxy.loc[lastday_mask]
        pred_proxy_lastday = pred_proxy[lastday_mask.to_numpy()]
        ap_proxy_lastday = _average_precision_safe(y_proxy_lastday, pred_proxy_lastday)
        cv_proxy_lastday_scores.append(ap_proxy_lastday)

        row = {
            "fold": i,
            "val_week": pd.Timestamp(val_week).date().isoformat(),
            "ap_labeled": float(ap_labeled),
            "ap_proxy_week": float(ap_proxy),
            "ap_proxy_lastday": float(ap_proxy_lastday),
            "train_rows": int(full_train_mask.sum()),
            "valid_labeled_rows": int(len(val_idx)),
            "valid_proxy_rows": int(valid_week_mask.sum()),
            "valid_proxy_lastday_rows": int(lastday_mask.sum()),
        }
        for k in topk_list:
            row[f"recall_proxy_top{k}"] = _recall_at_k(y_proxy, pred_proxy, k)
            row[f"recall_proxy_lastday_top{k}"] = _recall_at_k(
                y_proxy_lastday, pred_proxy_lastday, k
            )
        fold_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics_live.csv", index=False)
        print(
            f"  fold={i} week={val_week.date()} AP_labeled={ap_labeled:.6f} "
            f"AP_proxy={ap_proxy:.6f} AP_lastday={ap_proxy_lastday:.6f}"
        )

        if bool(cfg["tracking"].get("save_oof", True)):
            fold_oof = pd.DataFrame(
                {
                    "event_id": proxy_part["event_id"].values,
                    "fold": i,
                    "val_week": pd.Timestamp(val_week).date().isoformat(),
                    "target_proxy": y_proxy.values,
                    "predict": pred_proxy,
                }
            )
            oof_frames.append(fold_oof)
            oof_lastday_frames.append(
                pd.DataFrame(
                    {
                        "event_id": proxy_part.loc[lastday_mask, "event_id"].values,
                        "fold": i,
                        "val_week": pd.Timestamp(val_week).date().isoformat(),
                        "target_proxy": y_proxy_lastday.values,
                        "predict": pred_proxy_lastday,
                    }
                )
            )

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    cv_proxy_mean = float(np.mean(cv_proxy_scores))
    cv_proxy_std = float(np.std(cv_proxy_scores))
    cv_proxy_lastday_mean = float(np.mean(cv_proxy_lastday_scores))
    cv_proxy_lastday_std = float(np.std(cv_proxy_lastday_scores))
    print(f"CV AP_labeled mean={cv_mean:.6f} std={cv_std:.6f}")
    print(f"CV AP_proxy   mean={cv_proxy_mean:.6f} std={cv_proxy_std:.6f}")
    print(
        f"CV AP_lastday mean={cv_proxy_lastday_mean:.6f} "
        f"std={cv_proxy_lastday_std:.6f}"
    )

    print("[3/4] Building final train/test matrices...")
    test_pl = build_test_frame(cfg, max_rows=args.max_test_rows)
    test_df = test_pl.to_pandas(use_pyarrow_extension_array=False)
    test_df = _build_test_sequence_features(cfg, test_df)
    test_df = _merge_pretrain_profile(test_df, pretrain_profile)
    test_df = _prepare_pandas(test_df, cat_cols, cat_maps)

    if not args.disable_graph_risk:
        labeled_full = train_df.loc[is_labeled]
        y_labeled_full = (labeled_full["target"] == 1).astype(np.int8)
        w_labeled_full = w.loc[labeled_full.index]
        final_risk_tables = fit_risk_tables(
            labeled_full,
            y_labeled_full,
            token_cols=GRAPH_TOKEN_COLS,
            pair_cols=GRAPH_PAIR_COLS,
            alpha=20.0,
            sample_weight=w_labeled_full,
        )
        train_df = apply_risk_tables(
            train_df, final_risk_tables, token_cols=GRAPH_TOKEN_COLS, pair_cols=GRAPH_PAIR_COLS
        )
        test_df = apply_risk_tables(
            test_df, final_risk_tables, token_cols=GRAPH_TOKEN_COLS, pair_cols=GRAPH_PAIR_COLS
        )

    feature_cols, _ = _model_features(train_df)
    final_model, used_device = _fit_with_fallback(
        cfg=cfg,
        device=runtime_device,
        allow_fallback=allow_fallback,
        x_train=train_df[feature_cols],
        y_train=y,
        w_train=w,
    )
    if used_device != runtime_device:
        runtime_device = used_device
        print(f"[device] final model switched to {runtime_device}")

    print("[4/4] Writing submission...")
    pred = _predict_proba(final_model, test_df[feature_cols], runtime_device=runtime_device)
    submission = pd.DataFrame({"event_id": test_df["event_id"], "predict": pred})

    submission_path = artifacts_dir / cfg["inference"]["submission_name"]
    submission.to_csv(submission_path, index=False)
    print(f"Submission written: {submission_path}")

    fold_df = pd.DataFrame(fold_rows)
    fold_path = run_dir / "fold_metrics.csv"
    fold_df.to_csv(fold_path, index=False)

    summary = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "submission_path": str(submission_path),
        "config_path": args.config,
        "disable_graph_risk": bool(args.disable_graph_risk),
        "requested_device": requested_device,
        "runtime_device": runtime_device,
        "cuda_available": has_cuda,
        "train_rows": int(len(train_df)),
        "labeled_rows": int(is_labeled.sum()),
        "unlabeled_rows": int((~is_labeled).sum()),
        "cv_ap_labeled_mean": cv_mean,
        "cv_ap_labeled_std": cv_std,
        "cv_ap_proxy_mean": cv_proxy_mean,
        "cv_ap_proxy_std": cv_proxy_std,
        "cv_ap_proxy_lastday_mean": cv_proxy_lastday_mean,
        "cv_ap_proxy_lastday_std": cv_proxy_lastday_std,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, ensure_ascii=False, indent=2)

    if bool(cfg["tracking"].get("save_oof", True)) and oof_frames:
        pd.concat(oof_frames, axis=0, ignore_index=True).to_csv(
            run_dir / "oof_proxy_predictions.csv", index=False
        )
    if bool(cfg["tracking"].get("save_oof", True)) and oof_lastday_frames:
        pd.concat(oof_lastday_frames, axis=0, ignore_index=True).to_csv(
            run_dir / "oof_proxy_lastday_predictions.csv", index=False
        )

    if bool(cfg["tracking"].get("save_feature_importance", True)):
        fi = pd.DataFrame(
            {"feature": feature_cols, "importance": final_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        fi.to_csv(run_dir / "feature_importance.csv", index=False)

    index_path = runs_dir / "runs_index.csv"
    index_row = pd.DataFrame(
        [
            {
                "run_name": run_name,
                "timestamp_utc": summary["timestamp_utc"],
                "disable_graph_risk": bool(args.disable_graph_risk),
                "runtime_device": runtime_device,
                "train_rows": summary["train_rows"],
                "labeled_rows": summary["labeled_rows"],
                "unlabeled_rows": summary["unlabeled_rows"],
                "cv_ap_labeled_mean": cv_mean,
                "cv_ap_proxy_mean": cv_proxy_mean,
                "cv_ap_proxy_lastday_mean": cv_proxy_lastday_mean,
                "submission_path": str(submission_path),
                "run_dir": str(run_dir),
            }
        ]
    )
    if index_path.exists():
        prev = pd.read_csv(index_path)
        pd.concat([prev, index_row], axis=0, ignore_index=True).to_csv(index_path, index=False)
    else:
        index_row.to_csv(index_path, index=False)

    print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
