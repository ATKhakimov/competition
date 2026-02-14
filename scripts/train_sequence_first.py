#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, CatBoostError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from competition.behavioral_features import load_or_build_pretrain_profile
from competition.datasets import build_test_frame, build_train_frame, build_train_week_frame_full
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


def _resolve_bool_mode(mode: str, cfg_val: bool) -> bool:
    if mode == "config":
        return cfg_val
    if mode == "true":
        return True
    if mode == "false":
        return False
    raise ValueError(f"Unsupported mode={mode}")


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
        s = df[c]
        dt = s.dtype
        if (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(dt, pd.CategoricalDtype)
        ):
            cat_cols.append(c)
    return all_cols, cat_cols


def _build_model(
    cfg: dict,
    device: str,
    model_family: str,
) -> XGBClassifier | CatBoostClassifier:
    params = cfg["model"]
    if model_family == "xgb":
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
    if model_family == "catboost":
        cb_kwargs = dict(
            iterations=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            depth=int(params["max_depth"]),
            l2_leaf_reg=float(params["reg_lambda"]),
            min_data_in_leaf=max(1, int(float(params["min_child_weight"]))),
            bootstrap_type="Bernoulli",
            subsample=float(params["subsample"]),
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=int(params["random_state"]),
            task_type=("GPU" if device == "cuda" else "CPU"),
            allow_writing_files=False,
            verbose=False,
        )
        if device == "cpu":
            cb_kwargs["rsm"] = float(params["colsample_bytree"])
        return CatBoostClassifier(**cb_kwargs)
    raise ValueError(f"Unsupported model_family={model_family}")


def _fit_with_fallback(
    cfg: dict,
    device: str,
    allow_fallback: bool,
    model_family: str,
    cat_feature_indices: list[int],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
) -> tuple[XGBClassifier | CatBoostClassifier, str]:
    model = _build_model(cfg, device, model_family)
    try:
        if model_family == "catboost":
            model.fit(
                x_train,
                y_train,
                sample_weight=w_train,
                cat_features=cat_feature_indices,
                verbose=False,
            )
        else:
            model.fit(x_train, y_train, sample_weight=w_train, verbose=False)
        return model, device
    except (xgb.core.XGBoostError, CatBoostError) as exc:
        if device == "cuda" and allow_fallback:
            print(f"[warn] CUDA training failed, fallback to CPU: {exc}")
            m = _build_model(cfg, "cpu", model_family)
            if model_family == "catboost":
                m.fit(
                    x_train,
                    y_train,
                    sample_weight=w_train,
                    cat_features=cat_feature_indices,
                    verbose=False,
                )
            else:
                m.fit(x_train, y_train, sample_weight=w_train, verbose=False)
            return m, "cpu"
        raise


def _predict_scores(
    model: XGBClassifier | CatBoostClassifier,
    x: pd.DataFrame,
    runtime_device: str,
    model_family: str,
) -> np.ndarray:
    if model_family == "xgb" and runtime_device == "cuda":
        model.get_booster().set_param({"device": "cpu"})
    if model_family == "xgb":
        return model.predict(x, output_margin=True)
    return np.asarray(model.predict(x, prediction_type="RawFormulaVal"), dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


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
    parser.add_argument("--model-family", choices=["xgb", "catboost"], default="xgb")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--use-unlabeled", choices=["config", "true", "false"], default="config")
    parser.add_argument("--unlabeled-weight", type=float, default=None)
    parser.add_argument("--max-labeled-rows", type=int, default=None)
    parser.add_argument("--max-unlabeled-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--max-eval-week-rows", type=int, default=None)
    parser.add_argument("--history-window-events", choices=["all", "5", "20", "100"], default="all")
    parser.add_argument("--disable-ultra-burst", action="store_true")
    parser.add_argument("--disable-full-week-eval", action="store_true")
    parser.add_argument("--disable-pretrain-profile", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_family = str(args.model_family).lower()
    include_unlabeled = _resolve_bool_mode(
        args.use_unlabeled, bool(cfg["dataset"].get("include_unlabeled", False))
    )
    cfg["dataset"]["include_unlabeled"] = include_unlabeled
    if args.unlabeled_weight is not None:
        cfg["dataset"]["unlabeled_weight"] = float(args.unlabeled_weight)

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or _run_id()
    runs_dir = Path(cfg["tracking"]["runs_dir"])
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir = artifacts_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    requested_device = args.device or str(cfg["model"].get("device", "auto")).lower()
    runtime_device, has_cuda = _resolve_device(requested_device)
    allow_fallback = bool(cfg["model"].get("gpu_fallback_to_cpu", True)) and requested_device == "auto"
    use_profile = not args.disable_pretrain_profile
    max_prev_events = None if args.history_window_events == "all" else int(args.history_window_events)
    include_ultra_burst = not args.disable_ultra_burst
    use_full_week_eval = not args.disable_full_week_eval

    print(
        f"[device] requested={requested_device} resolved={runtime_device} "
        f"cuda_available={has_cuda}"
    )
    print(
        "[setup] sequence-first "
        f"model_family={model_family} "
        f"include_unlabeled={include_unlabeled} "
        f"pretrain_profile={use_profile} "
        f"history_window_events={args.history_window_events} "
        f"ultra_burst={include_ultra_burst} "
        f"full_week_eval={use_full_week_eval}"
    )

    print("[1/4] Building train frame...")
    train_pl = build_train_frame(
        cfg,
        max_labeled_rows=args.max_labeled_rows,
        max_unlabeled_rows=args.max_unlabeled_rows,
    )
    train_base_df = train_pl.to_pandas(use_pyarrow_extension_array=False)
    print(f"train rows={len(train_base_df):,}")

    train_df = add_sequence_first_features(
        train_base_df.copy(),
        max_prev_events=max_prev_events,
        include_ultra_burst=include_ultra_burst,
    )
    if use_profile:
        profile = load_or_build_pretrain_profile(cfg)
        train_df = _merge_profile(train_df, profile)
    else:
        profile = None

    y = (train_df["target"] == 1).astype(np.int8)
    w = np.where(train_df["target"] == 1, 1.0, float(cfg["dataset"].get("reviewed_negative_weight", 1.0)))
    w = np.where(train_df["target"] == -1, float(cfg["dataset"].get("unlabeled_weight", 0.1)), w)
    w = pd.Series(w, index=train_df.index)
    week_col = pd.to_datetime(train_df["week_start"])
    week_col_base = pd.to_datetime(train_base_df["week_start"])
    topk_list = [int(v) for v in cfg["validation"].get("topk", [100, 500, 1000])]
    is_labeled = train_df["target"] >= 0

    feature_cols_raw, cat_cols = _feature_columns(train_df)
    cat_maps = _fit_category_maps(train_df[feature_cols_raw], cat_cols)
    train_df = _prepare_categories(train_df, cat_cols, cat_maps)
    feature_cols, _ = _feature_columns(train_df)
    cat_col_set = set(cat_cols)
    cat_feature_indices = [i for i, c in enumerate(feature_cols) if c in cat_col_set]

    print("[2/4] Temporal CV...")
    folds = rolling_week_folds(
        train_df.loc[is_labeled].copy(),
        n_folds=int(cfg["validation"]["n_folds"]),
        min_train_weeks=int(cfg["validation"]["min_train_weeks"]),
    )
    cv_scores = []
    cv_all_events_scores = []
    cv_all_events_sampled_scores = []
    cv_lastday_scores = []
    cv_seen_scores = []
    cv_cold_scores = []
    cv_primary_pos_rate_scores = []
    fold_rows = []

    for i, (val_week, _, valid_mask_labeled) in enumerate(folds, start=1):
        full_train_mask = week_col < val_week
        x_train = train_df.loc[full_train_mask, feature_cols]
        y_train = y.loc[full_train_mask]
        w_train = w.loc[full_train_mask]
        labeled_slice = train_df.loc[is_labeled].copy()
        val_idx = labeled_slice.index[valid_mask_labeled]
        x_valid = train_df.loc[val_idx, feature_cols]
        y_valid = y.loc[val_idx]

        model, used_device = _fit_with_fallback(
            cfg,
            runtime_device,
            allow_fallback,
            model_family,
            cat_feature_indices,
            x_train,
            y_train,
            w_train,
        )
        if used_device != runtime_device:
            runtime_device = used_device
            allow_fallback = False
            print(f"[device] switched to {runtime_device}")

        pred_valid = _predict_scores(model, x_valid, runtime_device, model_family)
        ap_labeled = _average_precision_safe(y_valid, pred_valid)
        cv_scores.append(ap_labeled)

        valid_week_mask = week_col == val_week
        valid_week_part = train_df.loc[valid_week_mask]
        x_valid_week = valid_week_part[feature_cols]
        y_valid_week = y.loc[valid_week_mask]
        pred_valid_week = _predict_scores(model, x_valid_week, runtime_device, model_family)
        ap_all_events_sampled = _average_precision_safe(y_valid_week, pred_valid_week)
        cv_all_events_sampled_scores.append(ap_all_events_sampled)

        eval_mode = "sampled_week"
        valid_week_primary = valid_week_part
        y_primary = y_valid_week
        pred_primary = pred_valid_week
        if use_full_week_eval:
            full_week_pl = build_train_week_frame_full(
                cfg,
                pd.Timestamp(val_week).to_pydatetime(),
                max_rows=args.max_eval_week_rows,
            )
            full_week_base = full_week_pl.to_pandas(use_pyarrow_extension_array=False)
            history_base = train_base_df.loc[week_col_base < val_week].copy()
            eval_base = pd.concat([history_base, full_week_base], axis=0, ignore_index=True)
            eval_feat = add_sequence_first_features(
                eval_base,
                max_prev_events=max_prev_events,
                include_ultra_burst=include_ultra_burst,
            )
            if use_profile and profile is not None:
                eval_feat = _merge_profile(eval_feat, profile)
            eval_week_mask = pd.to_datetime(eval_feat["week_start"]) == pd.Timestamp(val_week)
            valid_week_primary = eval_feat.loc[eval_week_mask].copy()
            valid_week_primary = _prepare_categories(valid_week_primary, cat_cols, cat_maps)
            y_primary = (valid_week_primary["target"] == 1).astype(np.int8)
            pred_primary = _predict_scores(
                model, valid_week_primary[feature_cols], runtime_device, model_family
            )
            eval_mode = "full_week_unsampled"

        ap_all_events = _average_precision_safe(y_primary, pred_primary)
        cv_all_events_scores.append(ap_all_events)
        cv_primary_pos_rate_scores.append(float(y_primary.mean()))

        lastday_mask = _lastday_mask(valid_week_primary)
        y_last = y_primary.loc[lastday_mask]
        pred_last = pred_primary[lastday_mask.to_numpy()]
        ap_last = _average_precision_safe(y_last, pred_last)
        cv_lastday_scores.append(ap_last)

        seen_customers = set(train_base_df.loc[week_col_base < val_week, "customer_id"].unique())
        seen_mask = valid_week_primary["customer_id"].isin(seen_customers)
        cold_mask = ~seen_mask
        y_seen = y_primary.loc[seen_mask]
        y_cold = y_primary.loc[cold_mask]
        pred_seen = pred_primary[seen_mask.to_numpy()]
        pred_cold = pred_primary[cold_mask.to_numpy()]
        ap_seen = _average_precision_safe(y_seen, pred_seen)
        ap_cold = _average_precision_safe(y_cold, pred_cold)
        cv_seen_scores.append(ap_seen)
        cv_cold_scores.append(ap_cold)

        row = {
            "fold": i,
            "val_week": pd.Timestamp(val_week).date().isoformat(),
            "ap_all_events": float(ap_all_events),
            "ap_all_events_sampled": float(ap_all_events_sampled),
            "ap_labeled": float(ap_labeled),
            "ap_proxy_lastday": float(ap_last),
            "ap_seen": float(ap_seen),
            "ap_cold": float(ap_cold),
            "eval_all_events_mode": eval_mode,
            "train_rows": int(full_train_mask.sum()),
            "valid_rows": int(len(val_idx)),
            "valid_week_rows_sampled": int(valid_week_mask.sum()),
            "valid_week_rows_primary": int(len(valid_week_primary)),
            "valid_lastday_rows": int(lastday_mask.sum()),
            "valid_seen_rows": int(seen_mask.sum()),
            "valid_cold_rows": int(cold_mask.sum()),
            "valid_week_positives_primary": int(y_primary.sum()),
            "valid_week_pos_rate_primary": float(y_primary.mean()),
            "valid_seen_positives": int(y_seen.sum()),
            "valid_cold_positives": int(y_cold.sum()),
        }
        for k in topk_list:
            row[f"recall_all_events_top{k}"] = _recall_at_k(y_primary, pred_primary, k)
            row[f"recall_lastday_top{k}"] = _recall_at_k(y_last, pred_last, k)
        fold_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics_live.csv", index=False)
        print(
            f"  fold={i} week={val_week.date()} AP_all_events={ap_all_events:.6f} "
            f"(sampled={ap_all_events_sampled:.6f}) "
            f"AP_labeled={ap_labeled:.6f} AP_lastday={ap_last:.6f} "
            f"AP_seen={ap_seen:.6f} AP_cold={ap_cold:.6f}"
        )

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    cv_all_events_mean = float(np.mean(cv_all_events_scores))
    cv_all_events_std = float(np.std(cv_all_events_scores))
    cv_all_events_sampled_mean = float(np.mean(cv_all_events_sampled_scores))
    cv_all_events_sampled_std = float(np.std(cv_all_events_sampled_scores))
    cv_last_mean = float(np.mean(cv_lastday_scores))
    cv_last_std = float(np.std(cv_lastday_scores))
    cv_seen_mean = float(np.mean(cv_seen_scores))
    cv_seen_std = float(np.std(cv_seen_scores))
    cv_cold_mean = float(np.mean(cv_cold_scores))
    cv_cold_std = float(np.std(cv_cold_scores))
    cv_primary_pos_rate_mean = float(np.mean(cv_primary_pos_rate_scores))
    cv_primary_pos_rate_std = float(np.std(cv_primary_pos_rate_scores))
    print(f"CV AP_labeled mean={cv_mean:.6f} std={cv_std:.6f}")
    print(f"CV AP_all_events mean={cv_all_events_mean:.6f} std={cv_all_events_std:.6f}")
    print(
        f"CV AP_all_events_sampled mean={cv_all_events_sampled_mean:.6f} "
        f"std={cv_all_events_sampled_std:.6f}"
    )
    print(f"CV AP_lastday mean={cv_last_mean:.6f} std={cv_last_std:.6f}")
    print(f"CV AP_seen    mean={cv_seen_mean:.6f} std={cv_seen_std:.6f}")
    print(f"CV AP_cold    mean={cv_cold_mean:.6f} std={cv_cold_std:.6f}")
    print(
        f"CV valid_pos_rate_primary mean={cv_primary_pos_rate_mean:.6f} "
        f"std={cv_primary_pos_rate_std:.6f}"
    )

    print("[3/4] Building test frame...")
    test_pl = build_test_frame(cfg, max_rows=args.max_test_rows)
    test_df = test_pl.to_pandas(use_pyarrow_extension_array=False)
    test_df = add_sequence_first_features(
        test_df,
        max_prev_events=max_prev_events,
        include_ultra_burst=include_ultra_burst,
    )
    if use_profile and profile is not None:
        test_df = _merge_profile(test_df, profile)
    test_df = _prepare_categories(test_df, cat_cols, cat_maps)

    model, used_device = _fit_with_fallback(
        cfg,
        runtime_device,
        allow_fallback,
        model_family,
        cat_feature_indices,
        train_df[feature_cols],
        y,
        w,
    )
    if used_device != runtime_device:
        runtime_device = used_device
        print(f"[device] final model switched to {runtime_device}")

    print("[4/4] Writing submission...")
    pred = _predict_scores(model, test_df[feature_cols], runtime_device, model_family)
    if not np.isfinite(pred).all():
        bad = int((~np.isfinite(pred)).sum())
        raise ValueError(f"Invalid prediction values: non-finite count={bad}")

    submission = pd.DataFrame({"event_id": test_df["event_id"], "predict": pred})
    latest_submission_path = artifacts_dir / cfg["inference"]["submission_name"]
    ext = latest_submission_path.suffix or ".csv"
    stem = latest_submission_path.stem
    submission_path = submissions_dir / f"{run_name}__{stem}{ext}"
    submission.to_csv(submission_path, index=False)
    shutil.copy2(submission_path, latest_submission_path)
    print(f"Submission written: {submission_path}")
    print(f"Submission latest: {latest_submission_path}")
    if float(np.std(pred)) < 1e-8:
        print("[warn] submission scores are almost constant; ranking will be weak")

    sigmoid_pred = _sigmoid(pred)
    sigmoid_submission_path = submissions_dir / f"{run_name}__{stem}_sigmoid{ext}"
    pd.DataFrame({"event_id": test_df["event_id"], "predict": sigmoid_pred}).to_csv(
        sigmoid_submission_path, index=False
    )
    latest_sigmoid_submission_path = latest_submission_path.with_name(
        f"{latest_submission_path.stem}_sigmoid{latest_submission_path.suffix or '.csv'}"
    )
    shutil.copy2(sigmoid_submission_path, latest_sigmoid_submission_path)
    print(f"Sigmoid submission written: {sigmoid_submission_path}")
    print(f"Sigmoid submission latest: {latest_sigmoid_submission_path}")

    score_q = submission["predict"].quantile([0.5, 0.9, 0.99, 0.999]).to_dict()
    score_q_sigmoid = pd.Series(sigmoid_pred).quantile([0.5, 0.9, 0.99, 0.999]).to_dict()
    print(
        "score stats: "
        f"mean={submission['predict'].mean():.6f} std={submission['predict'].std():.6f} "
        f"q50={score_q.get(0.5):.6f} q90={score_q.get(0.9):.6f} "
        f"q99={score_q.get(0.99):.6f} q999={score_q.get(0.999):.6f}"
    )

    pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False)
    fi = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    fi.sort_values("importance", ascending=False).to_csv(run_dir / "feature_importance.csv", index=False)

    summary = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "submission_path": str(submission_path),
        "submission_sigmoid_path": str(sigmoid_submission_path),
        "submission_latest_path": str(latest_submission_path),
        "submission_sigmoid_latest_path": str(latest_sigmoid_submission_path),
        "config_path": args.config,
        "pipeline": "sequence_first_no_graph",
        "model_family": model_family,
        "requested_device": requested_device,
        "runtime_device": runtime_device,
        "cuda_available": has_cuda,
        "use_pretrain_profile": use_profile,
        "full_week_eval_unsampled": use_full_week_eval,
        "max_eval_week_rows": args.max_eval_week_rows,
        "include_unlabeled": include_unlabeled,
        "unlabeled_weight": float(cfg["dataset"].get("unlabeled_weight", 0.1)),
        "history_window_events": args.history_window_events,
        "ultra_burst_enabled": include_ultra_burst,
        "train_rows": int(len(train_df)),
        "labeled_rows": int(is_labeled.sum()),
        "unlabeled_rows": int((~is_labeled).sum()),
        "weight_sum_all": float(w.sum()),
        "weight_sum_unlabeled": float(w.loc[~is_labeled].sum()),
        "weight_unlabeled_ratio": float(w.loc[~is_labeled].sum() / max(float(w.sum()), 1e-9)),
        "cv_ap_labeled_mean": cv_mean,
        "cv_ap_labeled_std": cv_std,
        "cv_ap_all_events_mean": cv_all_events_mean,
        "cv_ap_all_events_std": cv_all_events_std,
        "cv_ap_all_events_sampled_mean": cv_all_events_sampled_mean,
        "cv_ap_all_events_sampled_std": cv_all_events_sampled_std,
        "cv_ap_proxy_mean": cv_all_events_mean,
        "cv_ap_proxy_std": cv_all_events_std,
        "cv_ap_proxy_lastday_mean": cv_last_mean,
        "cv_ap_proxy_lastday_std": cv_last_std,
        "cv_ap_seen_mean": cv_seen_mean,
        "cv_ap_seen_std": cv_seen_std,
        "cv_ap_cold_mean": cv_cold_mean,
        "cv_ap_cold_std": cv_cold_std,
        "cv_valid_pos_rate_primary_mean": cv_primary_pos_rate_mean,
        "cv_valid_pos_rate_primary_std": cv_primary_pos_rate_std,
        "submission_rows": int(len(submission)),
        "submission_event_id_unique": int(submission["event_id"].nunique()),
        "submission_score_mean": float(submission["predict"].mean()),
        "submission_score_std": float(submission["predict"].std()),
        "submission_score_q50": float(score_q.get(0.5, np.nan)),
        "submission_score_q90": float(score_q.get(0.9, np.nan)),
        "submission_score_q99": float(score_q.get(0.99, np.nan)),
        "submission_score_q999": float(score_q.get(0.999, np.nan)),
        "submission_sigmoid_score_mean": float(np.mean(sigmoid_pred)),
        "submission_sigmoid_score_std": float(np.std(sigmoid_pred)),
        "submission_sigmoid_score_q50": float(score_q_sigmoid.get(0.5, np.nan)),
        "submission_sigmoid_score_q90": float(score_q_sigmoid.get(0.9, np.nan)),
        "submission_sigmoid_score_q99": float(score_q_sigmoid.get(0.99, np.nan)),
        "submission_sigmoid_score_q999": float(score_q_sigmoid.get(0.999, np.nan)),
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
                "model_family": model_family,
                "runtime_device": runtime_device,
                "include_unlabeled": include_unlabeled,
                "full_week_eval_unsampled": use_full_week_eval,
                "history_window_events": args.history_window_events,
                "ultra_burst_enabled": include_ultra_burst,
                "train_rows": summary["train_rows"],
                "cv_ap_all_events_mean": cv_all_events_mean,
                "cv_ap_all_events_sampled_mean": cv_all_events_sampled_mean,
                "cv_ap_labeled_mean": cv_mean,
                "cv_ap_proxy_mean": cv_all_events_mean,
                "cv_ap_proxy_lastday_mean": cv_last_mean,
                "cv_ap_seen_mean": cv_seen_mean,
                "cv_ap_cold_mean": cv_cold_mean,
                "cv_valid_pos_rate_primary_mean": cv_primary_pos_rate_mean,
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
