# Competition Starter

## Current Status
- Plan fixed in `PLAN.md`.
- Stage 1 baseline pipeline implemented:
- Stage 1 baseline pipeline implemented:
  - temporal CV
  - run tracking (`artifacts/runs/<run_name>`)
  - PU+reviewed weighting
  - weak unlabeled negatives
  - graph-risk token/pair features
  - behavioral sequence features (`Δtime`, rolling counts, novelty)
  - pre-train customer profile features (cached at `artifacts/cache/customer_profile_pretrain.parquet`)
- Latest generated submission: `artifacts/submission_baseline.csv`.

## Run

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Full run
```bash
python scripts/train_baseline.py --config conf/pipeline.yaml
```

### Sequence-first (no graph, recommended now)
```bash
python scripts/train_sequence_first.py \
  --config conf/sequence_first.yaml \
  --run-name seq_first_full_001 \
  --device cuda | tee artifacts/seq_first_full_001.log
```
Primary offline metric in this pipeline: `AP_all_events` (event-level on full validation week).
Inference writes two files: raw-margin submission and `*_sigmoid.csv` (same ranking, bounded scores).

Model swap on same pipeline:
```bash
# XGBoost (default)
python scripts/train_sequence_first.py --config conf/sequence_first.yaml --run-name seq_xgb --model-family xgb --device cuda

# CatBoost
python scripts/train_sequence_first.py --config conf/sequence_first.yaml --run-name seq_cat --model-family catboost --device cuda
```

### Sequence-first ablation knobs
```bash
# History window by number of previous events: all|100|20|5
python scripts/train_sequence_first.py --config conf/sequence_first.yaml --run-name seq_h20 --history-window-events 20 --device cuda

# Disable ultra-short burst block (1m/5m/30m features)
python scripts/train_sequence_first.py --config conf/sequence_first.yaml --run-name seq_no_burst --history-window-events 20 --disable-ultra-burst --device cuda

# Include unlabeled greens in training (for stress tests only)
python scripts/train_sequence_first.py --config conf/sequence_first.yaml --run-name seq_with_green --use-unlabeled true --unlabeled-weight 0.01 --device cuda
```

### Named run (recommended)
```bash
python scripts/train_baseline.py --config conf/pipeline.yaml --run-name exp_001
```

### Force device
```bash
# Auto: use GPU when available, else CPU
python scripts/train_baseline.py --config conf/pipeline.yaml --run-name exp_auto --device auto

# Force GPU
python scripts/train_baseline.py --config conf/pipeline.yaml --run-name exp_gpu --device cuda
```

### Research ablation flags
```bash
# Disable sequence and pretrain-profile features
python scripts/train_baseline.py --config conf/pipeline.yaml --run-name exp_base --disable-sequence --disable-pretrain-profile

# Graph risk modes: full | count | off
python scripts/train_baseline.py --config conf/pipeline.yaml --run-name exp_cnt --graph-risk-mode count

# Train on labeled only (red+yellow)
python scripts/train_baseline.py --config conf/pipeline.yaml --run-name exp_labeled_only --use-unlabeled false
```

### Fast smoke run
```bash
python scripts/train_baseline.py \
  --config conf/pipeline.yaml \
  --max-labeled-rows 20000 \
  --max-unlabeled-rows 20000 \
  --max-test-rows 50000
```

### Disable graph-risk features
```bash
python scripts/train_baseline.py --config conf/pipeline.yaml --disable-graph-risk
```

### Show run history
```bash
python scripts/show_runs.py
```

### Online graph leakage checks
```bash
python scripts/check_graph_online.py --config conf/pipeline.yaml
```

### Submission sanity checks
```bash
python scripts/check_submission.py --submission artifacts/submission_sequence_first.csv --test-file db/test.parquet
```

## Monitoring

### Live training log
```bash
python scripts/train_baseline.py --config conf/pipeline.yaml --run-name exp_002 | tee artifacts/exp_002.log
tail -f artifacts/exp_002.log
```

### GPU utilization
```bash
watch -n 1 nvidia-smi
```

### Metrics files
- Run summary: `artifacts/runs/<run_name>/summary.json`
- Live fold metrics (updated during CV): `artifacts/runs/<run_name>/fold_metrics_live.csv`
- Fold metrics: `artifacts/runs/<run_name>/fold_metrics.csv`
- Main CV KPI: `cv_ap_all_events_mean` in summary/index.
- OOF proxy preds: `artifacts/runs/<run_name>/oof_proxy_predictions.csv`
- OOF proxy last-day preds: `artifacts/runs/<run_name>/oof_proxy_lastday_predictions.csv`
- Feature importance: `artifacts/runs/<run_name>/feature_importance.csv`
- All runs index: `artifacts/runs/runs_index.csv`

## Files Added
- `conf/pipeline.yaml`
- `notebooks/research_ablation.ipynb`
- `src/competition/datasets.py`
- `src/competition/features.py`
- `src/competition/graph_risk.py`
- `src/competition/splits.py`
- `scripts/train_baseline.py`
