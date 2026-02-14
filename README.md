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
- OOF proxy preds: `artifacts/runs/<run_name>/oof_proxy_predictions.csv`
- OOF proxy last-day preds: `artifacts/runs/<run_name>/oof_proxy_lastday_predictions.csv`
- Feature importance: `artifacts/runs/<run_name>/feature_importance.csv`
- All runs index: `artifacts/runs/runs_index.csv`

## Files Added
- `conf/pipeline.yaml`
- `src/competition/datasets.py`
- `src/competition/features.py`
- `src/competition/graph_risk.py`
- `src/competition/splits.py`
- `scripts/train_baseline.py`
