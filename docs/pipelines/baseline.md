# Baseline Pipeline

Baseline script: `scripts/train_baseline.py`  
Config: `conf/pipeline.yaml`

## Typical run

```bash
python scripts/train_baseline.py \
  --config conf/pipeline.yaml \
  --run-name baseline_001 \
  --device cuda
```

## Main toggles

- `--disable-graph-risk`
- `--graph-risk-mode full|count|off`
- `--disable-sequence`
- `--disable-pretrain-profile`
- `--use-unlabeled config|true|false`

## Validation

- Rolling temporal folds from `src/competition/splits.py`
- Labeled-week validation with proxy metrics
- Summary and fold-level outputs under `artifacts/runs/<run_name>/`

## Utilities

```bash
python scripts/check_graph_online.py --config conf/pipeline.yaml
python scripts/check_submission.py --submission artifacts/submission_baseline.csv --test-file db/test.parquet
python scripts/show_runs.py
```
