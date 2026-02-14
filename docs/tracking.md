# Tracking

## Run artifacts (filesystem)

Each training run writes:

- `artifacts/runs/<run_name>/summary.json`
- `artifacts/runs/<run_name>/fold_metrics_live.csv`
- `artifacts/runs/<run_name>/fold_metrics.csv`
- optional feature importance and OOF files

## MLflow

Integrated into:

- `scripts/train_sequence_first.py`
- `scripts/pretrain_foundation.py`

### Recommended server command

Use SQLite backend in this environment:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 1 \
  --backend-store-uri sqlite:///artifacts/mlflow.db \
  --default-artifact-root file:./artifacts/mlruns_artifacts
```

### Training with MLflow tracking

```bash
python scripts/train_sequence_first.py \
  --config conf/sequence_first.yaml \
  --run-name seq_mlflow_001 \
  --mlflow-uri http://127.0.0.1:5000 \
  --mlflow-experiment competition-sequence-first
```

```bash
python scripts/pretrain_foundation.py \
  --config conf/foundation_pretrain.yaml \
  --run-name foundation_mlflow_001 \
  --mlflow-uri http://127.0.0.1:5000 \
  --mlflow-experiment competition-foundation
```

### Remote server (SSH)

If MLflow runs on a remote host, forward port:

```bash
ssh -L 5000:127.0.0.1:5000 <user>@<server>
```

Then open `http://127.0.0.1:5000` in local browser.
