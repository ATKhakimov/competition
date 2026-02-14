# Foundation Pipeline

Foundation pretrain script: `scripts/pretrain_foundation.py`  
Export script: `scripts/export_foundation_embeddings.py`  
Config: `conf/foundation_pretrain.yaml`

## Goal

Pretrain a temporal encoder on unlabeled sequences, then export embeddings for downstream fraud ranking.

## Architecture (current)

- Temporal Transformer encoder
- Multi-task objective:
  - next event type
  - next channel
  - masked event/channel modeling
  - time-delta regression

## Pretrain run

```bash
python scripts/pretrain_foundation.py \
  --config conf/foundation_pretrain.yaml \
  --run-name foundation_pretrain_001 \
  --device cuda \
  --mlflow-uri http://127.0.0.1:5000 \
  --mlflow-experiment competition-foundation
```

## Export embeddings

```bash
python scripts/export_foundation_embeddings.py \
  --config conf/foundation_pretrain.yaml \
  --checkpoint artifacts/foundation/foundation_pretrain_001/checkpoint_final.pt \
  --run-name foundation_export_001 \
  --device cuda
```

## Exported artifacts

- `artifacts/foundation/<run_name>/event_embeddings.parquet`
- `artifacts/foundation/<run_name>/customer_embeddings.parquet`
- `artifacts/foundation/<run_name>/export_summary.json`

## Current status

Foundation track is active and intended to be combined with downstream CatBoost rather than replacing tabular features.
