# Artifacts

## Main directories

- `artifacts/submissions/`  
  Per-run submissions (`raw` and `sigmoid` variants).
- `artifacts/runs/`  
  Run-level metrics and diagnostics.
- `artifacts/foundation/`  
  Foundation checkpoints and embedding exports.
- `artifacts/reports/`  
  Aggregated comparison reports.

## Submission files

Latest pointers:

- `artifacts/submission_sequence_first.csv`
- `artifacts/submission_sequence_first_sigmoid.csv`
- `artifacts/submission_baseline.csv`

Per-run files:

- `artifacts/submissions/<run_name>__submission_sequence_first.csv`
- `artifacts/submissions/<run_name>__submission_sequence_first_sigmoid.csv`

## How to validate submission

```bash
python scripts/check_submission.py \
  --submission artifacts/submission_sequence_first.csv \
  --test-file db/test.parquet
```

## Run history

```bash
python scripts/show_runs.py
```
