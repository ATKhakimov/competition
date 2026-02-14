# FAQ

## Where are submissions?

- Latest: `artifacts/submission_sequence_first.csv`
- Per run: `artifacts/submissions/<run_name>__submission_sequence_first.csv`

## Raw vs sigmoid submission

- Raw file uses model margin/logit.
- Sigmoid file maps raw scores to `[0, 1]`.
- Ranking is usually almost identical because sigmoid is monotonic.

## How to tell run is finished?

Check:

1. Process is gone from `ps`.
2. `summary.json` exists in `artifacts/runs/<run_name>/`.
3. `fold_metrics_live.csv` contains all folds.

## Where to watch progress?

- Tabular training: `fold_metrics_live.csv`
- Foundation training: MLflow metrics and step logs
- GPU: `nvidia-smi`

## How to open MLflow over SSH?

Forward port:

```bash
ssh -L 5000:127.0.0.1:5000 <user>@<server>
```

Open `http://127.0.0.1:5000` locally.
