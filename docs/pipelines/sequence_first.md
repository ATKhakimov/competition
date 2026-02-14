# Sequence-First Pipeline

Primary training script: `scripts/train_sequence_first.py`  
Primary config: `conf/sequence_first.yaml`

## Typical run

```bash
python scripts/train_sequence_first.py \
  --config conf/sequence_first.yaml \
  --run-name seq_cat_h20 \
  --model-family catboost \
  --device cuda \
  --history-window-events 20 \
  --use-unlabeled false
```

## Important options

- `--model-family xgb|catboost`
- `--history-window-events all|5|20|100`
- `--disable-ultra-burst`
- `--disable-pretrain-profile`
- `--disable-full-week-eval`
- `--use-unlabeled config|true|false`
- `--unlabeled-weight <float>`

Additional ablations:

- `--disable-fingerprint-novelty`
- `--disable-geo-lang-drift`
- `--disable-session-fallback`

## Outputs

For each run:

- `artifacts/runs/<run_name>/summary.json`
- `artifacts/runs/<run_name>/fold_metrics.csv`
- `artifacts/runs/<run_name>/fold_metrics_live.csv`
- `artifacts/runs/<run_name>/feature_importance.csv`

Submissions:

- `artifacts/submissions/<run_name>__submission_sequence_first.csv`
- `artifacts/submissions/<run_name>__submission_sequence_first_sigmoid.csv`

## Evaluation notes

- Selection metric: `cv_ap_all_events_mean`
- Validation protocol: rolling weekly temporal CV
- Full-week unsampled validation is enabled by default in this script
