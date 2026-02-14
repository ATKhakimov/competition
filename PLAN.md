# Sequence-First Plan to 0.12 PR-AUC

## 1) Objective
- Public LB progress: ~`0.02 -> 0.0693291616` (best as of 2026-02-14).
- Target: `>= 0.12` PR-AUC.
- Primary offline selection metric: `AP_all_events` (event-level on full validation week).

## 2) Problem Framing (Locked)
- Task is treated as **temporal behavioral anomaly ranking** with weak supervision.
- Labels:
  - `red` (`target=1`) = positive.
  - `yellow` (`target=0`) = reviewed negative.
  - `green` = unlabeled/unknown (not default negative).
- Modeling unit remains event-level, but features are strictly prefix (history-before-event).

## 3) Current Primary Architecture (v1, No Graph)
- Entry point: `scripts/train_sequence_first.py`.
- Config: `conf/sequence_first.yaml`.
- Training data: labeled only (`red + yellow`), no green in gradients.
- Model family: XGBoost / CatBoost (current stable leader on honest full-week eval: CatBoost GPU).
- Features:
  - Transaction context (time, amount transforms, base categoricals).
  - Device/session risk flags (target-free).
  - Sequence prefix features (rolling counts/sums, novelty, time-since-last, session-so-far).
  - Optional pretrain customer profile.

## 4) Validation Protocol (Locked)
- Time-based rolling CV (weekly), validation is full week `W` (all events).
- Report all:
  - `AP_all_events` (primary)
  - `AP_labeled`
  - `AP_lastday` (diagnostic)
  - `AP_seen`, `AP_cold` (diagnostic)
- Additional slices to track:
  - cold-start customers (low history),
  - lastday-only recall@topK.

## 5) Fast Diagnostic Experiment Backlog (A–G)
Run in this order unless blocked.

### A. Submission/Data Sanity
1. Submission integrity checks:
- length match, `event_id` uniqueness/order, no nulls.
- prediction distribution (mean/std/quantiles).
2. Metric split sanity:
- compare all-events vs lastday-only behavior.

### B. Model Ceiling Check
3. Same features, model swap:
- XGB vs CatBoost (native cats).
4. Ranking objective check:
- logistic objective vs grouped rank objective (`customer_id` or `customer_id x day`).

### C. Sequence Window Stress Test
5. History window ablation:
- N in `{5, 20, 100, all}`.
6. Ultra-short burst pack:
- add `cnt_1m`, `cnt_5m`, `cnt_30m`, `amt_sum_5m`, `cnt_5m/cnt_1d`.

### D. Device Novelty Additions
7. Fingerprint novelty block:
- `is_new_fingerprint_for_customer`,
- `fingerprint_tslast`,
- `fingerprint_cnt_prev`.
8. Geo/lang drift block:
- timezone/lang change indicators over recent window.

### E. Session Quality & Prefix
9. Session prefix block:
- `session_event_index`, `session_cnt_so_far`, `session_amt_sum_so_far`, `session_duration_so_far`.
10. Session fallback test:
- if `session_id` weak, fallback to `customer + 30m bucket`.

### F. Label Strategy Stress Test
11. Compare:
- `R+Y only`,
- `R+Y + green(very low weight)`,
- `R+Y + hard negatives`.
12. Two-stage cascade:
- Stage A filter (`toxic/burst/device-risk`),
- Stage B rerank on filtered set.

### G. Sequence Encoder Probe
13. Minimal GRU on last `N=50` events.
14. Hybrid:
- GRU embedding + booster.

## 6) Decision Gates
- Keep a change only if it improves `AP_all_events` and does not collapse week-to-week stability.
- If `AP_all_events` improves but LB does not, inspect score distribution tails and cold-start slice.
- If CatBoost/Rank objective adds `>= +0.01` on `AP_all_events`, promote to primary branch.

## 7) Required Outputs Per Run
- `summary.json`, `fold_metrics.csv`, `fold_metrics_live.csv`.
- prediction distribution diagnostics (quantiles).
- run row in `artifacts/runs/runs_index.csv`.

## 8) Near-Term Milestones
1. Lock best tabular recipe (`CatBoost`, window, burst, label strategy) by honest `AP_all_events`.
2. Finish Foundation pretrain (`Temporal Transformer`) and export embeddings.
3. Add embedding features to downstream CatBoost and run A/B against current best tabular-only.
4. Only after this decide whether to keep pure tabular or hybrid (`tabular + foundation embeddings`).

## 9) Status Snapshot (Updated 2026-02-14)
- Completed A1 submission sanity tooling: `scripts/check_submission.py`.
- Completed C first pass (window/burst ablations on 20k labeled).
- Enabled strict `full_week_unsampled` offline evaluation (no `unlabeled_modulo` in validation slice).
- Current honest baseline (full eval):
  - `history_window_events=20`
  - `ultra_burst=true`
  - `include_unlabeled=true`, `unlabeled_weight=0.01`
  - run: `artifacts/runs/seq_first_full_005_h20_burst_fullweek_eval_u001`
  - `cv_ap_all_events_mean = 0.002066`
  - diagnostics: `cv_ap_all_events_sampled_mean = 0.811704`, `cv_valid_pos_rate_primary_mean = 0.000570`
- Model comparison (same setup, honest full eval):
  - `XGB`: `cv_ap_all_events_mean = 0.002066` (`seq_first_full_005_h20_burst_fullweek_eval_u001`)
  - `CatBoost GPU`: `cv_ap_all_events_mean = 0.003771` (`seq_first_full_006_h20_burst_fullweek_eval_u001_cat_v4`)
  - current leader: `CatBoost GPU`
- Label strategy check (`history=20`, burst on, full-week eval, CatBoost):
  - `R+Y+green(0.01)`: `cv_ap_all_events_mean = 0.003833` (`debug_gpu0_once`)
  - `R+Y only`: `cv_ap_all_events_mean = 0.003970` (`debug_gpu1_labeled_only`)
  - decision: prefer `R+Y only` for now.
- D/E feature-pack ablation (new fingerprint/drift/session-fallback block):
  - `pack ON`: `cv_ap_all_events_mean = 0.003458` (`seq_de_pack_on_gpu0`)
  - `pack OFF`: `cv_ap_all_events_mean = 0.003732` (`seq_de_pack_off_gpu1`)
  - decision: keep new D/E blocks disabled by default until redesign.
- Latest tracked sequence run (MLflow-enabled sanity run):
  - `seq_mlflow_cat_h20_labeled_gpu0_http`
  - `cv_ap_all_events_mean = 0.003341`
- Best leaderboard submission (public LB):
  - file: `artifacts/submissions/seq_first_full_006_h20_burst_fullweek_eval_u001_cat_v4__submission_sequence_first_sigmoid.csv`
  - score: `0.0693291616`

## 10) Foundation Track (New)
- Goal: pretrain `Transaction Foundation Encoder` on unlabeled sequences, then feed embeddings into downstream CatBoost.
- Implemented components:
  - data pipeline: `src/competition/foundation_data.py`
  - model: `src/competition/foundation_model.py`
  - pretrain entrypoint: `scripts/pretrain_foundation.py`
  - embedding export: `scripts/export_foundation_embeddings.py`
  - config: `conf/foundation_pretrain.yaml`
- Current foundation recipe (v0):
  - Backbone: `TemporalTransformer` (`d_model=512`, `layers=8`, `heads=8`, `seq_len=128`)
  - Objectives: next-event (`event_type`, `channel`) + masked event modeling + `MSE(time_delta)`
  - Data: `pretrain + train + pretest + test` (unlabeled pretrain stage, no label leakage)
- Active run:
  - name: `foundation_pretrain_gpu1_001`
  - MLflow experiment: `competition-foundation` (id=`2`)
  - run_id: `d3a49bae94a04f6a88e75224ecf5d20d`
  - status at update time: `RUNNING` on GPU1
- Next after pretrain:
  1. export `event_embedding` and `customer_embedding` parquet.
  2. join embeddings into sequence-first/baseline train tables.
  3. compare `tabular-only` vs `tabular+embeddings` on honest `AP_all_events`.

## 11) Tracking / Ops Notes
- MLflow tracking integrated into:
  - `scripts/train_sequence_first.py`
  - `scripts/pretrain_foundation.py`
- Important: file-based backend (`file:./artifacts/mlruns`) produced unstable `Run not found` errors in this environment.
- Stable server command:
  - `mlflow server --host 0.0.0.0 --port 5000 --workers 1 --backend-store-uri sqlite:///artifacts/mlflow.db --default-artifact-root file:./artifacts/mlruns_artifacts`
- Current ETA visibility:
  - no `tqdm`/explicit ETA yet; progress is logged by fold (tabular) or by `step=X/Y` (foundation pretrain).

## 12) Explicitly De-Prioritized
- Label-based graph-risk as primary signal.
- Green-as-negative default training.
- Random split validation.
