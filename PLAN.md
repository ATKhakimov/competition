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
- Model: GPU XGBoost ranker-style scoring (`output_margin` used in submission).
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
1. Complete A + B + C (first pass) and pick strongest non-encoder setup.
2. Complete D + E and lock best feature pack.
3. Run F (label strategy) to lock production training set recipe.
4. Run G only after tabular ceiling plateaus.

## 10) Status Snapshot (Current)
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
- Best leaderboard submission (public LB):
  - file: `artifacts/submissions/seq_first_full_006_h20_burst_fullweek_eval_u001_cat_v4__submission_sequence_first_sigmoid.csv`
  - score: `0.0693291616`

## 9) Explicitly De-Prioritized
- Label-based graph-risk as primary signal.
- Green-as-negative default training.
- Random split validation.
