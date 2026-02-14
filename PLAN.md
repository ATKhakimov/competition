# Graph-First Plan (Staj)

## 1. Goal
- Build a competition-ready anti-fraud pipeline with graph methods as the primary signal source.
- Optimize for PR-AUC (`average_precision_score`) with strict time-aware validation.
- Start with 1x RTX 5090 + high RAM, keep a clean upgrade path to multi-GPU.

## 2. Constraints Observed in Data
- Historical events volume: ~190.8M (`pretrain + train + pretest`).
- Test events: 633,683 rows.
- Labeled train events: 87,514 rows (`target=1`: 51,438; `target=0`: 36,076).
- Label setup is sparse; most train events are unlabeled.

## 3. Label Strategy (Locked)
- `target=1`: positive class, weight 1.0.
- `target=0`: reviewed negative, weight 0.8.
- Unlabeled (green): weak negatives in training, weight 0.05-0.2 (tunable).

## 4. Target Architecture
1. Graph construction:
- Heterogeneous graph centered on `customer` and behavioral tokens:
  `mcc_code`, `event_type_nm`, `channel_indicator_sub_type`, device fingerprint, session, time buckets.
- Weighted edges with count + recency decay.

2. Graph features:
- 1-hop/2-hop risk propagation from positive/negative seeds.
- Degree and weighted degree statistics.
- PPR-like risk scores.
- Community-level risk aggregates.
- Behavioral drift features (history vs current event context).

3. Embeddings:
- Node2Vec/DeepWalk on customer-token graph.
- Optional light hetero-GNN (GraphSAGE/HGT) for embedding refinement.

4. Final ranker:
- Event-level model (XGBoost/CatBoost/LightGBM family) on graph + tabular features.
- Output `event_id,predict` for all test events.

## 5. Validation Protocol
- Rolling weekly splits over train period (no future leakage).
- Metrics:
  - Main: PR-AUC (`average_precision_score`).
  - Monitoring: recall@topK and stability across folds.
- Final model selected by late-period fold robustness, not single-fold peak.

## 6. Infrastructure Strategy (Locked)
- First implementation: single-GPU-first (1x5090) + CPU preprocessing.
- Add multi-GPU only if one of these triggers is met:
  1. Embedding/GNN epoch time > 4-6 hours.
  2. VRAM bottleneck remains after AMP + sampling.
  3. Measured metric gain from larger effective batch/model depth.

## 7. Execution Roadmap
1. Stage 1 (now):
- Project skeleton, config, reproducible baseline with temporal CV and submission generation.
- Added run tracking artifacts and fold-level reporting.
- Upgraded to behavioral baseline:
  - customer pre-train profile features
  - sequence features (`Δtime`, rolling windows, novelty flags)
  - leaderboard-like proxy on last-day events per customer

2. Stage 2:
- Build graph store and propagation features.

3. Stage 3:
- Add graph embeddings, blend with baseline features.

4. Stage 4:
- Calibrate, ensemble, and produce final submissions.

## 8. Deliverables
- Config-driven training/inference pipeline.
- Temporal CV report.
- Proxy validation report closer to leaderboard behavior.
- Reproducible baseline submission.
- Incremental graph modules ready for Stage 2.

## 9. Immediate Next Actions (after first LB feedback)
- Align offline validation with leaderboard:
  - Track both labeled-only AP and proxy AP on week slices with unlabeled negatives.
  - Monitor recall@topK for operational ranking quality.
- Start graph-store implementation:
  - customer-token edges with recency/count weights.
  - exported graph artifacts for repeatable feature generation.
- Iterate model selection on proxy metric first, then submit.
