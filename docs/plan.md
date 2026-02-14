# Plan Snapshot

Source of truth: `PLAN.md`

## Current direction

1. Keep improving strong tabular baseline (`sequence-first`, CatBoost-centric).
2. Develop foundation encoder and export sequence embeddings.
3. Compare:
   - tabular-only
   - tabular + foundation embeddings

## Key recent findings

- CatBoost is currently stronger than XGBoost on honest full-week evaluation.
- Labeled-only training (`R+Y`) outperformed low-weight unlabeled variant in latest controlled checks.
- The first D/E novelty pack iteration did not improve primary metric and is currently disabled by default.

## Foundation track

- Implemented in code and currently active.
- Pretrain + export pipelines are already available.
- Next major checkpoint: downstream A/B with exported embeddings joined into CatBoost pipeline.

## Update workflow

After each completed experiment batch:

1. Update `PLAN.md` status snapshot with exact run IDs and metrics.
2. Keep only decisions backed by `cv_ap_all_events_mean`.
3. Track whether offline gain transfers to leaderboard.
