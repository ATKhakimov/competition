# Competition Project Documentation

This documentation describes how to run training pipelines, track experiments, and work with artifacts in this repository.

## Project at a glance

- Main objective: improve fraud ranking quality with a strong temporal setup.
- Primary offline metric: `AP_all_events`.
- Main tabular entrypoint: `scripts/train_sequence_first.py`.
- Foundation (embedding) entrypoint: `scripts/pretrain_foundation.py`.

## Documentation map

- **Setup**: environment and local docs.
- **Pipelines**: sequence-first, baseline, and foundation pretrain/export.
- **Tracking**: MLflow and run artifacts.
- **Artifacts**: what files are produced and where.
- **Plan Snapshot**: current direction and decisions.
- **FAQ**: operational answers.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Run sequence-first:

```bash
python scripts/train_sequence_first.py \
  --config conf/sequence_first.yaml \
  --run-name seq_local_001 \
  --device cuda
```

Run docs locally:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```
