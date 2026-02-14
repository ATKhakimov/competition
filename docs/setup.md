# Setup

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Data layout

Required files are expected in `db/`:

- `train_part_*.parquet`
- `train_labels.parquet`
- `test.parquet`
- `pretrain_part_*.parquet`
- `pretest.parquet`

## Documentation tooling

Install docs dependencies:

```bash
pip install -r requirements-docs.txt
```

Run local docs server:

```bash
mkdocs serve -a 0.0.0.0:8000
```

Build static docs:

```bash
mkdocs build
```

Output will be generated in `site/`.
