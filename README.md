# Plant Disease VisionOps

Production-style computer vision system for plant disease classification with efficient image pipelines, imbalance-aware CNN training, per-class error analysis, FastAPI inference, Streamlit demo, and drift monitoring.

## Milestone 1: Dataset Audit

This milestone discovers and validates a local class-organized image dataset. It does not train a
model or provide inference services.

Use this directory layout:

```text
data/raw/
├── healthy/
│   ├── leaf_001.jpg
│   └── leaf_002.png
└── late_blight/
    └── leaf_003.jpeg
```

Only `.jpg`, `.jpeg`, and `.png` files directly inside each class directory are scanned. Extension
matching is case-insensitive. Dataset files under `data/raw/` are ignored by Git.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

### Run the audit

```bash
python -m plant_disease_visionops.data.audit_dataset \
  --data-dir data/raw \
  --out-dir reports
```

The installed `audit-dataset` command is equivalent:

```bash
audit-dataset --data-dir data/raw --out-dir reports
```

The command writes `reports/data_audit.json` for machine consumption and
`reports/data_audit.md` for review. Reports include total and per-class counts, corrupt-image
details, size statistics computed from valid images, and a valid-image class imbalance summary.
No report is generated when the dataset directory is missing or contains no supported images.

## Milestone 2: Dataset Splits

Generate reproducible, class-stratified metadata after auditing the local dataset:

```bash
python -m plant_disease_visionops.data.make_splits \
  --data-dir data/raw \
  --out-dir data/processed \
  --reports-dir reports \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

The installed `make-splits` command is equivalent. It writes:

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/class_to_index.json`
- `data/processed/split_summary.json`
- `reports/split_summary.md`

CSV filepaths are relative to the dataset root. Each class must contain at least three valid images
because train, validation, and test ratios must all be greater than zero. Invalid images are
excluded, and the same seed and inputs produce the same split membership.

No images are copied or moved, no dataset is downloaded, and no model is trained by this command.

## Verify

```bash
pytest
ruff check .
```
