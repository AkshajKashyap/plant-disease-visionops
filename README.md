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

### Verify

```bash
pytest
ruff check .
```
