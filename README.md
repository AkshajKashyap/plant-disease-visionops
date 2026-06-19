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

## Milestone 3: PyTorch Data Loading

The PyTorch data layer consumes the Milestone 2 CSV files, resolves their relative filepaths under
`data/raw`, converts every image to RGB, and applies separate training and evaluation transforms.
Training uses random resized crops and horizontal flips; validation and test transforms are
deterministic. All splits use ImageNet normalization.

Inspect a transformed training batch without training a model:

```bash
python -m plant_disease_visionops.data.inspect_batch \
  --raw-data-dir data/raw \
  --processed-dir data/processed \
  --split train \
  --batch-size 8 \
  --image-size 224 \
  --out-dir artifacts/figures
```

The command prints the batch tensor shape, class indices, and per-class counts. It saves a
denormalized grid to `artifacts/figures/sample_batch_train.png`. The installed `inspect-batch`
command is equivalent.

For Python callers, use
`plant_disease_visionops.data.loaders.create_datasets` or
`plant_disease_visionops.data.loaders.create_dataloaders`. Training loaders shuffle by default;
validation and test loaders do not.

## Milestone 3.5: Real Dataset Setup

Datasets are never downloaded automatically. Download or extract a dataset manually beneath
`data/external/`, which is ignored by Git:

```text
data/external/plant_village/
├── train/healthy/*.jpg
├── train/late_blight/*.jpg
├── valid/healthy/*.jpg
└── test/late_blight/*.jpg
```

If images are nested beneath wrapper folders such as `train`, `valid`, `test`, `color`, or
`PlantVillage`, prepare a flat raw layout first:

```bash
python -m plant_disease_visionops.data.prepare_raw_layout \
  --input-dir data/external/plant_village \
  --output-dir data/raw \
  --mode copy
```

Use `--mode symlink` to avoid duplicating image bytes. Symlinks use absolute source paths, so the
downloaded dataset must remain in place. The command refuses a non-empty `data/raw` directory;
`--overwrite` explicitly replaces it. Every destination filename includes a source-path hash, and
the operation is recorded in `reports/raw_layout_manifest.json`.

If the downloaded dataset already has direct class folders, place or copy them under `data/raw`
and skip the preparation command. Validate either setup before auditing:

```bash
python -m plant_disease_visionops.data.validate_layout --data-dir data/raw
```

The complete manual integration workflow is:

```bash
# 1. Audit valid and corrupt files.
python -m plant_disease_visionops.data.audit_dataset \
  --data-dir data/raw --out-dir reports

# 2. Generate deterministic split metadata.
python -m plant_disease_visionops.data.make_splits \
  --data-dir data/raw --out-dir data/processed --reports-dir reports \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42

# 3. Inspect one transformed batch without training.
python -m plant_disease_visionops.data.inspect_batch \
  --raw-data-dir data/raw --processed-dir data/processed \
  --split train --batch-size 8 --image-size 224 --out-dir artifacts/figures
```

The preparation helper organizes candidate files by extension. The audit remains the authority for
detecting corrupt or unreadable images before split generation.

## Verify

```bash
pytest
ruff check .
```
