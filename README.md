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

## Milestone 4: Baseline CNN

Train the compact three-block CNN after audit, split generation, and batch inspection succeed:

```bash
python -m plant_disease_visionops.training.train_baseline \
  --raw-data-dir data/raw \
  --processed-dir data/processed \
  --out-dir artifacts/models/baseline_cnn \
  --reports-dir reports \
  --figures-dir artifacts/figures \
  --image-size 128 \
  --batch-size 16 \
  --epochs 3 \
  --learning-rate 0.001 \
  --num-workers 2 \
  --seed 42
```

The equivalent Make target is `make train-baseline`. Training automatically selects CUDA or MPS
when available and otherwise uses CPU; pass `--device cpu` to force CPU execution. The best model
is selected by validation macro F1, reloaded, and evaluated on the test split.

The run writes:

- `artifacts/models/baseline_cnn/best_model.pt`
- `artifacts/models/baseline_cnn/last_model.pt`
- `artifacts/models/baseline_cnn/history.json`
- `reports/baseline_cnn_results.json`
- `reports/baseline_cnn_results.md`
- `artifacts/figures/baseline_cnn_confusion_matrix.png`
- `artifacts/figures/baseline_cnn_training_curves.png`

Metrics and reports come only from the completed run. This compact CNN is a pipeline baseline, not
the final architecture; no pretrained model or ResNet is used in this milestone.

## Milestone 5: Transfer Learning and Comparison

The original `train_baseline` command remains supported. The generalized command can run the same
baseline explicitly:

```bash
python -m plant_disease_visionops.training.train_experiment \
  --model-name baseline_cnn \
  --experiment-name baseline_cnn_generalized \
  --pretrained false \
  --freeze-backbone false \
  --raw-data-dir data/raw \
  --processed-dir data/processed \
  --out-dir artifacts/models/baseline_cnn_generalized \
  --reports-dir reports \
  --figures-dir artifacts/figures \
  --image-size 128 --batch-size 16 --epochs 3 \
  --learning-rate 0.001 --num-workers 2 --seed 42
```

Run ResNet18 transfer learning with ImageNet weights:

```bash
python -m plant_disease_visionops.training.train_experiment \
  --model-name resnet18 \
  --pretrained true \
  --freeze-backbone false \
  --raw-data-dir data/raw \
  --processed-dir data/processed \
  --out-dir artifacts/models/resnet18_transfer \
  --reports-dir reports \
  --figures-dir artifacts/figures \
  --image-size 128 --batch-size 16 --epochs 3 \
  --learning-rate 0.0003 --num-workers 2 --seed 42
```

The experiment name defaults to the final `--out-dir` component, so this run writes
`reports/resnet18_transfer_results.json` and matching method-specific Markdown and figure files.
Use `--freeze-backbone true` to train only the replacement classifier. Pretrained weights may
require a one-time torchvision download; use `--pretrained false` for offline random initialization.

Compare every `reports/*_results.json` file, including the existing Milestone 4 baseline report:

```bash
python -m plant_disease_visionops.evaluation.compare_experiments \
  --reports-dir reports \
  --out-md reports/experiment_comparison.md \
  --out-json reports/experiment_comparison.json
```

PlantVillage-style test accuracy can be inflated by clean backgrounds, near-duplicate acquisition
conditions, and synthetic augmentation. Model selection should therefore be followed by robustness
evaluation on field images and later drift tests; a higher in-distribution score is not evidence of
production robustness by itself.

## Verify

```bash
pytest
ruff check .
```
