# Reproducibility Guide

This guide rebuilds the pipeline from a manually obtained image dataset. The repository does not
download data, commit images, or distribute trained checkpoints. Commands should be run from the
repository root.

## 1. Install

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Pretrained ResNet18 training may download torchvision ImageNet weights once. Tests and evaluation
of an existing project checkpoint do not require that download.

## 2. Prepare the Dataset Layout

Download and extract a dataset manually beneath `data/external/`. If it is nested under folders
such as `train`, `valid`, `test`, `color`, or `PlantVillage`, flatten it into the expected class
layout:

```bash
python -m plant_disease_visionops.data.prepare_raw_layout \
  --input-dir data/external/plant_village \
  --output-dir data/raw \
  --mode copy \
  --manifest-path reports/raw_layout_manifest.json
```

Use `--mode symlink` to avoid copying image bytes. The command refuses a non-empty output directory;
`--overwrite` must be supplied explicitly when replacement is intended.

Expected layout:

```text
data/raw/
├── class_name_a/*.jpg
├── class_name_b/*.jpeg
└── class_name_c/*.png
```

## 3. Validate the Layout

```bash
python -m plant_disease_visionops.data.validate_layout \
  --data-dir data/raw
```

## 4. Audit the Dataset

```bash
python -m plant_disease_visionops.data.audit_dataset \
  --data-dir data/raw \
  --out-dir reports
```

This writes `reports/data_audit.json` and `reports/data_audit.md` from local files.

## 5. Create Deterministic Splits

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

The split command excludes corrupt images, stratifies within each class, and checks for filepath
overlap. It writes CSV metadata rather than copying images.

## 6. Inspect a Batch

```bash
python -m plant_disease_visionops.data.inspect_batch \
  --raw-data-dir data/raw \
  --processed-dir data/processed \
  --split train \
  --batch-size 8 \
  --image-size 128 \
  --out-dir artifacts/figures
```

## 7. Train the Baseline CNN

```bash
python -m plant_disease_visionops.training.train_baseline \
  --raw-data-dir data/raw \
  --processed-dir data/processed \
  --out-dir artifacts/models/baseline_cnn_3ep \
  --reports-dir reports \
  --figures-dir artifacts/figures \
  --image-size 128 \
  --batch-size 16 \
  --epochs 3 \
  --learning-rate 0.001 \
  --num-workers 2 \
  --seed 42
```

## 8. Train ResNet18

```bash
python -m plant_disease_visionops.training.train_experiment \
  --model-name resnet18 \
  --experiment-name resnet18_transfer_3ep \
  --pretrained true \
  --freeze-backbone false \
  --raw-data-dir data/raw \
  --processed-dir data/processed \
  --out-dir artifacts/models/resnet18_transfer_3ep \
  --reports-dir reports \
  --figures-dir artifacts/figures \
  --image-size 128 \
  --batch-size 16 \
  --epochs 3 \
  --learning-rate 0.0003 \
  --num-workers 2 \
  --seed 42
```

Use `--pretrained false` for an offline, randomly initialized run. That is a different experiment
and should not be expected to reproduce the reported transfer-learning metrics.

## 9. Compare Experiments

```bash
python -m plant_disease_visionops.evaluation.compare_experiments \
  --reports-dir reports \
  --out-md reports/experiment_comparison.md \
  --out-json reports/experiment_comparison.json
```

## 10. Evaluate Robustness

```bash
python -m plant_disease_visionops.evaluation.evaluate_robustness \
  --checkpoint artifacts/models/resnet18_transfer_3ep/best_model.pt \
  --experiment-name resnet18_transfer_3ep \
  --raw-data-dir data/raw \
  --processed-dir data/processed \
  --split test \
  --reports-dir reports \
  --figures-dir artifacts/figures \
  --image-size 128 \
  --batch-size 16 \
  --num-workers 2 \
  --seed 42
```

The full command runs the clean test pass plus seven corruptions at three severities, for 22 total
passes over the test split.

## 11. Analyze Failures

Clean test failures:

```bash
python -m plant_disease_visionops.evaluation.analyze_failures \
  --checkpoint artifacts/models/resnet18_transfer_3ep/best_model.pt \
  --experiment-name resnet18_transfer_3ep \
  --raw-data-dir data/raw \
  --processed-dir data/processed \
  --split test \
  --reports-dir reports \
  --figures-dir artifacts/figures \
  --image-size 128 \
  --batch-size 32 \
  --num-workers 2 \
  --seed 42 \
  --max-examples 80
```

Worst observed corruption:

```bash
python -m plant_disease_visionops.evaluation.analyze_failures \
  --checkpoint artifacts/models/resnet18_transfer_3ep/best_model.pt \
  --experiment-name resnet18_transfer_3ep \
  --raw-data-dir data/raw \
  --processed-dir data/processed \
  --split test \
  --reports-dir reports \
  --figures-dir artifacts/figures \
  --image-size 128 \
  --batch-size 32 \
  --num-workers 2 \
  --seed 42 \
  --max-examples 80 \
  --corruption brightness_decrease \
  --severity 3
```

## Verification and Reproducibility Notes

```bash
pytest
ruff check .
```

- The split membership is deterministic for unchanged files and seed 42.
- Training seeds Python, NumPy, and PyTorch, but exact floating-point results can vary by hardware,
  driver, library version, and accelerator kernels.
- Result JSON files are the source of truth for the metrics in the project documentation.
- `data/`, `artifacts/models/`, and generated figures are Git-ignored. Reproduction requires the
  same source dataset and locally regenerated checkpoints.
