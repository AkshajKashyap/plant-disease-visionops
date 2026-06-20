# Model Card: Plant Disease ResNet18 Classifier

## Model Details

- **Model:** ResNet18 transfer-learning classifier
- **Experiment:** `resnet18_transfer_3ep`
- **Task:** single-label classification of RGB leaf images into 38 crop/disease classes
- **Initialization:** torchvision ImageNet pretrained weights
- **Training:** all backbone and classifier parameters fine-tuned for 3 epochs
- **Input:** RGB image resized and center-cropped to 128 x 128 for evaluation
- **Output:** 38 uncalibrated class logits
- **Artifact availability:** checkpoints are generated locally under `artifacts/models/` and are
  intentionally excluded from Git. This repository contains reports, not a distributed model
  artifact.

## Intended Use

This model and repository are intended for:

- demonstrating a reproducible computer vision workflow from data audit through failure analysis;
- benchmarking a compact CNN against transfer learning on a curated leaf-image dataset;
- studying class-level errors and sensitivity to synthetic image corruptions; and
- serving as an engineering foundation for later field-data validation.

## Not Intended Use

The model is not intended for autonomous crop diagnosis, pesticide selection, treatment decisions,
yield forecasting, or safety-critical agricultural use. It has not been validated on field images,
mobile-camera captures, multiple leaves, mixed symptoms, or crops outside the training labels.

**This is not a medical or agricultural diagnostic tool.** A prediction should not replace review
by a qualified plant pathologist, agronomist, extension specialist, or other domain expert.

## Dataset

The evaluated local dataset is a curated PlantVillage-style collection organized into 38 class
folders. It contains 175,734 valid images and no corrupt images were found by the project audit.
Every audited image is 256 x 256 pixels. Class counts range from 4,104 to 5,054 images, a 1.23
maximum-to-minimum ratio.

| Split | Images | Share |
|---|---:|---:|
| Train | 123,019 | 70% |
| Validation | 26,362 | 15% |
| Test | 26,353 | 15% |

Splits are class-stratified with seed 42. The filepath leakage check found zero overlap across
train, validation, and test metadata. This check does not rule out near-duplicate or augmented
versions of the same source image across splits.

Dataset acquisition and licensing are not automated by this repository. Users must establish the
provenance and permitted use of any dataset they place under `data/`.

## Preprocessing

- Images are decoded with Pillow and converted to RGB.
- Training uses a random resized crop to 128 x 128, random horizontal flip, and ImageNet
  normalization.
- Validation and test use deterministic resize, center crop to 128 x 128, and ImageNet
  normalization.
- Corruption evaluation applies the selected deterministic corruption before the normal evaluation
  transform.

## Architectures Compared

| Experiment | Architecture | Parameters | Initialization | Fine-tuning |
|---|---|---:|---|---|
| `baseline_cnn` | Three convolution blocks, global pooling, linear classifier | 98,374 | Random | Full model |
| `resnet18_transfer_3ep` | ResNet18 with a 38-class replacement head | 11,196,006 | ImageNet | Full model |

Both reported experiments used 128 x 128 inputs, batch size 16, and 3 epochs. The baseline used a
learning rate of 0.001; ResNet18 used 0.0003.

## Clean Performance

Metrics are from the saved best checkpoint for each completed experiment and the 26,353-image test
split.

| Model | Test accuracy | Test macro F1 |
|---|---:|---:|
| Baseline CNN | 0.9154 | 0.9146 |
| ResNet18 transfer | 0.9839 | 0.9838 |

## Robustness Performance

ResNet18 was evaluated on seven deterministic corruptions at three severity levels each. The mean
macro F1 over all 21 corrupted conditions was 0.8150. The clean score does not describe behavior
under distribution shift.

| Condition | Severity | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Clean | n/a | 0.9839 | 0.9838 |
| Brightness decrease | 3 | 0.3086 | 0.3286 |
| Zoom in | 3 | 0.3754 | 0.4022 |
| Contrast decrease | 3 | 0.5021 | 0.4895 |
| Gaussian noise | 3 | 0.5947 | 0.5632 |
| Gaussian blur | 3 | 0.6742 | 0.6758 |

## Failure Analysis

- Clean test data: 425 mistakes from 26,353 images, a 1.61% error rate.
- Brightness decrease severity 3: 18,220 mistakes, a 69.14% error rate.
- On clean images, the most common confusion was Tomato Target Spot predicted as Tomato Early
  Blight (21 examples).
- Under severe darkening, predictions collapsed heavily toward Tomato Late Blight, which appeared
  as the predicted label in 11,027 mistakes.
- Some clean errors had confidence above 0.99. The output probabilities are therefore not suitable
  as calibrated certainty estimates.

## Risks and Limitations

- Curated, centered leaf images do not represent farm backgrounds, variable cameras, occlusion,
  multiple leaves, or mixed disease symptoms.
- Background and augmentation patterns may make classes easier to separate and may leak source
  characteristics across metadata splits.
- Train, validation, and test images come from the same source distribution.
- Severe lighting reduction caused a 0.6553 absolute macro F1 drop.
- No independent field dataset, geographic subgroup analysis, probability calibration, abstention
  policy, or human review workflow has been evaluated.
- The closed-set classifier must choose among its 38 labels even for unsupported inputs.

## Recommended Next Steps

1. Deduplicate by source image before splitting and document dataset provenance and licensing.
2. Evaluate on independently collected field images across devices, locations, and lighting.
3. Measure calibration and add abstention or out-of-distribution handling.
4. Train with realistic lighting and acquisition augmentations, then repeat robustness analysis.
5. Add expert review for ambiguous labels and high-risk confusions.
6. Define monitoring and human escalation requirements before considering deployment.

## Evidence

- [Experiment comparison](experiment_comparison.md)
- [Robustness report](resnet18_transfer_3ep_robustness.md)
- [Clean failure analysis](resnet18_transfer_3ep_failures_clean.md)
- [Severe brightness failure analysis](resnet18_transfer_3ep_failures_brightness_decrease_s3.md)
- [Detailed limitations](limitations.md)
