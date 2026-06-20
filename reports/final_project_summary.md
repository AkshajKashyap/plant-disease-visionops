# Plant Disease VisionOps: Final Project Summary

## Problem

Plant disease image classifiers are often presented as a single accuracy score. This project builds
the surrounding workflow needed to ask harder questions: Was the dataset valid? Were splits
reproducible? Does transfer learning improve the baseline? Which classes fail? What happens when
lighting, focus, framing, or camera conditions change?

The result is a tested, CLI-driven pipeline for 38-class leaf-image classification. It covers data
ingestion, audit, deterministic split metadata, PyTorch loading, baseline and transfer-learning
experiments, per-class evaluation, synthetic corruption testing, and visual failure analysis.

## Pipeline

```mermaid
flowchart LR
    A[Manual dataset download] --> B[Prepare and validate layout]
    B --> C[Audit images and class balance]
    C --> D[Stratified split metadata]
    D --> E[PyTorch transforms and loaders]
    E --> F[Baseline CNN]
    E --> G[ResNet18 transfer learning]
    F --> H[Experiment comparison]
    G --> H
    H --> I[Robustness evaluation]
    I --> J[Failure reports and galleries]
```

All dataset files, generated checkpoints, and full-size figures remain local and are excluded from
Git. Structured reports and compressed failure galleries provide the reviewable evidence.

## Dataset Audit

| Item | Result |
|---|---:|
| Classes | 38 |
| Discovered images | 175,734 |
| Valid images | 175,734 |
| Invalid images | 0 |
| Image dimensions | 256 x 256 |
| Class count range | 4,104 to 5,054 |
| Maximum-to-minimum class ratio | 1.23 |

The seed-42, 70/15/15 stratified split produced 123,019 training, 26,362 validation, and 26,353
test images. No filepath occurred in more than one split.

## Experiments

| Experiment | Test accuracy | Test macro F1 |
|---|---:|---:|
| Baseline CNN, 3 epochs | 0.9154 | 0.9146 |
| ResNet18 transfer, 3 epochs | 0.9839 | 0.9838 |

Transfer learning improved test macro F1 by about 6.93 percentage points over the compact baseline.
This strong clean result is useful, but it is not the final conclusion.

## Why Clean Accuracy Is Insufficient

The same ResNet18 checkpoint was evaluated on seven deterministic corruptions at severity levels
1 through 3. Its average macro F1 across the 21 corrupted conditions was 0.8150, but performance
varied sharply by condition.

| Evaluation condition | Accuracy | Macro F1 | Macro F1 drop from clean |
|---|---:|---:|---:|
| Clean test split | 0.9839 | 0.9838 | 0.0000 |
| Brightness decrease, severity 3 | 0.3086 | 0.3286 | 0.6553 |
| Zoom in, severity 3 | 0.3754 | 0.4022 | 0.5816 |
| Contrast decrease, severity 3 | 0.5021 | 0.4895 | 0.4943 |

A model that is nearly perfect on curated images can still fail on an ordinary acquisition change
such as insufficient light. Robustness evaluation is therefore the central finding, not an
optional appendix to the clean score.

## Failure Analysis

| Condition | Mistakes | Test images | Error rate |
|---|---:|---:|---:|
| Clean | 425 | 26,353 | 1.61% |
| Brightness decrease, severity 3 | 18,220 | 26,353 | 69.14% |

Clean failures concentrate in visually related disease labels, including Tomato Target Spot versus
Tomato Early Blight and two corn leaf-spot/blight classes. Under severe darkening, the prediction
distribution collapses: Tomato Late Blight accounts for 11,027 wrong predictions. High-confidence
wrong predictions also show that softmax confidence is not calibrated reliability.

- [Clean misclassification gallery](assets/resnet18_transfer_3ep_failures_clean.jpg)
- [Severe brightness misclassification gallery](assets/resnet18_transfer_3ep_failures_brightness_decrease_s3.jpg)

## Final Takeaway

ResNet18 transfer learning produces a strong in-distribution result, but the robustness and failure
analysis change its interpretation. The project demonstrates a credible experimental pipeline and
an important negative result: clean-background performance does not establish field readiness.
Independent field evaluation, source-level deduplication, calibration, realistic augmentation, and
human review are prerequisites for any practical use.

See the [model card](model_card.md), [reproducibility guide](reproducibility.md), and
[limitations](limitations.md) for the operational details.
