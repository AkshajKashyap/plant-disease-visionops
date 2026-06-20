# Limitations

## Dataset Representativeness

The dataset contains curated, centered leaf images with uniform 256 x 256 dimensions. This is
substantially simpler than real agricultural imagery, which may include soil, hands, tools,
multiple plants, clutter, occlusion, motion blur, mixed symptoms, variable growth stages, and
different cameras. Results should be described as performance on this dataset, not as general
plant-disease diagnostic accuracy.

## Background and Augmentation Artifacts

Clean backgrounds, capture setups, and augmentation signatures can become predictive shortcuts.
The filenames and repeated visual patterns indicate that augmented variants may be present. The
split pipeline guarantees unique filepaths across train, validation, and test, but it does not group
or deduplicate images by their original, pre-augmentation source. Related variants could therefore
appear in different splits and inflate clean performance.

## Same-Source Evaluation

All three splits come from the same local dataset and acquisition style. A stratified random test
split measures in-distribution generalization, not geographic, seasonal, device, farm, or laboratory
generalization. No external dataset has been used as a holdout.

## Severe Lighting Sensitivity

Brightness decrease severity 3 reduced ResNet18 accuracy from 0.9839 to 0.3086 and macro F1 from
0.9838 to 0.3286. Failure analysis found 18,220 mistakes among 26,353 images. This is a practical
warning: the model can fail systematically under dark input while remaining highly confident.
Synthetic corruption is only a proxy and does not cover the full range of field lighting.

## No Field Validation

The model has not been evaluated prospectively on images collected in farms, greenhouses, or by
mobile users. It has not been tested with unsupported crops, absent leaves, multiple diseases,
nutrient deficiencies, insect damage, or healthy plants outside the represented classes. The
closed-set classifier always chooses one of 38 labels, even when none is appropriate.

## No Probability Calibration

Softmax confidence is reported for error analysis, but no temperature scaling, expected calibration
error, reliability diagram, conformal prediction, abstention threshold, or out-of-distribution
detector has been evaluated. Several clean errors exceed 0.99 confidence. Probabilities must not be
interpreted as diagnostic certainty.

## No Human-in-the-Loop Workflow

There is no expert review interface, escalation rule, feedback capture, or label-correction process.
No acceptable-risk threshold has been defined for agricultural decisions. Any practical system
would need domain experts to review uncertain, unsupported, or high-impact cases and a mechanism to
record outcomes safely.

## Scope Not Yet Covered

- No model deployment API or user interface is included.
- No drift monitoring or post-deployment evaluation is included.
- No robustness-oriented retraining, weighted loss, focal loss, or calibration experiment has been
  run.
- No subgroup analysis by device, location, cultivar, farm, season, or demographic/economic impact
  is possible with the available metadata.
- Dataset provenance and licensing must be established by the user supplying the local data.

## Practical and Ethical Caution

This project is an engineering and evaluation portfolio, not a medical or agricultural diagnostic
tool. Incorrect predictions could lead to unnecessary pesticide use, missed disease, crop loss,
financial harm, or environmental damage. Model output should not drive treatment without qualified
human review, independent field validation, and an explicit risk-management process.

See the [model card](model_card.md) for intended use and recommended next steps.
