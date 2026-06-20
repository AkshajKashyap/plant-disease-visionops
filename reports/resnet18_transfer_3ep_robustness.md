# Robustness Evaluation: resnet18_transfer_3ep

- Checkpoint: `/home/akshaj/Building/plant-disease-visionops/artifacts/models/resnet18_transfer_3ep/best_model.pt`
- Evaluated split: `test`
- Model: `resnet18`
- Clean accuracy: 0.983873
- Clean macro F1: 0.983844

## Corruption Results

| Corruption | Severity | Accuracy | Accuracy drop | Macro F1 | Macro F1 drop |
|---|---:|---:|---:|---:|---:|
| brightness_decrease | 1 | 0.962585 | 0.021288 | 0.962554 | 0.021290 |
| brightness_decrease | 2 | 0.831974 | 0.151899 | 0.835320 | 0.148525 |
| brightness_decrease | 3 | 0.308618 | 0.675255 | 0.328559 | 0.655285 |
| brightness_increase | 1 | 0.979547 | 0.004326 | 0.979418 | 0.004427 |
| brightness_increase | 2 | 0.924335 | 0.059538 | 0.925293 | 0.058551 |
| brightness_increase | 3 | 0.811293 | 0.172580 | 0.813799 | 0.170045 |
| gaussian_blur | 1 | 0.974234 | 0.009638 | 0.974232 | 0.009612 |
| gaussian_blur | 2 | 0.881797 | 0.102076 | 0.879689 | 0.104155 |
| gaussian_blur | 3 | 0.674155 | 0.309718 | 0.675777 | 0.308067 |
| gaussian_noise | 1 | 0.982317 | 0.001556 | 0.982211 | 0.001633 |
| gaussian_noise | 2 | 0.925360 | 0.058513 | 0.922484 | 0.061360 |
| gaussian_noise | 3 | 0.594733 | 0.389140 | 0.563168 | 0.420677 |
| contrast_decrease | 1 | 0.965317 | 0.018556 | 0.965271 | 0.018573 |
| contrast_decrease | 2 | 0.852920 | 0.130953 | 0.848926 | 0.134918 |
| contrast_decrease | 3 | 0.502068 | 0.481805 | 0.489543 | 0.494302 |
| rotation | 1 | 0.982355 | 0.001518 | 0.982262 | 0.001582 |
| rotation | 2 | 0.968505 | 0.015368 | 0.968291 | 0.015554 |
| rotation | 3 | 0.903882 | 0.079991 | 0.899636 | 0.084208 |
| zoom_in | 1 | 0.959587 | 0.024286 | 0.960014 | 0.023830 |
| zoom_in | 2 | 0.727356 | 0.256517 | 0.756225 | 0.227619 |
| zoom_in | 3 | 0.375403 | 0.608470 | 0.402241 | 0.581604 |

## Worst Corruption Settings

| Corruption | Severity | Macro F1 | Macro F1 drop |
|---|---:|---:|---:|
| brightness_decrease | 3 | 0.328559 | 0.655285 |
| zoom_in | 3 | 0.402241 | 0.581604 |
| contrast_decrease | 3 | 0.489543 | 0.494302 |
| gaussian_noise | 3 | 0.563168 | 0.420677 |
| gaussian_blur | 3 | 0.675777 | 0.308067 |

> High clean accuracy on curated leaf datasets may not guarantee robustness to real-world lighting, focus, framing, and camera changes.
