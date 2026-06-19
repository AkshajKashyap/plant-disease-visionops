# Experiment Results: resnet18_transfer_1ep

## Experiment Metadata

- Model: `resnet18`
- Pretrained: True
- Freeze backbone: False
- Device: `cuda`
- Number of classes: 38
- Best validation epoch: 1
- Best checkpoint: `/home/akshaj/Building/plant-disease-visionops/artifacts/models/resnet18_transfer_1ep/best_model.pt`

## Dataset Splits

| Split | Images |
|---|---:|
| Train | 123019 |
| Validation | 26362 |
| Test | 26353 |

## Hyperparameters

| Parameter | Value |
|---|---:|
| image_size | 128 |
| batch_size | 16 |
| epochs | 1 |
| learning_rate | 0.0003 |
| num_workers | 2 |
| seed | 42 |
| dropout | 0.3 |

## Best Validation Metrics

- Loss: 0.058570
- Accuracy: 0.979668
- Macro F1: 0.979525
- Samples: 26362

## Test Metrics

- Loss: 0.054517
- Accuracy: 0.980761
- Macro F1: 0.980676
- Samples: 26353

## Weakest Test Classes

The ten classes with the lowest test F1 are shown, or all classes when fewer than ten exist.

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Tomato___Target_Spot | 685 | 0.956318 | 0.894891 | 0.924585 |
| Tomato___Late_blight | 694 | 0.977600 | 0.880403 | 0.926459 |
| Tomato___Early_blight | 720 | 0.876847 | 0.988889 | 0.929504 |
| Tomato___Spider_mites Two-spotted_spider_mite | 653 | 0.961661 | 0.921899 | 0.941360 |
| Tomato___healthy | 722 | 0.918263 | 0.995845 | 0.955482 |
| Tomato___Leaf_Mold | 705 | 0.995455 | 0.931915 | 0.962637 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 615 | 0.981450 | 0.946341 | 0.963576 |
| Tomato___Septoria_leaf_spot | 654 | 0.995153 | 0.941896 | 0.967793 |
| Corn_(maize)___Northern_Leaf_Blight | 715 | 0.956403 | 0.981818 | 0.968944 |
| Potato___Late_blight | 727 | 0.956000 | 0.986245 | 0.970887 |

This report records one experiment configuration. Metrics are measured from its saved best checkpoint and are not cross-experiment estimates.
