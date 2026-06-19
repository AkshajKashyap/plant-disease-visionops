# Experiment Results: resnet18_transfer_3ep

## Experiment Metadata

- Model: `resnet18`
- Pretrained: True
- Freeze backbone: False
- Device: `cuda`
- Number of classes: 38
- Best validation epoch: 2
- Best checkpoint: `/home/akshaj/Building/plant-disease-visionops/artifacts/models/resnet18_transfer_3ep/best_model.pt`

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
| epochs | 3 |
| learning_rate | 0.0003 |
| num_workers | 2 |
| seed | 42 |
| dropout | 0.3 |

## Best Validation Metrics

- Loss: 0.044776
- Accuracy: 0.984865
- Macro F1: 0.984758
- Samples: 26362

## Test Metrics

- Loss: 0.048430
- Accuracy: 0.983873
- Macro F1: 0.983844
- Samples: 26353

## Weakest Test Classes

The ten classes with the lowest test F1 are shown, or all classes when fewer than ten exist.

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Tomato___Early_blight | 720 | 0.898458 | 0.970833 | 0.933244 |
| Tomato___Target_Spot | 685 | 0.975385 | 0.925547 | 0.949813 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 615 | 0.976190 | 0.933333 | 0.954281 |
| Corn_(maize)___Northern_Leaf_Blight | 715 | 0.971751 | 0.962238 | 0.966971 |
| Tomato___Late_blight | 694 | 0.976540 | 0.959654 | 0.968023 |
| Blueberry___healthy | 681 | 1.000000 | 0.951542 | 0.975169 |
| Pepper,_bell___healthy | 745 | 0.963399 | 0.989262 | 0.976159 |
| Tomato___Septoria_leaf_spot | 654 | 0.982972 | 0.970948 | 0.976923 |
| Potato___healthy | 684 | 0.959155 | 0.995614 | 0.977044 |
| Tomato___Spider_mites Two-spotted_spider_mite | 653 | 0.992114 | 0.963247 | 0.977467 |

This report records one experiment configuration. Metrics are measured from its saved best checkpoint and are not cross-experiment estimates.
