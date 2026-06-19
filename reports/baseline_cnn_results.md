# Baseline CNN Results

> This is a small baseline CNN for pipeline validation and comparison. It is not the final model.

## Run Summary

- Number of classes: 38
- Device: `cuda`
- Best validation epoch: 3
- Best checkpoint: `/home/akshaj/Building/plant-disease-visionops/artifacts/models/baseline_cnn_3ep/best_model.pt`

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
| learning_rate | 0.001 |
| num_workers | 2 |
| seed | 42 |
| dropout | 0.3 |

## Best Validation Metrics

- Loss: 0.273345
- Accuracy: 0.916015
- Macro F1: 0.915317
- Samples: 26362

## Test Metrics

- Loss: 0.271334
- Accuracy: 0.915418
- Macro F1: 0.914563
- Samples: 26353

## Weakest Test Classes

The ten classes with the lowest test F1 are shown, or all classes when fewer than ten exist.

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Tomato___Late_blight | 694 | 0.892000 | 0.642651 | 0.747069 |
| Grape___Black_rot | 708 | 0.963636 | 0.673729 | 0.793017 |
| Tomato___Target_Spot | 685 | 0.805638 | 0.792701 | 0.799117 |
| Tomato___Early_blight | 720 | 0.806993 | 0.801389 | 0.804181 |
| Tomato___Septoria_leaf_spot | 654 | 0.940741 | 0.776758 | 0.850921 |
| Grape___Esca_(Black_Measles) | 720 | 0.781897 | 0.995833 | 0.875993 |
| Tomato___Leaf_Mold | 705 | 0.797701 | 0.984397 | 0.881270 |
| Apple___healthy | 753 | 0.979233 | 0.814077 | 0.889050 |
| Cherry_(including_sour)___healthy | 684 | 0.802353 | 0.997076 | 0.889179 |
| Potato___Late_blight | 727 | 0.986733 | 0.818432 | 0.894737 |
