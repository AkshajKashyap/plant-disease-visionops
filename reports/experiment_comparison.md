# Experiment Comparison

| Experiment | Model | Test accuracy | Test macro F1 | Best val macro F1 | Epochs | Image size | Batch size | Pretrained | Freeze backbone |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline_cnn | baseline_cnn | 0.915418 | 0.914563 | 0.915317 | 3 | 128 | 16 | false | false |
| resnet18_transfer_1ep | resnet18 | 0.980761 | 0.980676 | 0.979525 | 1 | 128 | 16 | true | false |
| resnet18_transfer_3ep | resnet18 | 0.983873 | 0.983844 | 0.984758 | 3 | 128 | 16 | true | false |

Rows reproduce metrics from each experiment's saved result JSON. They do not rerun evaluation or infer missing scores.
