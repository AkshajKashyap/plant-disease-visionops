# Dataset Split Summary

- Data directory: `/home/akshaj/Building/plant-disease-visionops/data/raw`
- Random seed: 42
- Ratios: train=0.7, val=0.15, test=0.15
- Discovered image files: 175734
- Valid images included: 175734
- Invalid/corrupt images excluded: 0

## Split Counts

| Split | Images |
|---|---:|
| train | 123019 |
| val | 26362 |
| test | 26353 |

## Per-Class Counts

| Class | Class index | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Apple___Apple_scab | 0 | 3528 | 756 | 756 |
| Apple___Black_rot | 1 | 3478 | 745 | 745 |
| Apple___Cedar_apple_rust | 2 | 3080 | 660 | 660 |
| Apple___healthy | 3 | 3514 | 753 | 753 |
| Blueberry___healthy | 4 | 3178 | 681 | 681 |
| Cherry_(including_sour)___healthy | 5 | 3195 | 685 | 684 |
| Cherry_(including_sour)___Powdery_mildew | 6 | 2946 | 631 | 631 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 7 | 2873 | 616 | 615 |
| Corn_(maize)___Common_rust_ | 8 | 3338 | 715 | 715 |
| Corn_(maize)___healthy | 9 | 3254 | 697 | 697 |
| Corn_(maize)___Northern_Leaf_Blight | 10 | 3339 | 716 | 715 |
| Grape___Black_rot | 11 | 3304 | 708 | 708 |
| Grape___Esca_(Black_Measles) | 12 | 3360 | 720 | 720 |
| Grape___healthy | 13 | 2961 | 635 | 634 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 14 | 3013 | 646 | 645 |
| Orange___Haunglongbing_(Citrus_greening) | 15 | 3518 | 754 | 754 |
| Peach___Bacterial_spot | 16 | 3216 | 689 | 689 |
| Peach___healthy | 17 | 3024 | 648 | 648 |
| Pepper,_bell___Bacterial_spot | 18 | 3348 | 717 | 717 |
| Pepper,_bell___healthy | 19 | 3479 | 746 | 745 |
| Potato___Early_blight | 20 | 3394 | 727 | 727 |
| Potato___healthy | 21 | 3192 | 684 | 684 |
| Potato___Late_blight | 22 | 3394 | 727 | 727 |
| Raspberry___healthy | 23 | 3116 | 668 | 668 |
| Soybean___healthy | 24 | 3538 | 758 | 758 |
| Squash___Powdery_mildew | 25 | 3038 | 651 | 651 |
| Strawberry___healthy | 26 | 3192 | 684 | 684 |
| Strawberry___Leaf_scorch | 27 | 3105 | 666 | 665 |
| Tomato___Bacterial_spot | 28 | 2978 | 638 | 638 |
| Tomato___Early_blight | 29 | 3360 | 720 | 720 |
| Tomato___healthy | 30 | 3370 | 722 | 722 |
| Tomato___Late_blight | 31 | 3240 | 694 | 694 |
| Tomato___Leaf_Mold | 32 | 3293 | 706 | 705 |
| Tomato___Septoria_leaf_spot | 33 | 3054 | 654 | 654 |
| Tomato___Spider_mites Two-spotted_spider_mite | 34 | 3046 | 653 | 653 |
| Tomato___Target_Spot | 35 | 3198 | 685 | 685 |
| Tomato___Tomato_mosaic_virus | 36 | 3133 | 672 | 671 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 37 | 3432 | 735 | 735 |

## Leakage Check

Leakage check **passed** with 0 overlapping filepaths.
