# Failure Analysis: resnet18_transfer_3ep

- Checkpoint: `/home/akshaj/Building/plant-disease-visionops/artifacts/models/resnet18_transfer_3ep/best_model.pt`
- Model: `resnet18`
- Evaluated split: `test`
- Corruption: `brightness_decrease`
- Severity: `3`
- Total images: 26353
- Total mistakes: 18220
- Error rate: 0.691382
- Misclassification grid: `/home/akshaj/Building/plant-disease-visionops/artifacts/figures/resnet18_transfer_3ep_failures_brightness_decrease_s3.png`

## Top True Labels by Mistake Count

| Label | Mistakes |
|---|---:|
| Soybean___healthy | 738 |
| Potato___Early_blight | 719 |
| Apple___Black_rot | 714 |
| Pepper,_bell___Bacterial_spot | 697 |
| Cherry_(including_sour)___healthy | 684 |
| Tomato___Leaf_Mold | 682 |
| Pepper,_bell___healthy | 666 |
| Strawberry___Leaf_scorch | 656 |
| Tomato___Tomato_mosaic_virus | 652 |
| Tomato___Target_Spot | 646 |

## Top Predicted Labels in Mistakes

| Label | Mistakes |
|---|---:|
| Tomato___Late_blight | 11027 |
| Corn_(maize)___healthy | 1929 |
| Apple___healthy | 1867 |
| Apple___Apple_scab | 904 |
| Tomato___Early_blight | 833 |
| Peach___healthy | 546 |
| Tomato___Bacterial_spot | 225 |
| Potato___Late_blight | 184 |
| Potato___healthy | 146 |
| Corn_(maize)___Northern_Leaf_Blight | 83 |

## Most Common Confusions

| True label | Predicted label | Mistakes |
|---|---|---:|
| Tomato___Leaf_Mold | Tomato___Late_blight | 655 |
| Tomato___Early_blight | Tomato___Late_blight | 620 |
| Cherry_(including_sour)___healthy | Apple___healthy | 591 |
| Tomato___Tomato_mosaic_virus | Tomato___Late_blight | 543 |
| Tomato___Bacterial_spot | Tomato___Late_blight | 498 |
| Tomato___Spider_mites Two-spotted_spider_mite | Tomato___Late_blight | 494 |
| Tomato___Target_Spot | Tomato___Late_blight | 487 |
| Tomato___Septoria_leaf_spot | Tomato___Late_blight | 482 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato___Late_blight | 482 |
| Pepper,_bell___Bacterial_spot | Tomato___Late_blight | 480 |

## Highest-Confidence Wrong Predictions

| Filepath | True label | Predicted label | Confidence | True probability |
|---|---|---|---:|---:|
| Tomato___Leaf_Mold/c9a5bad7-c42c-40e5-a8de-0631a5522018___Crnl_L.Mold 6872_180deg__d3bbbe85df3d.jpg | Tomato___Leaf_Mold | Tomato___Late_blight | 1.000000 | 0.000000 |
| Corn_(maize)___Common_rust_/RS_Rust 1680_flipLR__2752baeada22.jpg | Corn_(maize)___Common_rust_ | Tomato___Late_blight | 1.000000 | 0.000000 |
| Corn_(maize)___Common_rust_/RS_Rust 2358__60334d35fd78.jpg | Corn_(maize)___Common_rust_ | Tomato___Late_blight | 1.000000 | 0.000000 |
| Blueberry___healthy/ca4ea722-9bf0-4094-bf8b-e15091bab904___RS_HL 0620__d341c8c6a27e.jpg | Blueberry___healthy | Tomato___Late_blight | 1.000000 | 0.000000 |
| Tomato___Leaf_Mold/b75fa72e-186c-4d64-a88c-dc037f275415___Crnl_L.Mold 6531_180deg__2df3b3f5cb64.jpg | Tomato___Leaf_Mold | Tomato___Late_blight | 1.000000 | 0.000000 |
| Corn_(maize)___Common_rust_/RS_Rust 2039_flipLR__cd354e606233.jpg | Corn_(maize)___Common_rust_ | Tomato___Late_blight | 1.000000 | 0.000000 |
| Corn_(maize)___Common_rust_/RS_Rust 2039_flipLR__fab0ea93f891.jpg | Corn_(maize)___Common_rust_ | Tomato___Late_blight | 1.000000 | 0.000000 |
| Tomato___Early_blight/a3d07302-420d-4b4e-94f2-6694ef9d2088___RS_Erly.B 7652_180deg__748870a101dd.jpg | Tomato___Early_blight | Tomato___Late_blight | 1.000000 | 0.000000 |
| Tomato___Leaf_Mold/a95f0a82-5cad-423a-be56-8205317909e8___Crnl_L.Mold 8677_flipTB__066b084f6e5f.jpg | Tomato___Leaf_Mold | Tomato___Late_blight | 1.000000 | 0.000000 |
| Tomato___Leaf_Mold/462317c2-9028-473b-9dc6-523eb348e47b___Crnl_L.Mold 6875_180deg__26fd1f787d06.jpg | Tomato___Leaf_Mold | Tomato___Late_blight | 0.999999 | 0.000001 |

## Lowest-Confidence Correct Predictions

| Filepath | Label | Confidence |
|---|---|---:|
| Apple___Black_rot/deffd3c8-a9c7-4082-ba2a-4a54aad8ab17___JR_FrgE.S 2789_270deg__c58e24c1af92.jpg | Apple___Black_rot | 0.159183 |
| Pepper,_bell___healthy/f2eeb551-94e8-4ac2-a5de-4702e6349f42___JR_HL 8761__cfa294d88b90.jpg | Pepper,_bell___healthy | 0.198313 |
| Pepper,_bell___healthy/53572873-8205-48a9-81c0-6a60022c76ce___JR_HL 8550__f0219c116c26.jpg | Pepper,_bell___healthy | 0.205970 |
| Pepper,_bell___healthy/5d8fc784-788a-4a21-94e3-a96623a400fc___JR_HL 7915_newPixel25__4f4f37d9fd32.jpg | Pepper,_bell___healthy | 0.212161 |
| Tomato___healthy/2e875abf-377c-4663-a406-e02984215fa6___RS_HL 0499_180deg__ea48d2d1358d.jpg | Tomato___healthy | 0.234011 |
| Tomato___Target_Spot/aaad08ef-d75b-434a-83e2-d8b1cc519636___Com.G_TgS_FL 1109__7445036c9b69.jpg | Tomato___Target_Spot | 0.243225 |
| Tomato___healthy/f04f1e27-b6b0-43a6-b5c5-a4ec6d75ed3e___RS_HL 9737__b602a77994ae.jpg | Tomato___healthy | 0.255612 |
| Tomato___Target_Spot/c3f85772-c7c1-41f5-9fa1-3dacf50c96da___Com.G_TgS_FL 0754__a6025e68e29f.jpg | Tomato___Target_Spot | 0.257914 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/d5d5532f-7fbd-4e8c-9d72-66e3a46abf69___FAM_L.Blight 1662_flipLR__953cc681039a.jpg | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 0.259108 |
| Tomato___healthy/162db449-3361-40fa-a9bf-42bb63d43781___RS_HL 9647__40eb236f1f9a.jpg | Tomato___healthy | 0.267087 |

## Highest Per-Class Error Rates

| Class | Images | Mistakes | Error rate |
|---|---:|---:|---:|
| Cherry_(including_sour)___healthy | 684 | 684 | 1.000000 |
| Potato___Early_blight | 727 | 719 | 0.988996 |
| Strawberry___Leaf_scorch | 665 | 656 | 0.986466 |
| Soybean___healthy | 758 | 738 | 0.973615 |
| Tomato___Septoria_leaf_spot | 654 | 636 | 0.972477 |
| Pepper,_bell___Bacterial_spot | 717 | 697 | 0.972106 |
| Tomato___Tomato_mosaic_virus | 671 | 652 | 0.971684 |
| Tomato___Leaf_Mold | 705 | 682 | 0.967376 |
| Apple___Black_rot | 745 | 714 | 0.958389 |
| Tomato___Target_Spot | 685 | 646 | 0.943066 |

## Interpretation

This gallery is a targeted diagnostic for this checkpoint and evaluation condition. Repeated confusion pairs can identify classes that need closer data or model review, but the images may also expose labeling ambiguity, duplicated acquisition settings, or clean-background dataset artifacts. These results should not be interpreted as field performance without evaluation on independently collected field images.
