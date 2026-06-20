# Failure Analysis: resnet18_transfer_3ep

- Checkpoint: `/home/akshaj/Building/plant-disease-visionops/artifacts/models/resnet18_transfer_3ep/best_model.pt`
- Model: `resnet18`
- Evaluated split: `test`
- Corruption: `none (clean evaluation)`
- Severity: `n/a`
- Total images: 26353
- Total mistakes: 425
- Error rate: 0.016127
- Misclassification grid: `/home/akshaj/Building/plant-disease-visionops/artifacts/figures/resnet18_transfer_3ep_failures_clean.png`

## Top True Labels by Mistake Count

| Label | Mistakes |
|---|---:|
| Tomato___Target_Spot | 51 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 41 |
| Blueberry___healthy | 33 |
| Tomato___Late_blight | 28 |
| Corn_(maize)___Northern_Leaf_Blight | 27 |
| Apple___Cedar_apple_rust | 26 |
| Tomato___Spider_mites Two-spotted_spider_mite | 24 |
| Tomato___Early_blight | 21 |
| Tomato___Septoria_leaf_spot | 19 |
| Grape___Black_rot | 17 |

## Top Predicted Labels in Mistakes

| Label | Mistakes |
|---|---:|
| Tomato___Early_blight | 79 |
| Potato___healthy | 29 |
| Corn_(maize)___healthy | 29 |
| Pepper,_bell___healthy | 28 |
| Apple___Apple_scab | 25 |
| Tomato___Bacterial_spot | 24 |
| Corn_(maize)___Northern_Leaf_Blight | 20 |
| Raspberry___healthy | 19 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 19 |
| Tomato___Late_blight | 16 |

## Most Common Confusions

| True label | Predicted label | Mistakes |
|---|---|---:|
| Tomato___Target_Spot | Tomato___Early_blight | 21 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn_(maize)___Northern_Leaf_Blight | 20 |
| Apple___Cedar_apple_rust | Apple___Apple_scab | 19 |
| Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 13 |
| Blueberry___healthy | Potato___healthy | 12 |
| Grape___Black_rot | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 12 |
| Tomato___Late_blight | Potato___Late_blight | 12 |
| Tomato___Spider_mites Two-spotted_spider_mite | Tomato___Target_Spot | 12 |
| Tomato___Septoria_leaf_spot | Tomato___Early_blight | 11 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn_(maize)___healthy | 10 |

## Highest-Confidence Wrong Predictions

| Filepath | True label | Predicted label | Confidence | True probability |
|---|---|---|---:|---:|
| Blueberry___healthy/ca4ea722-9bf0-4094-bf8b-e15091bab904___RS_HL 0620__d341c8c6a27e.jpg | Blueberry___healthy | Tomato___Late_blight | 0.999994 | 0.000000 |
| Apple___Cedar_apple_rust/87ea6ae6-eaf6-468d-85cd-cded92ba526f___FREC_C.Rust 9963_90deg__04216762845a.jpg | Apple___Cedar_apple_rust | Apple___Apple_scab | 0.999990 | 0.000010 |
| Apple___Cedar_apple_rust/87ea6ae6-eaf6-468d-85cd-cded92ba526f___FREC_C.Rust 9963_90deg__859d01f5c3a2.jpg | Apple___Cedar_apple_rust | Apple___Apple_scab | 0.999990 | 0.000010 |
| Apple___Cedar_apple_rust/6fa42fea-4fcd-4cfe-bfbd-c03ee2d7bb3f___FREC_C.Rust 0163_new30degFlipTB__a75fce312dff.jpg | Apple___Cedar_apple_rust | Apple___Apple_scab | 0.999975 | 0.000025 |
| Tomato___Early_blight/0f111718-a749-42cf-9495-412f4b51acf0___RS_Erly.B 7482__62d64dc01526.jpg | Tomato___Early_blight | Tomato___Bacterial_spot | 0.999688 | 0.000045 |
| Potato___Late_blight/e9cc84b7-fc38-4e65-a129-c7a63df76780___RS_LB 2705__901e6edc6098.jpg | Potato___Late_blight | Potato___healthy | 0.999236 | 0.000764 |
| Strawberry___healthy/2fbe071f-efd0-4a93-97b5-38160dafa0c5___RS_HL 2146_90deg__190c9bd419aa.jpg | Strawberry___healthy | Raspberry___healthy | 0.999113 | 0.000887 |
| Orange___Haunglongbing_(Citrus_greening)/26454175-626b-436b-b59c-396872381437___UF.Citrus_HLB_Lab 0691__17a165d3d516.jpg | Orange___Haunglongbing_(Citrus_greening) | Corn_(maize)___healthy | 0.999109 | 0.000649 |
| Corn_(maize)___Northern_Leaf_Blight/c4eb96b2-1345-46d0-a796-161042da21df___RS_NLB 0824 copy_180deg__71ec1b247ee2.jpg | Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___healthy | 0.998896 | 0.000923 |
| Grape___Black_rot/a5b36885-7c84-4913-a69e-d0e881f18f8b___FAM_B.Rot 0621_flipLR__4ddcf63adaee.jpg | Grape___Black_rot | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 0.998526 | 0.001473 |

## Lowest-Confidence Correct Predictions

| Filepath | Label | Confidence |
|---|---|---:|
| Cherry_(including_sour)___Powdery_mildew/61705ba4-f43e-48f6-afd2-d62fe6f3922e___FREC_Pwd.M 5139_flipLR__fc41fff4a2f4.jpg | Cherry_(including_sour)___Powdery_mildew | 0.321813 |
| Peach___Bacterial_spot/dda063cf-1be6-4baf-825f-538fc3d6e606___Rut._Bact.S 3471__c037f1199f06.jpg | Peach___Bacterial_spot | 0.356601 |
| Blueberry___healthy/0d8e77c4-1930-456c-b429-e8774653a8cd___RS_HL 2272_180deg__b6ccdcb623af.jpg | Blueberry___healthy | 0.384674 |
| Tomato___Late_blight/6dbfb834-5654-4526-8014-2b33ef976892___RS_Late.B 6172_flipLR__bc45c9a97669.jpg | Tomato___Late_blight | 0.410149 |
| Tomato___Leaf_Mold/2d6f98ac-7545-4ea6-97f7-51374d16adc2___Crnl_L.Mold 7005__06ba2f48431f.jpg | Tomato___Leaf_Mold | 0.416965 |
| Tomato___Target_Spot/4ab417ba-7f9c-44ee-a7d1-1cd78fe68b58___Com.G_TgS_FL 0046_180deg__15962b18baa7.jpg | Tomato___Target_Spot | 0.433562 |
| Tomato___Late_blight/f37f0878-31f2-402b-81c6-dd1af6e15899___RS_Late.B 5222__371312514c0b.jpg | Tomato___Late_blight | 0.451305 |
| Tomato___Septoria_leaf_spot/e94aaf1a-0c1b-4020-87f0-93c74f929f1c___JR_Sept.L.S 8496__b62a1f7e2ca0.jpg | Tomato___Septoria_leaf_spot | 0.456437 |
| Pepper,_bell___Bacterial_spot/ef02f719-ca79-481f-9429-365fa1685a80___JR_B.Spot 9056_flipTB__7a0e40bbe5c6.jpg | Pepper,_bell___Bacterial_spot | 0.459743 |
| Pepper,_bell___Bacterial_spot/ef02f719-ca79-481f-9429-365fa1685a80___JR_B.Spot 9056_flipTB__b90c4e43ff36.jpg | Pepper,_bell___Bacterial_spot | 0.459743 |

## Highest Per-Class Error Rates

| Class | Images | Mistakes | Error rate |
|---|---:|---:|---:|
| Tomato___Target_Spot | 685 | 51 | 0.074453 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 615 | 41 | 0.066667 |
| Blueberry___healthy | 681 | 33 | 0.048458 |
| Tomato___Late_blight | 694 | 28 | 0.040346 |
| Apple___Cedar_apple_rust | 660 | 26 | 0.039394 |
| Corn_(maize)___Northern_Leaf_Blight | 715 | 27 | 0.037762 |
| Tomato___Spider_mites Two-spotted_spider_mite | 653 | 24 | 0.036753 |
| Tomato___Early_blight | 720 | 21 | 0.029167 |
| Tomato___Septoria_leaf_spot | 654 | 19 | 0.029052 |
| Grape___Black_rot | 708 | 17 | 0.024011 |

## Interpretation

This gallery is a targeted diagnostic for this checkpoint and evaluation condition. Repeated confusion pairs can identify classes that need closer data or model review, but the images may also expose labeling ambiguity, duplicated acquisition settings, or clean-background dataset artifacts. These results should not be interpreted as field performance without evaluation on independently collected field images.
