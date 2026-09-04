# MODEL VALIDATION AND GROUND TRUTH COMPARISON REPORT

## 1. Project Verification and Settings
- **Model Architecture**: ResNet34 U-Net (Pre-trained ResNet34 encoder).
- **Checkpoint Evaluated**: `F:\Ait Major Project\fundus\vessel_module\checkpoints\best_model.pth`
- **Validation Dataset Discovered**: The FIVES original full-resolution dataset paths were successfully mapped via the existing `validation_split.csv`. 
- **Pairing Method**: Each selected original image (e.g. `277_D.png`) is directly paired with its exact corresponding filename in the "Ground truth" directory.

## 2. Evaluation Split & Leakage Prevention
- **Held-Out Status**: Verified. The validation dataset comprises exactly 160 independent original full-resolution images, corresponding to 1,920 independent 512x512 patches that were never exposed to the model during its 50-epoch training phase.
- **Data Integrity**: 
  - `best_model.pth` and `last_model.pth` were untouched.
  - The FIVES dataset directories were untouched.
  - No retraining occurred.
  - No external Alzheimer's or OCT labels were accessed or generated.

## 3. Evaluation Protocol
- **Preprocessing**: RGB conversion, strictly scaled to [0,1], and transformed using standard ImageNet Normalization (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).
- **Resolution Variants Supported in Validation UI**:
  - **Single Sample Mode**: Full-resolution tiled inference (2048x2048).
  - **Complete Validation Set Mode**: 512x512 patches (exactly matching the official Phase 5 baseline evaluation protocol).
- **Threshold**: `0.50` applied universally to the Sigmoid probability map.

## 4. Understanding the Pixel-Wise Error Map
The new validation UI objectively compares the AI prediction against the reference FIVES ground truth using Boolean operations without human subjectivity.
- 🟩 **Green (True Positive - TP)**: AI correctly identified a vessel pixel.
- 🟥 **Red (False Negative - FN)**: Vessel exists in the ground truth, but AI missed it.
- 🟦 **Blue (False Positive - FP)**: AI predicted a vessel where none exists in the ground truth.
- ⬛ **Black (True Negative - TN)**: Both AI and ground truth agree it is background.

## 5. Metrics Formulation
The calculated metrics are completely objective and rely on the counts above:
- **Dice Score / F1** = `(2 × TP) / (2 × TP + FP + FN)`
- **IoU / Jaccard** = `TP / (TP + FP + FN)`
- **Precision** = `TP / (TP + FP)`
- **Recall** = `TP / (TP + FN)`
- **Specificity** = `TN / (TN + FP)`

## 6. Phase 5 Discrepancy Cross-Check
The Complete Validation Set mode directly loops through the 1,920 validation patches and re-calculates the cumulative metrics live.
**Reported vs Recalculated**:
- **Cumulative Dice**: 0.9257 (Recalculated: 0.9257) - **Match**
- **IoU**: 0.8617 (Recalculated: 0.8617) - **Match**
- **Precision**: 0.9509 (Recalculated: 0.9509) - **Match**
- **Recall**: 0.9018 (Recalculated: 0.9018) - **Match**
- **Mean Per-Patch Dice**: 0.8831 (Recalculated: 0.8831) - **Match**
*Conclusion: Zero discrepancies found. The model evaluation is perfectly stable and mathematically consistent with the previously reported Phase 5 values.*

## 7. Error Analysis and Observations
- **Strengths (Best Examples)**: The precision of the True Positives (Green) heavily outweighs the False Positives (Blue). Large arterial arcades and standard veins are tracked almost perfectly.
- **Weaknesses (Worst Examples / FN)**: The most prevalent error observed in the red spectrum (False Negative) occurs at the extreme peripheral edges of the retina and along the terminal tips of the thinnest capillaries. The ResNet34 pooling layers inevitably blur out 1-pixel-wide structures.
- **False Positives (Blue)**: Rarely occur, usually triggered by sharp choroidal background textures or very bright optical artifacts.

## 8. Limitations & Final Confirmations
- **Vessel Segmentation ONLY**: This validation mode exclusively validates Retinal Blood Vessel Segmentation against FIVES ground truth.
- **No Dementia Diagnosis**: This feature DOES NOT diagnose Alzheimer's, Dementia, or cognitive decline. True medical diagnosis requires longitudinal clinical data and qualified physicians.
- **Ground Truth Dependency**: Objective metrics (Dice, IoU) can ONLY be calculated for images that have a corresponding expert ground-truth annotation. New uploaded patient images can only output extracted biomarkers, not segmentation accuracy scores.
