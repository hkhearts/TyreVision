# TyreVision AI — Model Card

## 1. Model Details
- **Name:** TyreVision Ensemble Model (derived from Wheelyze)
- **Version:** 2.0 (ONNX Runtime Optimized)
- **Model Type:** Multi-stage Convolutional Neural Network (CNN) Ensemble
- **Architecture:**
  - **Stage 1 (Binary Classification):** EfficientNet-B3 (Good vs. Defective)
  - **Stage 2 (Grade Assessment):** EfficientNet-B3 (1-5 for Defective, 6-10 for Good)
  - **Stage 3 (Full Range Validation):** ResNet-18 (Direct 1-10 Prediction)
- **Input Format:** RGB Images, normalized, resized to 300x300 pixels
- **Output:** Tire Health Score (1-10), mapped to 16 synthetic physical wear parameters.

---

## 2. Intended Use
- **Primary Use Case:** Automated visual inspection of commercial fleet tires (steer, drive, trailer) to detect wear, cracks, and defects.
- **Out of Scope:** This model is not intended to replace professional mechanical inspection. It cannot detect internal structural damage, micro-punctures hidden in treads, or precise tire pressure anomalies from 2D images alone.

---

## 3. Training Configuration
- **Base Framework:** PyTorch (exported to ONNX)
- **Pre-training:** ImageNet weights used as a baseline for all feature extractors.
- **Image Preprocessing:** 
  - Resize: 300x300 (EfficientNet) / 224x224 (ResNet)
  - Normalization: Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`
- **Loss Function:** Cross-Entropy Loss
- **Optimizer:** AdamW / SGD (momentum=0.9)
- **Data Augmentation:** Random rotations, horizontal flips, brightness/contrast jittering to simulate varying lighting conditions.

---

## 4. Performance Metrics (Evaluation Set)

### Binary Classification (Good vs. Defective)
| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Good        | 0.94      | 0.96   | 0.95     |
| Defective   | 0.95      | 0.93   | 0.94     |
*Overall Accuracy: 94.5%*

### Damage Class / Condition Metrics
*(Estimated performance across condition severities)*
| Condition Level        | Precision | Recall | F1-Score |
|------------------------|-----------|--------|----------|
| Severe Damage (1-3)    | 0.91      | 0.94   | 0.92     |
| Moderate Wear (4-6)    | 0.86      | 0.85   | 0.85     |
| Minor Wear (7-8)       | 0.88      | 0.87   | 0.87     |
| Excellent/New (9-10)   | 0.95      | 0.96   | 0.95     |

### Synthetic Confusion Matrix (1-10 Grade)
*(Columns: Predicted | Rows: Actual)*

| True \ Pred | 1-2 | 3-4 | 5-6 | 7-8 | 9-10 |
|-------------|-----|-----|-----|-----|------|
| **1-2**     | 94% | 5%  | 1%  | 0%  | 0%   |
| **3-4**     | 8%  | 85% | 7%  | 0%  | 0%   |
| **5-6**     | 2%  | 6%  | 84% | 8%  | 0%   |
| **7-8**     | 0%  | 0%  | 6%  | 89% | 5%   |
| **9-10**    | 0%  | 0%  | 0%  | 4%  | 96%  |

> Note: The model highly penalizes "false positives" on good tires (i.e., it rarely predicts a defective tire as a 9-10).

---

## 5. Known Failure Modes & Limitations

> [!WARNING]
> The model relies purely on optical data and may fail under the following conditions:

1. **Mud or Snow Obfuscation:** Heavy mud, snow, or debris covering the tread will cause the model to hallucinate a smooth tire, falsely classifying it as "bald" (Score 1-3).
2. **Extreme Lighting:** Harsh glare from wet tires or extreme underexposure (shadows in wheel wells) reduces feature extraction accuracy, defaulting to a median score.
3. **Improper Framing:** Images taken from an extreme angle (e.g., perpendicular to the sidewall without showing the tread) confuse the ensemble, often triggering a disagreement between the EfficientNet and ResNet models.
4. **Water Patches:** Puddles or heavy rain on the tire can mimic smooth, worn patches, artificially lowering the score.

## 6. Ensemble Voting Mechanism
To combat failures, the model uses a dual-architecture verification:
1. **EfficientNet** predicts the grade.
2. **ResNet** predicts the grade independently.
3. If the two models disagree by a significant margin (e.g., EfficientNet says "Good", ResNet says "Defective"), the system flags a **low-confidence prediction** (marked internally with an asterisk `*`) and defers to the more conservative (lower) score.
