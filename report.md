# Deepfake Face Detection System — Technical Summary Report
### International Institute of Information Technology (IIIT), Bangalore — Deepfake ML Hackathon
**Team Members:** Soumya Gupta, Dhruv Reddy Sambavaram | **Repository:** [github.com/Soumya-Gupta7765/DeepFakeML](https://github.com/Soumya-Gupta7765/DeepFakeML)  
**Primary Architecture:** Dual-Model Super-Ensemble (**DenseNet-121 CNN** + **Swin-Tiny ViT**) with **8-Pass Multi-Scale TTA** & **Rank Averaging**  
**Benchmark Validation AUC-ROC:** **`0.9355` (93.55%)** | **Inference Runtime on NVIDIA T4 GPU:** **~70 seconds** (Official Limit: < 5 mins)

---

## 1. System Architecture & Inductive Biases
Deepfake facial manipulation introduces spatial artifacts across two distinct physical scales that no single architecture can simultaneously resolve:
1. **Local Boundary & Micro-Texture Scale (CNN):** Blending seams, interpolation blur along eyelid margins, and checkerboard deconvolution artifacts are captured by **DenseNet-121** (1,024-dim feature reuse). Dense connections allow earlier high-frequency spatial gradients to bypass successive pooling layers and persist directly into the final classification head.
2. **Global Anatomical & Illumination Scale (Vision Transformer):** Synthesized faces exhibit semantic inconsistencies (asymmetric cheek lighting, corneal reflection mismatch, unnatural ear-to-nose structural proportions). **Swin-Tiny** (`swin_tiny_patch4_window7_224`, 768-dim) models long-range cross-facial dependencies via shifted local window self-attention.

```
                           Input Image (224 × 224 RGB)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
         [Stream 1: Local CNN]                [Stream 2: Global ViT]
             DenseNet-121                          Swin-Tiny ViT
        (Micro-textures & seams)              (Global lighting & geometry)
                    │                                     │
                    ▼ (8-Pass Multi-Scale TTA)            ▼ (8-Pass Multi-Scale TTA)
               Prob_Dense                            Prob_Swin
                    └──────────────────┬──────────────────┘
                                       ▼
                     Uniform Percentile Rank Averaging
                         0.50 · r_swin + 0.50 · r_dense
                                       │
                                       ▼
                     Final Deepfake Confidence (0.9355 AUC)
```

**Regularized Projection Heads:** Both backbones route features through a custom two-stage head:  
`GAP ──► Dropout(0.30) ──► Linear(D ──► 256) ──► LayerNorm(256) ──► Activation (GELU/SiLU) ──► Dropout(0.15) ──► Linear(256 ──► 1)`.  
`LayerNorm(256)` ensures batch-size invariant inference stability.

---

## 2. Data Augmentations & 8-Pass Test-Time Augmentation (TTA)

### 2.1 Training Degradation Pipeline (`albumentations`)
To eliminate generator overfitting (memorizing specific synthesis signatures), we applied a multi-stage degradation protocol:
* **Compression Resilience:** `ImageCompression(quality_lower=60, quality_upper=100, p=0.4)` prevents social media recompression failures.
* **Frequency Suppression:** `GaussianBlur(blur_limit=(3, 5), p=0.3)` suppresses pixel noise, forcing reliance on semantic facial structure.
* **Photometric Invariance:** `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4)` simulates lighting diversity.
* **Occlusion Resistance:** `CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3)` randomly masks facial sub-regions.
* **Spatial Invariance:** `HorizontalFlip(p=0.5)` and `ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5)`.

### 2.2 8-Pass Multi-Scale Test-Time Augmentation (TTA)
During inference, each test face is evaluated across eight invariant orientations and scales:
1. Original image | 2. Horizontal flip (`torch.flip(x, [-1])`) | 3. Brightness boost (+5%) | 4. Flip + Brightness | 5. Contrast boost (+8%) | 6. Flip + Contrast | 7. Center Zoom ($1.06\times$ bilinear interpolation) | 8. Flip + Zoom.  
Multi-scale TTA smooths the decision field and boosted DenseNet standalone AUC from `0.9192` to `0.9251`.

---

## 3. Validation Strategy, Generalization & Ensembling

### 3.1 Leakage-Free Validation Protocol
* **Stratified 5-Fold Partitioning ($K=5$, seed=42):** Fold 0 (1,400 images: 696 Real, 704 Fake) was permanently reserved as a clean, untouched holdout validation benchmark during architecture exploration and hyperparameter tuning.
* **Stress-Test Robustness Suite:** Candidates were benchmarked under severe unseen corruptions: Gaussian noise ($\sigma=5\%$), JPEG compression ($Q=40$), Gaussian blur (radius 2), and center-crop zooming ($1.10\times$).

### 3.2 Percentile Rank Averaging Fusion
Because CNNs and ViTs produce differing logit calibration distributions, raw probability averaging creates calibration skew. We convert test predictions into uniform percentile rankings:
$$r_{\text{swin}} = \frac{\text{Rank}(p_{\text{swin}})}{N}, \quad r_{\text{dense}} = \frac{\text{Rank}(p_{\text{dense}})}{N}, \quad \text{Confidence} = 0.50 \cdot r_{\text{swin}} + 0.50 \cdot r_{\text{dense}}$$
Rank fusion boosted validation AUC-ROC from `0.9341` (linear blend) to **`0.9355`** (+2.13% over baseline).

---

## 4. Final Performance, Test Predictions & Efficiency Scorecard

### 4.1 Cross-Validation Benchmark Comparison
| Architecture | Type | Inference Technique | Validation AUC-ROC | Lift vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **EfficientNet-B3** | Single CNN | Single-Pass | `0.9142` (91.42%) | Baseline |
| **DenseNet-121** | Single CNN | Single-Pass / 8-Pass TTA | `0.9192` / `0.9251` | +1.09% |
| **Swin-Tiny (ViT)** | Vision Transformer | Single-Pass / 8-Pass TTA | `0.9267` / `0.9262` | +1.20% |
| **Linear Ensemble Blend** | CNN + ViT Hybrid | 8-Pass Multi-Scale TTA | `0.9341` (93.41%) | +1.99% |
| **RANK-AVERAGED ENSEMBLE** | **Percentile Rank Fusion** | **8-Pass Multi-Scale TTA** | **`0.9355` (93.55%)** | **+2.13%** |

### 4.2 Full-Dataset Retraining (100% Data) & Test Predictions (`predictions.csv`)
After validating hyperparameters, both models were retrained on 100% of labeled training data (7,000 images, 10 epochs, Cosine Annealing) using [`notebooks/01_train_full_dataset_final.py`](notebooks/01_train_full_dataset_final.py) to generate `predictions.csv`:
* **Row Count:** Exactly 3,000 test predictions matching `sample_submission.csv` UUIDs.
* **Class Split:** **1,503 Fake (50.10%)** vs. **1,497 Real (49.90%)** | **Confidence Bounds:** `[0.0010, 0.9990]`.
* **Inference Reproducibility:** 3,000 / 3,000 exact label match (100.00% reproduction agreement, zero penalty).

### 4.3 Efficiency Measurement on NVIDIA T4 GPU (Full 20/20 Points)
* **Full Test Set Runtime:** **~70 seconds** (Official Limit: < 5 minutes for full 20 pts).
* **Throughput:** ~120 images/second | **Peak GPU VRAM:** **~1.6 GB** (10% of 16 GB hardware budget).
* **Bundled Checkpoint Size (`model_weights.pt`):** **133.9 MB** (Well within 5,000 MB limit).
* **Explainability Bonus (+2 pts):** Grad-CAM heatmaps verify intense activations on periocular and mouth blending seams on fakes, with zero background shortcut reliance (see `outputs/gradcam_visualizations.png` and full 11-page `TECHNICAL_REPORT.pdf`).
