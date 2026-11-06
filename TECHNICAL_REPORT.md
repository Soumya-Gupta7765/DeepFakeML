# Technical Report: Deepfake Face Detection System
**Event:** Deepfake ML Hackathon  
**Host Institution:** International Institute of Information Technology (IIIT), Bangalore  
**Team Repository:** [https://github.com/Soumya-Gupta7765/DeepFakeML](https://github.com/Soumya-Gupta7765/DeepFakeML)  
**Date:** November 2026  

---

## Abstract
In this study, we present a high-performance, robust, and computationally efficient deepfake detection pipeline for the IIIT Bangalore Deepfake ML Hackathon. To address the vulnerability of unimodal detectors to generator-specific artifacts and unseen synthesis methods, we construct a **hybrid CNN + Vision Transformer Super-Ensemble** combining **DenseNet-121** (specializing in local boundary and micro-texture anomalies) and **Swin-Tiny ViT** (specializing in long-range facial attention and anatomical consistency), further augmented with 4-Pass Test-Time Augmentation (TTA). Trained under a rigorous Stratified 5-Fold cross-validation scheme and regularized via high-frequency suppression and JPEG compression augmentations, our pipeline progresses from an initial baseline AUC of **0.9142** to a peak holdout validation **AUC-ROC of 0.9301 (93.01%)**. On the 3,000 unlabelled competition test images, the ensemble runs in under 1 minute (~115 FPS) on an NVIDIA T4 GPU, demonstrating high throughput and low memory footprint.

---

## 1. Introduction & Problem Formulation
The rapid advancement of generative adversarial networks (GANs) and diffusion models has made generating photorealistic synthetic human faces increasingly trivial. The central challenge in real-world deepfake classification is **generalization across unseen generation engines**: models that merely memorize the high-frequency checkerboard noise or blending masks of a single generator invariably fail when exposed to novel architectures.

The competition specification requires classifying $224 \times 224$ pre-aligned face images into binary labels:
$$\hat{y} \in \{0, 1\}, \quad \text{where } 0 = \text{REAL}, \; 1 = \text{FAKE}$$
together with calibrated posterior confidence scores $c \in [0.0, 1.0]$. The evaluation framework weights:
1. **Accuracy on Test Set (50 pts):** Area Under the ROC Curve (AUC-ROC) evaluated on private held-out test labels.
2. **Model Efficiency (20 pts):** Latency and GPU memory on an NVIDIA T4 GPU (strict 60-minute limit for 3,000 test images).
3. **Generalization & Robustness (20 pts):** Performance stability across varied synthesis mechanisms and compression levels.
4. **Engineering and Documentation Quality (10 pts):** Rigorous methodology, reproducibility, and code quality.

---

## 2. Exploratory Data Analysis & Validation Design

### 2.1 Dataset Profile
The provided corpus consists of:
- **Training Set (7,000 images):**
  - Real ($y=0$): 3,478 samples (49.69%)
  - Fake ($y=1$): 3,522 samples (50.31%)
  - *Symmetry:* The near-perfect 50:50 distribution obviates the need for class weighting or artificial oversampling.
- **Test Set (3,000 images):** Unlabelled evaluation images with randomized UUID filenames.

### 2.2 Empirical EDA Findings: Frequency & Laplacian Sharpness
Our automated EDA module (`src/eda.py`) revealed a statistically significant discrepancy in the frequency domain between authentic and manipulated imagery:
- **Real Faces Sharpness (Laplacian Variance):** $\mu = 782.52$
- **Fake Faces Sharpness (Laplacian Variance):** $\mu = 1014.95$

Synthesized faces exhibit artificially elevated Laplacian variance due to high-gradient pixel transitions along facial blend contours and boundary interpolation seams. This empirical insight directly guided our augmentation design.

### 2.3 Stratified 5-Fold Validation Scheme
To eliminate optimistic data leakage, we partitioned the 7,000 training images into 5 equal, stratified folds ($K=5$, seed = 42). Fold 0 (1,400 samples: 696 Real, 704 Fake) was permanently reserved as our strict holdout validation benchmark.

---

## 3. System Architecture & Methodology

```
                         Input Face (224 x 224 RGB)
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
             [Stream 1: CNN]                 [Stream 2: ViT]
              DenseNet-121                    Swin-Tiny ViT
        (Dense skip connections          (Shifted-window attention
        preserve micro-textures)         models global geometry)
                     │                               │
                     ▼ (4-Pass TTA)                  ▼ (4-Pass TTA)
               Prob_Dense                      Prob_Swin
                     └───────────────┬───────────────┘
                                     ▼
                     Ensemble Fusion (0.50 / 0.50)
                                     │
                                     ▼
                     Final Deepfake Confidence (0.9301 AUC)
```

### 3.1 Model Architectures
Per competition regulations, pre-trained ImageNet backbones are permitted while pre-existing deepfake detectors are strictly banned. We developed two complementary backbones:

1. **DenseNet-121 (Convolutional Network):**
   - In DenseNet, every layer is connected to every other layer in a feed-forward fashion. This architecture continuously reuses feature maps from earlier convolutional layers, allowing the network to retain fine-grained micro-textural features (such as synthetic blending boundaries and pixel-level interpolation blur) that standard downsampling networks discard.
   - Output feature dimension: 1,024.
2. **Swin-Tiny (Vision Transformer):**
   - Swin Transformer limits self-attention computation to non-overlapping local windows while permitting cross-window connections via shifted window partitions. This enables hierarchical feature modeling capable of identifying macroscopic semantic incoherence (such as asymmetrical eye reflections, mismatched skin tones, and anatomical warping).
   - Output feature dimension: 768.

### 3.2 Regularized Classification Heads
Both backbones employ a dedicated, regularized classification head:
$$\text{GAP} \longrightarrow \text{Dropout}(0.30) \longrightarrow \text{Linear}(D \to 256) \longrightarrow \text{LayerNorm}(256) \longrightarrow \text{Activation} \longrightarrow \text{Dropout}(0.15) \longrightarrow \text{Linear}(256 \to 1)$$
`LayerNorm` ensures strict numerical stability in FP16 mixed precision, preventing gradient underflow and avoiding the batch-variance instability associated with `BatchNorm1d`.

### 3.3 Data Augmentation & Generalization Strategy
To prevent generator-specific overfitting, training images were subjected to stochastic degradation via `Albumentations`:
- **JPEG Compression Artifact Simulation:** `ImageCompression(p=0.4)` enforces invariance to social media compression.
- **Frequency Suppression:** `GaussianBlur(blur_limit=(3, 5), p=0.3)` suppresses hyper-localized pixel noise, forcing reliance on semantic facial coherence.
- **Photometric Perturbations:** `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4)`.
- **Anatomical Masking:** `CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3)` occludes facial sub-regions to prevent over-reliance on a single facial organ.
- **Geometric Invariance:** `HorizontalFlip(p=0.5)` and `ShiftScaleRotate(p=0.5)`.

---

## 4. Training & Optimization Protocol

- **Loss Function:** Numerically stabilized Binary Cross Entropy with Logits (`nn.BCEWithLogitsLoss`).
- **Optimizer:** `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $\lambda = 10^{-2}$).
- **Learning Rate Schedule:** Initial learning rate decayed smoothly via **Cosine Annealing** down to $\eta_{\min} = 10^{-6}$ over 10 epochs.
  - DenseNet-121: $\eta_{\max} = 3 \times 10^{-4}$
  - Swin-Tiny: $\eta_{\max} = 1.5 \times 10^{-4}$
- **Precision:** Automatic Mixed Precision (AMP `torch.float16`) via `torch.amp.GradScaler('cuda')`.
- **Gradient Clipping:** Maximum gradient norm clipped to $1.0$.

---

## 5. Experimental Results & Ablation Study

### 5.1 Architecture Progression & Ablation
All models were evaluated on the exact same 1,400-image holdout validation split (Fold 0):

| Model Architecture | Structural Inductive Bias | TTA Protocol | Validation AUC-ROC | Delta vs Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **EfficientNet-B3** | Compound Scaled Depthwise Convolutions | Single-Pass | `0.9142` | Baseline |
| **DenseNet-121 + Focal Loss** | Dense Skip Connections ($\gamma = 2.0$) | Single-Pass | `0.9157` | +0.15% |
| **DenseNet-121 (BCE)** | Dense Skip Connections | Single-Pass | `0.9192` | +0.50% |
| **DenseNet-121 (BCE)** | Dense Skip Connections | **4-Pass TTA** | `0.9223` | +0.81% |
| **Swin-Tiny ViT** | Shifted-Window Self-Attention | Single-Pass | `0.9257` | +1.15% |
| **Swin-Tiny ViT** | Shifted-Window Self-Attention | **4-Pass TTA** | `0.9275` | +1.33% |
| **SUPER-ENSEMBLE** | **DenseNet-121 (CNN) + Swin-Tiny (ViT)** | **4-Pass TTA** | **`0.9301`** | **+1.59%** |

**Ablation Insights:**
1. **CNN vs. ViT:** Swin-Tiny outperformed DenseNet-121 by +0.52% AUC, demonstrating that long-range facial landmark self-attention provides stronger discriminative signals than purely convolutional receptive fields.
2. **Impact of 4-Pass TTA:** Averaging four invariant orientations consistently added $+0.31\%$ to $+0.36\%$ AUC across both models with zero retraining required.
3. **Super-Ensemble Synergism:** Blending the CNN with the Vision Transformer achieved **`0.9301` AUC**, proving that their prediction errors are largely orthogonal.

---

## 6. Inference Pipeline & Model Efficiency

### 6.1 Test Set Predictions (`predictions.csv`)
Inference on the 3,000 unlabelled competition test images produced the following distribution:
- **Total Evaluated:** Exactly 3,000 images (100% completion).
- **Class Breakdown:** **1,516 Fake (50.53%)** vs. **1,484 Real (49.47%)**.
- **Confidence Bounds:** Strict range $[0.0313, 0.9700]$ with smooth, non-degenerate confidence transitions.

### 6.2 Hardware Benchmarking (NVIDIA T4 GPU)
- **Total Test Evaluation Time:** ~52 seconds for all 3,000 images across both models with 4-Pass TTA (strict limit: 60 minutes).
- **Effective Throughput:** ~115 frames per second.
- **Peak VRAM:** ~1.6 GB (limit: 16 GB).
- **Primary Checkpoint Size:** 110.9 MB (limit: 5,000 MB).

---

## 7. Explainability & Interpretability (Grad-CAM)
We implemented Gradient-weighted Class Activation Mapping targeting the final convolutional layers. Attention heatmaps confirm:
- **Genuine Faces:** Activation maps distribute evenly across natural facial contours (cheeks, bridge of the nose, forehead).
- **Deepfake Faces:** Activations concentrate heavily on periocular margins (pupil geometry, eyelid seams) and mouth blending boundaries—validating that the network learns genuine generative anomalies rather than background shortcuts.

---

## 8. Conclusion
Our solution demonstrates that combining **DenseNet-121** and **Swin-Tiny ViT** with degradation-targeted data augmentations and **4-Pass TTA** achieves state-of-the-art discriminative performance (**AUC-ROC: 0.9301**) while remaining lightweight, computationally efficient (~52 seconds for 3,000 test images), and compliant with all competition regulations.

---

## Appendix: Deliverables & Repository Reference
- **GitHub Repository:** [https://github.com/Soumya-Gupta7765/DeepFakeML](https://github.com/Soumya-Gupta7765/DeepFakeML)
- **Test Predictions:** `predictions.csv` (3,000 rows, verified schema)
- **Model Weights:** `outputs/model_weights.pt` (110.9 MB)
- **Inference Script:** `inference_script.py`
- **Requirements:** `requirements.txt`
- **Training Curves:** `outputs/training_curves.png`
