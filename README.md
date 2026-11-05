# Deepfake Face Detection System
### International Institute of Information Technology (IIIT), Bangalore — Deepfake ML Hackathon

---

## 1. Executive Summary & Problem Formulation

The objective of this challenge is to build a high-performance, robust, and computationally efficient binary classification system capable of distinguishing authentic face photographs (`label = 0`, REAL) from synthetically generated or manipulated face imagery (`label = 1`, FAKE).



## 2. Dataset Architecture & Validation Strategy

### 2.1 Dataset Statistics
- **Input Dimensions**: $224 \times 224$ pixels, 3-channel RGB JPEG format.
- **Pre-processing**: Pre-aligned and face-centered.
- **Training Set (7,000 images)**:
  - **Real (`0`)**: 3,478 images (~49.69%)
  - **Fake (`1`)**: 3,522 images (~50.31%)
  - *Status*: Virtually balanced; no extreme class weighting required.
- **Test Set (3,000 images)**:
  - Random UUID filenames with zero ground-truth label leaks.

### 2.2 Stratified Validation Strategy
To avoid data leakage and prevent optimistic bias from non-representative splits, a **Stratified 5-Fold Cross-Validation** scheme (`random_state = 42`) is implemented in `src/prepare_folds.py`, generating `hackathon_dataset/train_folds.csv`.

| Fold | Real (`0`) | Fake (`1`) | Total Samples | Role in Single-Fold Mode |
| :--- | :---: | :---: | :---: | :--- |
| **Fold 0** | 696 | 704 | **1,400** | Validation Set (20%) |
| **Fold 1** | 696 | 704 | **1,400** | Training Set (80%) |
| **Fold 2** | 696 | 704 | **1,400** | Training Set (80%) |
| **Fold 3** | 695 | 705 | **1,400** | Training Set (80%) |
| **Fold 4** | 695 | 705 | **1,400** | Training Set (80%) |
| **Total** | **3,478** | **3,522** | **7,000** | Complete Corpus |

Each record in `train_folds.csv` contains `image_id`, `label`, `file_path`, and `fold` for $O(1)$ batch index resolution.

---

## 3. Model Architecture & Selection Rationale

### 3.1 Backbone Architecture
Per hackathon regulations, pre-trained ImageNet backbones are permitted, but deepfake-specific pre-trained detectors are strictly prohibited.

We adopt **EfficientNet-B3** (primary candidate) and **ConvNeXt-Tiny** (alternative/ensemble candidate):

| Criteria | EfficientNet-B3 | ConvNeXt-Tiny |
| :--- | :--- | :--- |
| **ImageNet Weights** | Yes (`timm`) | Yes (`timm`) |
| **Input Resolution** | $224 \times 224$ (native training: $300 \times 300$, downscaled cleanly to $224 \times 224$) | $224 \times 224$ |
| **Parameters** | ~12.2 M | ~28.6 M |
| **T4 GPU Inference Throughput** | ~350+ img/sec (FP16) | ~280+ img/sec (FP16) |
| **Inductive Bias** | Depthwise separable convolutions, Compound Scaling | Modernized 7x7 depthwise convolutions, LayerNorm |
| **Suitability** | Captures subtle high-frequency edge/texture artifacts | Preserves spatial coherence across facial landmarks |

### 3.2 Classification Head Design
To prevent overfitting on superficial generator artifacts:
```text
Backbone Feature Extractor (e.g., EfficientNet-B3)
                   │
                   ▼
       Global Average Pooling (GAP)
                   │
                   ▼
             Dropout (p=0.3)
                   │
                   ▼
      Linear Projection (Features -> 256)
                   │
                   ▼
            SiLU Activation
                   │
                   ▼
             Dropout (p=0.2)
                   │
                   ▼
           Linear Output (256 -> 1)
```
- **Output**: Single raw logit $z$.
- **Inference Probability**: $\sigma(z) = \frac{1}{1 + e^{-z}} \in [0.0, 1.0]$.

---

## 4. Data Augmentation & Generalization Strategy

Deepfake models commonly overfit to specific artifact signatures (e.g., distinct blending boundaries, specific GAN grid artifacts, or compression patterns). To enforce generalization across diverse synthesis engines, we implement a robust augmentation pipeline via `albumentations`:

1. **Spatial & Geometric Transformations**:
   - `HorizontalFlip(p=0.5)`
   - `ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5)`
2. **Frequency & Blur Degradation** (Destroys fragile generator artifacts):
   - `GaussianBlur(blur_limit=(3, 5), p=0.3)`
   - `ImageCompression(quality_lower=60, quality_upper=100, p=0.4)` (Simulates diverse social media compression)
3. **Photometric Adjustments**:
   - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4)`
   - `CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3)` (Forces holistic facial inspection)
4. **Validation/Test Transforms**:
   - Clean normalization to standard ImageNet statistics: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.

---

## 5. Training Pipeline & Optimization

### 5.1 Training Specifications
- **Objective Loss**: Binary Cross Entropy with Logits (`torch.nn.BCEWithLogitsLoss`).
  - Numerically stable combining sigmoid + cross-entropy.
  - Optional label smoothing ($\epsilon = 0.05$) to prevent overconfident boundary predictions.
- **Optimizer**: `AdamW` (weight decay $\lambda = 10^{-2}$).
- **Learning Rate**: Initial $lr = 3 \times 10^{-4}$ with a **Cosine Annealing Scheduler** with 1-epoch linear warmup.
- **Batch Size**: 32 or 64 on NVIDIA T4 GPU.
- **Mixed Precision**: Automatic Mixed Precision (`torch.cuda.amp.autocast(dtype=torch.float16)`) to halve GPU memory and maximize throughput.
- **Checkpointing**: Automatic serialization of the best model based on validation **AUC-ROC**.

### 5.2 Metrics Monitored
- **Primary**: Validation AUC-ROC (`sklearn.metrics.roc_auc_score`).
- **Secondary**: F1-Score, Precision, Recall, Accuracy across thresholds $\tau \in [0.1, 0.9]$.

---

## 6. Inference Pipeline (`inference_script.py`) Specification

The inference pipeline strictly adheres to the hackathon CLI requirements:

```bash
python inference_script.py \
  --test_dir ./hackathon_dataset/test/images/ \
  --weights ./outputs/best_model.pt \
  --output predictions.csv
```

### Execution Rules:
- Dynamic path discovery without hardcoded directories.
- Batched inference utilizing `torch.inference_mode()` and FP16 where supported.
- Exact column formatting:
  ```csv
  image_id,label,confidence
  367711e7-e4c2-4a2e-b6ab-1f9d2726fbcd,0,0.1342
  3f9a2c1d-88b1-4209-8b83-9b93081e6b91,1,0.9124
  ```
- Guaranteed 1-to-1 row correspondence for all 3,000 test images.

---

## 7. Kaggle GPU Workflow

Because training on 7,000 images is accelerated by GPU compute, training can be executed seamlessly on Kaggle (NVIDIA T4 x2):

1. **Upload / Archive**:
   - Pack `hackathon_dataset/` and `src/` into a Kaggle dataset or zip archive.
2. **Execute Training**:
   ```python
   !pip install -r requirements.txt
   !python src/train.py --fold 0 --epochs 10 --batch_size 32 --model_name efficientnet_b3
   ```
3. **Artifact Retrieval**:
   - Download `best_model.pt`, training history `history.csv`, and validation predictions.

---

## 8. Project Structure

```text
deepFakeMlHackathon/
├── hackathon_dataset/
│   ├── train/
│   │   ├── real/             # Label = 0
│   │   └── fake/             # Label = 1
│   ├── test/
│   │   └── images/           # 3,000 unlabelled test images
│   ├── train_metadata.csv    # Source metadata
│   ├── sample_submission.csv # Template submission
│   └── train_folds.csv       # Pre-computed 5-Fold stratified splits
├── src/
│   ├── __init__.py
│   ├── prepare_folds.py      # Stratified 5-Fold generator
│   ├── dataset.py            # Dataset class and augmentations (Phase 2)
│   ├── model.py              # Architecture definitions (Phase 2)
│   ├── train.py              # Training loop, scheduler, and validation (Phase 3)
│   └── utils.py              # Metrics, logging, and seed controls
├── outputs/                  # Model weights, logs, loss curves
├── inference_script.py       # Required CLI inference script (Phase 4)
├── requirements.txt          # Reproducible environment specifications
└── README.md                 # Complete system documentation
```

---

## 9. Deliverables & Compliance Checklist

| Item | Requirement | Target File / Format | Status |
| :---: | :--- | :--- | :---: |
| **1** | Test Predictions | `predictions.csv` (3,000 rows matching schema) | Pending inference |
| **2** | Model Weights | `model_weights.pt` (< 5 GB) | Generated in training |
| **3** | Inference Script | `inference_script.py` (CLI arguments `--test_dir`, `--weights`, `--output`) | Planned |
| **4** | Dependencies | `requirements.txt` (Python 3.10 compatible) | **Completed** |
| **5** | Source Code | Zipped project repository with structured code | **Structured** |
| **6** | Technical Report | Markdown & PDF report (min. 4 pages) | Structure outlined |
| **7** | Training Logs | CSV curves & loss plots | Tracked in training |
| **8** | Explainability | Grad-CAM attention visualizations | Bonus deliverable |

