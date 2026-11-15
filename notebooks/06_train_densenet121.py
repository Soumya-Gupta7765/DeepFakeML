# ==============================================================================
# Model 3: DenseNet-121 Training & Auto-Ensemble Script for Kaggle GPU
# ==============================================================================
# DenseNet-121 uses dense connections where each layer receives feature maps
# from all preceding layers. This preserves fine micro-textural details.
# This script:
#   1. Trains DenseNet-121 on Fold 0 (directly comparable to EfficientNet-B3).
#   2. Tracks validation AUC-ROC and saves best weights to 'outputs/densenet_weights.pt'.
#   3. Runs inference on test images -> 'predictions_densenet.csv'.
#   4. If 'predictions.csv' (EfficientNet) exists, blends both into 'predictions_blended.csv'.

import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm", "albumentations", "scikit-learn"])

import os
import time
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

# 1. Reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)

# 2. Locate Dataset
DATASET_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train_metadata.csv" in files:
        DATASET_DIR = root
        break

if DATASET_DIR is None:
    raise FileNotFoundError("Could not find train_metadata.csv in /kaggle/input")

print(f"Dataset found at: {DATASET_DIR}")
train_meta_path = os.path.join(DATASET_DIR, "train_metadata.csv")
df = pd.read_csv(train_meta_path)

file_paths = []
for _, row in df.iterrows():
    sub = "real" if row["label"] == 0 else "fake"
    file_paths.append(os.path.join(DATASET_DIR, "train", sub, f"{row['image_id']}.jpg"))
df["file_path"] = file_paths

# Stratified 5-Fold split (Fold 0 = Validation)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
    df.loc[val_idx, "fold"] = fold

train_df = df[df["fold"] != 0].reset_index(drop=True)
val_df = df[df["fold"] == 0].reset_index(drop=True)
print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

# 3. Augmentations
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.ImageCompression(p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.3),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

valid_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

class DeepfakeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row["file_path"])
        if image is None:
            image = np.array(Image.open(row["file_path"]).convert("RGB"))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        label = torch.tensor([float(row["label"])], dtype=torch.float32)
        return image, label, str(row["image_id"])

train_loader = DataLoader(DeepfakeDataset(train_df, train_transforms), batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(DeepfakeDataset(val_df, valid_transforms), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# 4. DenseNet-121 Architecture
class DenseNetClassifier(nn.Module):
    def __init__(self, model_name="densenet121", pretrained=True, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        in_features = self.backbone.num_features  # 1024 for DenseNet121
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(p=drop_rate / 2.0),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = DenseNetClassifier("densenet121", pretrained=True).to(device)

# 5. Training Loop
EPOCHS = 10
LR = 3e-4

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

best_auc = 0.0
best_thresh = 0.5
history = []
os.makedirs("outputs", exist_ok=True)

print("\nStarting DenseNet-121 Training on GPU...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    val_targets, val_probs = [], []
    with torch.inference_mode():
        for images, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            val_loss += loss.item() * images.size(0)
            val_targets.extend(labels.cpu().numpy().flatten())
            val_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
    val_loss /= len(val_loader.dataset)
    val_targets = np.array(val_targets)
    val_probs = np.nan_to_num(np.array(val_probs), nan=0.5)

    val_auc = roc_auc_score(val_targets, val_probs)
    cur_best_f1, cur_thresh = 0.0, 0.5
    for t in np.arange(0.1, 0.95, 0.05):
        f1 = f1_score(val_targets, (val_probs >= t).astype(int), zero_division=0)
        if f1 > cur_best_f1:
            cur_best_f1, cur_thresh = f1, t

    scheduler.step()
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {cur_best_f1:.4f} (thresh={cur_thresh:.2f})")

    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc, "val_f1": cur_best_f1})

    if val_auc > best_auc:
        best_auc = val_auc
        best_thresh = cur_thresh
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_auc": best_auc,
            "best_threshold": best_thresh,
            "model_name": "densenet121"
        }, "outputs/densenet_weights.pt")
        print(f"  --> [*] Saved new best model to outputs/densenet_weights.pt (AUC: {best_auc:.4f})")

print(f"\nDenseNet-121 Training Complete! Best Validation AUC: {best_auc:.4f}")

# 6. Automatic Test Inference
TEST_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "test" in root and "images" in root:
        TEST_DIR = root
        break

if TEST_DIR:
    print(f"\nRunning test inference on {TEST_DIR}...")
    test_files = sorted([os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    class TestDataset(Dataset):
        def __init__(self, file_paths):
            self.file_paths = file_paths
        def __len__(self):
            return len(self.file_paths)
        def __getitem__(self, idx):
            path = self.file_paths[idx]
            image_id = os.path.splitext(os.path.basename(path))[0]
            with Image.open(path) as img:
                arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
            return tensor, image_id

    test_loader = DataLoader(TestDataset(test_files), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    
    # Load best checkpoint
    ckpt = torch.load("outputs/densenet_weights.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_ids, all_probs = [], []
    with torch.inference_mode():
        for images, ids in tqdm(test_loader, desc="DenseNet Inference (TTA)"):
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                # Pass 1: Original
                p1 = torch.sigmoid(model(images)).squeeze(-1)
                # Pass 2: Flipped (TTA)
                p2 = torch.sigmoid(model(torch.flip(images, dims=[-1]))).squeeze(-1)
                probs = (p1 + p2) / 2.0
            all_ids.extend(ids)
            all_probs.extend(probs.cpu().numpy().tolist())

    densenet_probs = np.clip(np.array(all_probs), 0.0, 1.0)
    df_densenet = pd.DataFrame({
        "image_id": all_ids,
        "label": (densenet_probs >= best_thresh).astype(int),
        "confidence": np.round(densenet_probs, 4)
    })
    df_densenet.to_csv("predictions_densenet.csv", index=False)
    print("Saved predictions_densenet.csv successfully!")

    # 7. Auto-Blend with EfficientNet-B3 (if existing predictions.csv found)
    if os.path.exists("predictions.csv"):
        print("\nFound existing EfficientNet predictions.csv! Generating blended ensemble...")
        df_eff = pd.read_csv("predictions.csv")
        eff_probs = df_eff["confidence"].values
        
        # Weighted blend (55% EfficientNet [0.9142] + 45% DenseNet)
        blended_probs = 0.55 * eff_probs + 0.45 * densenet_probs
        blended_probs = np.clip(blended_probs, 0.0, 1.0)
        df_blend = pd.DataFrame({
            "image_id": all_ids,
            "label": (blended_probs >= 0.50).astype(int),
            "confidence": np.round(blended_probs, 4)
        })
        df_blend.to_csv("predictions_blended.csv", index=False)
        print("Successfully generated 'predictions_blended.csv' (55% EfficientNet + 45% DenseNet)!")

