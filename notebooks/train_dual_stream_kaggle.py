# ==============================================================================
# Dual-Stream (Spatial RGB + Frequency FFT) Training Script for Kaggle GPU
# ==============================================================================

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

# 3. Frequency Feature Extractor
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def compute_fft(image_rgb, target_size=224):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if gray.shape[0] != target_size or gray.shape[1] != target_size:
        gray = cv2.resize(gray, (target_size, target_size))
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = 20.0 * np.log(np.abs(fshift) + 1e-8)
    min_v, max_v = np.min(mag), np.max(mag)
    norm = (mag - min_v) / (max_v - min_v + 1e-8)
    fft_3ch = np.stack([norm] * 3, axis=-1).astype(np.float32)
    fft_norm = (fft_3ch - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(fft_norm).permute(2, 0, 1).float()

# 4. Augmentations
train_transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

valid_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

class DualStreamDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_bgr = cv2.imread(row["file_path"])
        if image_bgr is None:
            image_rgb = np.array(Image.open(row["file_path"]).convert("RGB"))
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Stream 2: Compute FFT frequency spectrum
        fft_tensor = compute_fft(image_rgb, target_size=224)

        # Stream 1: Spatial augmentations
        if self.transform:
            spatial_tensor = self.transform(image=image_rgb)["image"]
        else:
            spatial_arr = (image_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            spatial_tensor = torch.from_numpy(spatial_arr).permute(2, 0, 1).float()

        label = torch.tensor([float(row["label"])], dtype=torch.float32)
        return spatial_tensor, fft_tensor, label, str(row["image_id"])

train_loader = DataLoader(DualStreamDataset(train_df, train_transforms), batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(DualStreamDataset(val_df, valid_transforms), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# 5. Model Definition
class DualStreamClassifier(nn.Module):
    def __init__(self, spatial_name="efficientnet_b3", freq_name="resnet18", pretrained=True, drop_rate=0.3):
        super().__init__()
        self.spatial_backbone = timm.create_model(spatial_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        self.freq_backbone = timm.create_model(freq_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        total_dim = self.spatial_backbone.num_features + self.freq_backbone.num_features
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=drop_rate / 2.0),
            nn.Linear(256, 1)
        )

    def forward(self, x_spatial, x_freq):
        f_spatial = self.spatial_backbone(x_spatial)
        f_freq = self.freq_backbone(x_freq)
        fused = torch.cat([f_spatial, f_freq], dim=1)
        return self.fusion_head(fused)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = DualStreamClassifier().to(device)

# 6. Training Loop with AMP & Cosine Annealing
EPOCHS = 10
LR = 2.5e-4

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

best_auc = 0.0
best_thresh = 0.5
os.makedirs("outputs", exist_ok=True)

print("Starting Dual-Stream (Spatial + FFT) Training on GPU...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for x_spat, x_freq, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        x_spat, x_freq, labels = x_spat.to(device), x_freq.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x_spat, x_freq)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * x_spat.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    val_targets, val_probs = [], []
    with torch.inference_mode():
        for x_spat, x_freq, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            x_spat, x_freq, labels = x_spat.to(device), x_freq.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x_spat, x_freq)
                loss = criterion(logits, labels)
            val_loss += loss.item() * x_spat.size(0)
            val_targets.extend(labels.cpu().numpy().flatten())
            val_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
    val_loss /= len(val_loader.dataset)
    val_targets = np.array(val_targets)
    val_probs = np.array(val_probs)

    val_auc = roc_auc_score(val_targets, val_probs)
    cur_best_f1, cur_thresh = 0.0, 0.5
    for t in np.arange(0.1, 0.95, 0.05):
        f1 = f1_score(val_targets, (val_probs >= t).astype(int), zero_division=0)
        if f1 > cur_best_f1:
            cur_best_f1, cur_thresh = f1, t

    scheduler.step()
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {cur_best_f1:.4f} (thresh={cur_thresh:.2f})")

    if val_auc > best_auc:
        best_auc = val_auc
        best_thresh = cur_thresh
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_auc": best_auc,
            "best_threshold": best_thresh,
            "model_name": "dual_stream_fft"
        }, "outputs/dual_stream_weights.pt")
        print(f"  --> [*] Saved new best model to outputs/dual_stream_weights.pt (AUC: {best_auc:.4f})")

print(f"\nDual-Stream Training Complete! Best Validation AUC: {best_auc:.4f}")
