# ============================================================
# Deepfake ML Hackathon — Single-Cell Kaggle Training Script
# ============================================================

# Step 1: Install prerequisites
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

# Step 2: Seed everything
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

# Step 3: Locate Dataset on Kaggle
DATASET_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train_metadata.csv" in files:
        DATASET_DIR = root
        break

if DATASET_DIR is None:
    raise FileNotFoundError("Could not find train_metadata.csv in /kaggle/input. Please ensure your dataset is attached!")

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

# Step 4: Transforms and Dataset
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.4),
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

# Step 5: Model
class DeepfakeClassifier(nn.Module):
    def __init__(self, model_name="efficientnet_b3", pretrained=True, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=drop_rate / 2.0),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = DeepfakeClassifier("efficientnet_b3", pretrained=True).to(device)

# Step 6: Training with AMP & Cosine Annealing
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

print("Starting training on GPU...")
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
            "model_name": "efficientnet_b3"
        }, "outputs/model_weights.pt")
        print(f"  --> [*] Saved new best model to outputs/model_weights.pt (AUC: {best_auc:.4f})")

# Step 7: Plot curves
hist_df = pd.DataFrame(history)
hist_df.to_csv("outputs/training_history.csv", index=False)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss", marker="o")
plt.plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss", marker="s")
plt.title("Loss Curves"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_df["epoch"], hist_df["val_auc"], label="Val AUC-ROC", marker="^", color="green")
plt.title("Validation AUC-ROC"); plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=300)
print(f"\nDone! Best AUC: {best_auc:.4f} (Saved to outputs/model_weights.pt)")

