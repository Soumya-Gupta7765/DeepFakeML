# ==============================================================================
# DEEPFAKE ML HACKATHON — MASTER END-TO-END PIPELINE (ALL-IN-ONE)
# ==============================================================================
# This single script runs the entire championship pipeline on Kaggle GPU:
#   1. Prepares dataset & Stratified 5-Fold split (Fold 0 as validation).
#   2. Trains Model 1: DenseNet-121 (Best CNN, ~7 mins on GPU).
#   3. Trains Model 2: Swin-Tiny Vision Transformer (Best ViT, ~8 mins on GPU).
#   4. Evaluates both models with 4-Pass Test-Time Augmentation (TTA).
#   5. Evaluates the CNN + ViT Super-Ensemble on Validation (calculates final AUC).
#   6. Generates final test predictions on all 3,000 images -> 'predictions.csv'.
#   7. Saves all weights, curves, and history logs into 'outputs/'.
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

# ------------------------------------------------------------------------------
# 1. SETUP & REPRODUCIBILITY
# ------------------------------------------------------------------------------
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
os.makedirs("outputs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Compute Device: {device}")
if torch.cuda.is_available():
    print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

# ------------------------------------------------------------------------------
# 2. DATASET DISCOVERY & STRATIFIED 5-FOLD
# ------------------------------------------------------------------------------
DATASET_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train_metadata.csv" in files:
        DATASET_DIR = root
        break

if DATASET_DIR is None:
    raise FileNotFoundError("Could not find train_metadata.csv in /kaggle/input! Make sure your dataset is attached.")

print(f"[*] Dataset located at: {DATASET_DIR}")
train_meta = pd.read_csv(os.path.join(DATASET_DIR, "train_metadata.csv"))

# Attach file paths
file_paths = []
for _, row in train_meta.iterrows():
    sub = "real" if row["label"] == 0 else "fake"
    file_paths.append(os.path.join(DATASET_DIR, "train", sub, f"{row['image_id']}.jpg"))
train_meta["file_path"] = file_paths

# Stratified 5-Fold split (Fold 0 = Validation)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_meta["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_meta, train_meta["label"])):
    train_meta.loc[val_idx, "fold"] = fold

train_df = train_meta[train_meta["fold"] != 0].reset_index(drop=True)
val_df = train_meta[train_meta["fold"] == 0].reset_index(drop=True)
print(f"[*] Dataset Splits: Train = {len(train_df)} samples | Validation = {len(val_df)} samples")

# ------------------------------------------------------------------------------
# 3. ROBUST DEEPFAKE DATA AUGMENTATIONS
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# 4. ARCHITECTURE DEFINITIONS
# ------------------------------------------------------------------------------
class DenseNetClassifier(nn.Module):
    """DenseNet-121 with regularized LayerNorm classification head."""
    def __init__(self, model_name="densenet121", pretrained=True, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        in_features = self.backbone.num_features  # 1024
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


class SwinClassifier(nn.Module):
    """Swin-Tiny Vision Transformer with LayerNorm head."""
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        in_features = self.backbone.num_features  # 768
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=drop_rate / 2.0),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.head(self.backbone(x))


# ------------------------------------------------------------------------------
# 5. GENERAL TRAINING ENGINE
# ------------------------------------------------------------------------------
def train_model(model, model_name, weights_path, epochs=10, lr=3e-4):
    print("\n" + "=" * 65)
    print(f"  TRAINING: {model_name.upper()} (10 EPOCHS)")
    print("=" * 65)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    
    best_auc = 0.0
    best_thresh = 0.5
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels, _ in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}/{epochs} (Train)", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                loss = criterion(model(images), labels)
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
            for images, labels, _ in tqdm(val_loader, desc=f"[{model_name}] Epoch {epoch}/{epochs} (Val)", leave=False):
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
        scheduler.step()
        
        # Best F1 threshold
        cur_best_f1, cur_thresh = 0.0, 0.5
        for t in np.arange(0.1, 0.95, 0.05):
            f1 = f1_score(val_targets, (val_probs >= t).astype(int), zero_division=0)
            if f1 > cur_best_f1: cur_best_f1, cur_thresh = f1, t
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {cur_best_f1:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc, "val_f1": cur_best_f1})
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_thresh = cur_thresh
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_auc": best_auc,
                "best_threshold": best_thresh,
                "model_name": model_name
            }, weights_path)
            print(f"  --> [*] Saved new best {model_name} (AUC: {best_auc:.4f})")
            
    print(f"\n[*] {model_name.upper()} Training Finished! Peak Val AUC: {best_auc:.4f}")
    return best_auc, history


# ------------------------------------------------------------------------------
# 6. TRAIN MODEL 1 (DENSENET-121) & MODEL 2 (SWIN TRANSFORMER)
# ------------------------------------------------------------------------------
# Train Model 1: DenseNet-121
densenet_model = DenseNetClassifier("densenet121", pretrained=True).to(device)
dense_auc, dense_hist = train_model(densenet_model, "densenet121", "outputs/densenet_weights.pt", epochs=10, lr=3e-4)

# Train Model 2: Swin-Tiny Vision Transformer
swin_model = SwinClassifier("swin_tiny_patch4_window7_224", pretrained=True).to(device)
swin_auc, swin_hist = train_model(swin_model, "swin_tiny", "outputs/swin_weights.pt", epochs=10, lr=1.5e-4)


# ------------------------------------------------------------------------------
# 7. 4-PASS TTA EVALUATION & SUPER-ENSEMBLE VALIDATION
# ------------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  EVALUATING SUPER-ENSEMBLE WITH 4-PASS TTA ON VALIDATION SET")
print("=" * 65)

# Load best checkpoints
ckpt_dense = torch.load("outputs/densenet_weights.pt", map_location=device, weights_only=False)
densenet_model.load_state_dict(ckpt_dense["model_state_dict"])
densenet_model.eval()

ckpt_swin = torch.load("outputs/swin_weights.pt", map_location=device, weights_only=False)
swin_model.load_state_dict(ckpt_swin["model_state_dict"])
swin_model.eval()

def predict_4pass_tta(model, images):
    with torch.amp.autocast('cuda'):
        p1 = torch.sigmoid(model(images)).squeeze(-1)
        p2 = torch.sigmoid(model(torch.flip(images, dims=[-1]))).squeeze(-1)
        p3 = torch.sigmoid(model(images * 1.05)).squeeze(-1)
        p4 = torch.sigmoid(model(torch.flip(images, dims=[-1]) * 1.05)).squeeze(-1)
    return (p1 + p2 + p3 + p4) / 4.0

p_dense_val, p_swin_val, y_true = [], [], []
with torch.inference_mode():
    for images, labels, _ in tqdm(val_loader, desc="Running 4-Pass TTA Validation"):
        images = images.to(device)
        p_dense_val.extend(predict_4pass_tta(densenet_model, images).cpu().numpy().tolist())
        p_swin_val.extend(predict_4pass_tta(swin_model, images).cpu().numpy().tolist())
        y_true.extend(labels.numpy().flatten())

p_dense_val = np.array(p_dense_val)
p_swin_val = np.array(p_swin_val)
y_true = np.array(y_true)

auc_dense_tta = roc_auc_score(y_true, p_dense_val)
auc_swin_tta = roc_auc_score(y_true, p_swin_val)

# Optimal 50/50 blend between CNN (DenseNet) & ViT (Swin)
p_ensemble_val = 0.50 * p_swin_val + 0.50 * p_dense_val
super_ensemble_auc = roc_auc_score(y_true, p_ensemble_val)

print("\n" + "*" * 65)
print(f"[*] DenseNet-121 (4-Pass TTA) Val AUC : {auc_dense_tta:.4f}")
print(f"[*] Swin-Tiny    (4-Pass TTA) Val AUC : {auc_swin_tta:.4f}")
print(f"[*] >>> FINAL SUPER-ENSEMBLE VAL AUC : {super_ensemble_auc:.4f} <<<")
print("*" * 65)


# ------------------------------------------------------------------------------
# 8. TEST INFERENCE ON ALL 3,000 TEST IMAGES -> PREDICTIONS.CSV
# ------------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  GENERATING PREDICTIONS ON 3,000 UNLABELLED TEST IMAGES")
print("=" * 65)

TEST_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "test" in root and "images" in root:
        TEST_DIR = root
        break

if TEST_DIR is None:
    raise FileNotFoundError("Could not find test/images folder!")

test_files = sorted([os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"[*] Found {len(test_files)} test images in {TEST_DIR}")

class TestDataset(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        arr = (np.array(img, dtype=np.float32) / 255.0 - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).float(), os.path.splitext(os.path.basename(path))[0]

test_loader = DataLoader(TestDataset(test_files), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

test_dense_probs, test_swin_probs, test_ids = [], [], []
with torch.inference_mode():
    for images, ids in tqdm(test_loader, desc="Running Ensemble Test Inference"):
        images = images.to(device)
        test_dense_probs.extend(predict_4pass_tta(densenet_model, images).cpu().numpy().tolist())
        test_swin_probs.extend(predict_4pass_tta(swin_model, images).cpu().numpy().tolist())
        test_ids.extend(ids)

test_dense_probs = np.array(test_dense_probs)
test_swin_probs = np.array(test_swin_probs)

# Super-ensemble blend (50% Swin + 50% DenseNet)
final_confidences = 0.50 * test_swin_probs + 0.50 * test_dense_probs
final_confidences = np.clip(final_confidences, 0.0, 1.0)
final_labels = (final_confidences >= 0.50).astype(int)

# Export predictions.csv (Official Format)
predictions_df = pd.DataFrame({
    "image_id": test_ids,
    "label": final_labels,
    "confidence": np.round(final_confidences, 4)
})
predictions_df.to_csv("predictions.csv", index=False)
print(f"\n[+] 'predictions.csv' generated successfully! ({len(predictions_df)} rows)")
print(f"[+] Class Breakdown:\n{predictions_df['label'].value_counts().to_dict()}")

# Also save standalone predictions for individual models
pd.DataFrame({"image_id": test_ids, "label": (test_dense_probs >= 0.5).astype(int), "confidence": np.round(test_dense_probs, 4)}).to_csv("outputs/predictions_densenet.csv", index=False)
pd.DataFrame({"image_id": test_ids, "label": (test_swin_probs >= 0.5).astype(int), "confidence": np.round(test_swin_probs, 4)}).to_csv("outputs/predictions_swin.csv", index=False)

# Copy the best single model weights as 'model_weights.pt' in outputs/ for submission convenience
import shutil
best_weight_source = "outputs/swin_weights.pt" if auc_swin_tta >= auc_dense_tta else "outputs/densenet_weights.pt"
shutil.copyfile(best_weight_source, "outputs/model_weights.pt")
print(f"[+] Submission weights copied from {best_weight_source} -> outputs/model_weights.pt")

# Save training curves plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot([h["epoch"] for h in dense_hist], [h["val_auc"] for h in dense_hist], label=f"DenseNet-121 (Peak: {dense_auc:.4f})", marker="o", color="#1f77b4")
plt.plot([h["epoch"] for h in swin_hist], [h["val_auc"] for h in swin_hist], label=f"Swin-Tiny (Peak: {swin_auc:.4f})", marker="s", color="#2ca02c")
plt.title("Validation AUC Progression"); plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot([h["epoch"] for h in dense_hist], [h["train_loss"] for h in dense_hist], label="DenseNet Train Loss", linestyle="--", color="#1f77b4")
plt.plot([h["epoch"] for h in swin_hist], [h["train_loss"] for h in swin_hist], label="Swin Train Loss", linestyle="--", color="#2ca02c")
plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=300)
print("[+] Updated training curves saved to 'outputs/training_curves.png'")

print("\n" + "=" * 65)
print("  ALL COMPLETED SUCCESSFULLY!")
print(f"  Final Super-Ensemble Validation AUC: {super_ensemble_auc:.4f}")
print("  Generated Files: 'predictions.csv', 'outputs/model_weights.pt', 'outputs/training_curves.png'")
print("=" * 65)
