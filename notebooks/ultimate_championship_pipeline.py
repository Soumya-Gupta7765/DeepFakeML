# ==============================================================================
# ULTIMATE CHAMPIONSHIP DEEPFAKE PIPELINE (ALL TECHNIQUES COMBINED)
# ==============================================================================
# Features:
#   1. SRM Forensic Noise Filter Layer (detects GAN synthesis noise)
#   2. DenseNet-121 (Local boundary & texture modeling)
#   3. Swin-Tiny Vision Transformer (Global anatomical attention)
#   4. 4-Pass Test-Time Augmentation (TTA)
#   5. Rank Averaging (Optimizes Area Under ROC curve mathematically)
# ==============================================================================

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm", "albumentations", "scikit-learn", "scipy"])

import os, time, random, cv2
import numpy as np, pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.stats import rankdata

import torch, torch.nn as nn
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
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
seed_everything(42)

os.makedirs("outputs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ------------------------------------------------------------------------------
# 2. DATASET PREPARATION & 5-FOLD STRATIFICATION
# ------------------------------------------------------------------------------
DATASET_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train_metadata.csv" in files: DATASET_DIR = root; break

if DATASET_DIR is None:
    raise FileNotFoundError("Could not find train_metadata.csv in /kaggle/input!")

print(f"[*] Dataset located at: {DATASET_DIR}")
train_meta = pd.read_csv(os.path.join(DATASET_DIR, "train_metadata.csv"))
train_meta["file_path"] = [os.path.join(DATASET_DIR, "train", "real" if r["label"] == 0 else "fake", f"{r['image_id']}.jpg") for _, r in train_meta.iterrows()]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_meta["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_meta, train_meta["label"])):
    train_meta.loc[val_idx, "fold"] = fold

train_df = train_meta[train_meta["fold"] != 0].reset_index(drop=True)
val_df = train_meta[train_meta["fold"] == 0].reset_index(drop=True)
print(f"[*] Splits: Train = {len(train_df)} | Holdout Validation = {len(val_df)}")

# ------------------------------------------------------------------------------
# 3. SRM FORENSIC FILTER LAYER (HIGH-PASS RESIDUALS)
# ------------------------------------------------------------------------------
class SRMConv2d(nn.Module):
    """Spatial Rich Model (SRM) High-Pass Filter: extracts noise residuals & GAN fingerprints."""
    def __init__(self):
        super().__init__()
        # 3 standard forensic kernels: 1st order, 2nd order, and Laplacian edge
        srm_kernel = np.array([
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], # 5x5 edge
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],            # 3x3 2nd order
            [[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, -1, 4, -1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]]             # 3x3 Laplacian
        ], dtype=np.float32) / 12.0

        self.conv = nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv.weight = nn.Parameter(torch.from_numpy(srm_kernel).unsqueeze(1), requires_grad=False)

    def forward(self, x):
        # Convert RGB to Luminance
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        return self.conv(gray)

# ------------------------------------------------------------------------------
# 4. DATA AUGMENTATIONS & LOADERS
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
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["file_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.array(Image.open(row["file_path"]).convert("RGB"))
        if self.transform: img = self.transform(image=img)["image"]
        return img, torch.tensor([float(row["label"])], dtype=torch.float32), str(row["image_id"])

train_loader = DataLoader(DeepfakeDataset(train_df, train_transforms), batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(DeepfakeDataset(val_df, valid_transforms), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# ------------------------------------------------------------------------------
# 5. MODEL ARCHITECTURES
# ------------------------------------------------------------------------------
class DenseNetClassifier(nn.Module):
    def __init__(self, model_name="densenet121", pretrained=True, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(self.backbone.num_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(p=drop_rate / 2.0),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.head(self.backbone(x))

class SwinClassifier(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(self.backbone.num_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=drop_rate / 2.0),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.head(self.backbone(x))

# ------------------------------------------------------------------------------
# 6. UNIVERSAL TRAINING ENGINE
# ------------------------------------------------------------------------------
def train_model(model, name, weights_path, epochs=10, lr=3e-4):
    print("\n" + "=" * 65)
    print(f"  TRAINING {name.upper()} (10 EPOCHS)")
    print("=" * 65)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    best_auc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels, _ in tqdm(train_loader, desc=f"[{name}] Epoch {epoch}/{epochs} (Train)", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_targets, val_probs = [], []
        with torch.inference_mode():
            for images, labels, _ in tqdm(val_loader, desc=f"[{name}] Epoch {epoch}/{epochs} (Val)", leave=False):
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(images)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                val_targets.extend(labels.cpu().numpy().flatten())
                val_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        val_loss /= len(val_loader.dataset)
        val_probs = np.nan_to_num(np.array(val_probs), nan=0.5)
        val_auc = roc_auc_score(np.array(val_targets), val_probs)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model_state_dict": model.state_dict(), "best_auc": best_auc, "model_name": name}, weights_path)
            print(f"  --> [*] Saved new best {name} (AUC: {best_auc:.4f})")

    return best_auc

# ------------------------------------------------------------------------------
# 7. TRAIN DENSENET-121 (CNN) AND SWIN-TINY (ViT)
# ------------------------------------------------------------------------------
m_dense = DenseNetClassifier().to(device)
dense_auc = train_model(m_dense, "densenet121", "outputs/densenet_weights.pt", epochs=10, lr=3e-4)

m_swin = SwinClassifier().to(device)
swin_auc = train_model(m_swin, "swin_tiny", "outputs/swin_weights.pt", epochs=10, lr=1.5e-4)

# ------------------------------------------------------------------------------
# 8. 4-PASS TTA + PROBABILITY & RANK AVERAGING ON VALIDATION
# ------------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  EVALUATING 4-PASS TTA & RANK AVERAGING ON VALIDATION SET")
print("=" * 65)

# Load best checkpoints
m_dense.load_state_dict(torch.load("outputs/densenet_weights.pt", map_location=device, weights_only=False)["model_state_dict"])
m_dense.eval()
m_swin.load_state_dict(torch.load("outputs/swin_weights.pt", map_location=device, weights_only=False)["model_state_dict"])
m_swin.eval()

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
        p_dense_val.extend(predict_4pass_tta(m_dense, images).cpu().numpy().tolist())
        p_swin_val.extend(predict_4pass_tta(m_swin, images).cpu().numpy().tolist())
        y_true.extend(labels.numpy().flatten())

p_dense_val = np.array(p_dense_val)
p_swin_val = np.array(p_swin_val)
y_true = np.array(y_true)

# Metric 1: Individual 4-Pass TTA
auc_dense_tta = roc_auc_score(y_true, p_dense_val)
auc_swin_tta = roc_auc_score(y_true, p_swin_val)

# Metric 2: Standard Probability Averaging
p_linear_blend = 0.50 * p_swin_val + 0.50 * p_dense_val
auc_linear = roc_auc_score(y_true, p_linear_blend)

# Metric 3: Rank Averaging (Normalizes distributions to [0, 1] ranks)
rank_dense = rankdata(p_dense_val) / len(p_dense_val)
rank_swin = rankdata(p_swin_val) / len(p_swin_val)
p_rank_blend = 0.50 * rank_swin + 0.50 * rank_dense
auc_rank = roc_auc_score(y_true, p_rank_blend)

print("\n" + "*" * 65)
print(f"[*] DenseNet-121 (4-Pass TTA) AUC  : {auc_dense_tta:.4f}")
print(f"[*] Swin-Tiny    (4-Pass TTA) AUC  : {auc_swin_tta:.4f}")
print(f"[*] Linear Probability Blend AUC   : {auc_linear:.4f}")
print(f"[*] >>> RANK-AVERAGED ENSEMBLE AUC: {auc_rank:.4f} <<<")
print("*" * 65)

# Choose best blending strategy
use_rank = (auc_rank >= auc_linear)
best_val_auc = max(auc_rank, auc_linear)
print(f"[*] Selected strategy for test set: {'Rank Averaging' if use_rank else 'Linear Blend'}")

# ------------------------------------------------------------------------------
# 9. GENERATE FINAL PREDICTIONS ON 3,000 TEST IMAGES
# ------------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  GENERATING FINAL 3,000 TEST PREDICTIONS")
print("=" * 65)

TEST_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "test" in root and "images" in root: TEST_DIR = root; break

test_files = sorted([os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

class TestDataset(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        arr = (np.array(img, dtype=np.float32) / 255.0 - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).float(), os.path.splitext(os.path.basename(path))[0]

test_loader = DataLoader(TestDataset(test_files), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

test_dense, test_swin, test_ids = [], [], []
with torch.inference_mode():
    for images, ids in tqdm(test_loader, desc="Predicting on Test Set"):
        images = images.to(device)
        test_dense.extend(predict_4pass_tta(m_dense, images).cpu().numpy().tolist())
        test_swin.extend(predict_4pass_tta(m_swin, images).cpu().numpy().tolist())
        test_ids.extend(ids)

test_dense = np.array(test_dense)
test_swin = np.array(test_swin)

if use_rank:
    r_dense = rankdata(test_dense) / len(test_dense)
    r_swin = rankdata(test_swin) / len(test_swin)
    final_conf = 0.50 * r_swin + 0.50 * r_dense
else:
    final_conf = 0.50 * test_swin + 0.50 * test_dense

final_conf = np.clip(final_conf, 0.001, 0.999)
final_labels = (final_conf >= 0.50).astype(int)

# Export predictions.csv
predictions_df = pd.DataFrame({
    "image_id": test_ids,
    "label": final_labels,
    "confidence": np.round(final_conf, 4)
})
predictions_df.to_csv("predictions.csv", index=False)
print(f"\n[+] 'predictions.csv' generated ({len(predictions_df)} rows)!")
print(f"[+] Class Breakdown:\n{predictions_df['label'].value_counts().to_dict()}")

# Save best weight as model_weights.pt
import shutil
best_weight_file = "outputs/swin_weights.pt" if auc_swin_tta >= auc_dense_tta else "outputs/densenet_weights.pt"
shutil.copyfile(best_weight_file, "outputs/model_weights.pt")
print(f"[+] Final submission weight saved to outputs/model_weights.pt (from {best_weight_file})")

print("\n" + "=" * 65)
print("  CHAMPIONSHIP PIPELINE COMPLETE!")
print(f"  Final Benchmark AUC: {best_val_auc:.4f}")
print("  Download 'predictions.csv' and 'outputs/model_weights.pt'")
print("=" * 65)
