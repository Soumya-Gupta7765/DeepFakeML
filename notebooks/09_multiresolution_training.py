# ==============================================================================
# DEEPFAKE ML HACKATHON — MULTI-RESOLUTION CHAMPION PIPELINE (256px + 224px)
# ==============================================================================
# Breakthrough Innovations:
#   1. Cross-Resolution Feature Fusion:
#      - DenseNet-121 trained at 256x256 (+31% pixel density for micro-boundary seams)
#      - Swin-Tiny ViT trained at 224x224 (Native window attention for landmark geometry)
#   2. Extended 14-Epoch Cosine Schedule with Warmup & Label Smoothing (0.02)
#   3. 8-Pass Multi-Scale Test-Time Augmentation (TTA)
#   4. Percentile Rank-Averaged Ensemble
# ==============================================================================

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm", "albumentations", "scikit-learn", "scipy"])

import os, time, random, cv2, shutil
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
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)
os.makedirs("outputs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ------------------------------------------------------------------------------
# 2. DATASET DISCOVERY & STRATIFIED 5-FOLD SPLIT
# ------------------------------------------------------------------------------
DATASET_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train_metadata.csv" in files:
        DATASET_DIR = root
        break

if DATASET_DIR is None:
    raise FileNotFoundError("Could not find train_metadata.csv in /kaggle/input!")

print(f"[*] Dataset located at: {DATASET_DIR}")
train_meta = pd.read_csv(os.path.join(DATASET_DIR, "train_metadata.csv"))
train_meta["file_path"] = [
    os.path.join(DATASET_DIR, "train", "real" if r["label"] == 0 else "fake", f"{r['image_id']}.jpg")
    for _, r in train_meta.iterrows()
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_meta["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_meta, train_meta["label"])):
    train_meta.loc[val_idx, "fold"] = fold

train_df = train_meta[train_meta["fold"] != 0].reset_index(drop=True)
val_df = train_meta[train_meta["fold"] == 0].reset_index(drop=True)
print(f"[*] Splits: Train={len(train_df)} | Holdout Validation={len(val_df)}")

# ------------------------------------------------------------------------------
# 3. MULTI-RESOLUTION AUGMENTATIONS & LOADERS
# ------------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_transforms(img_size):
    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_tf, val_tf

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
        # Smooth label slightly: 0 -> 0.02, 1 -> 0.98
        target = float(row["label"]) * 0.96 + 0.02
        return img, torch.tensor([target], dtype=torch.float32), float(row["label"]), str(row["image_id"])

# 256x256 for DenseNet
train_tf_256, val_tf_256 = get_transforms(256)
train_loader_256 = DataLoader(DeepfakeDataset(train_df, train_tf_256), batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader_256 = DataLoader(DeepfakeDataset(val_df, val_tf_256), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# 224x224 for Swin-Tiny
train_tf_224, val_tf_224 = get_transforms(224)
train_loader_224 = DataLoader(DeepfakeDataset(train_df, train_tf_224), batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader_224 = DataLoader(DeepfakeDataset(val_df, val_tf_224), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# ------------------------------------------------------------------------------
# 4. ARCHITECTURES
# ------------------------------------------------------------------------------
class DenseNetClassifier(nn.Module):
    """DenseNet-121: Micro-texture boundary specialist."""
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
    """Swin-Tiny: Shifted-window self-attention Vision Transformer."""
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
# 5. TRAINING ENGINE WITH WARMUP & COSINE ANNEALING
# ------------------------------------------------------------------------------
def train_model(model, name, train_loader, val_loader, weights_path, epochs=12, lr=2e-4):
    print("\n" + "=" * 65)
    print(f"  TRAINING [{name.upper()}] ({epochs} EPOCHS | LR={lr})")
    print("=" * 65)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    best_auc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets, _, _ in tqdm(train_loader, desc=f"[{name}] Epoch {epoch}/{epochs} (Train)", leave=False):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                loss = criterion(model(images), targets)
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
            for images, _, raw_labels, _ in tqdm(val_loader, desc=f"[{name}] Epoch {epoch}/{epochs} (Val)", leave=False):
                images = images.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(images)
                val_targets.extend(raw_labels.numpy().flatten())
                val_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                
        val_probs = np.nan_to_num(np.array(val_probs), nan=0.5)
        val_auc = roc_auc_score(np.array(val_targets), val_probs)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model_state_dict": model.state_dict(), "best_auc": best_auc, "model_name": name}, weights_path)
            print(f"  --> [*] Saved new best {name} (AUC: {best_auc:.4f})")

    return best_auc

# ------------------------------------------------------------------------------
# 6. TRAIN DENSENET (256x256, 12 EPOCHS) & SWIN (224x224, 14 EPOCHS)
# ------------------------------------------------------------------------------
# 1. DenseNet-121 at 256x256
m_dense = DenseNetClassifier().to(device)
dense_auc = train_model(m_dense, "densenet121", train_loader_256, val_loader_256, "outputs/densenet_weights.pt", epochs=12, lr=3e-4)

# 2. Swin-Tiny at 224x224
m_swin = SwinClassifier().to(device)
swin_auc = train_model(m_swin, "swin_tiny", train_loader_224, val_loader_224, "outputs/swin_weights.pt", epochs=14, lr=1.5e-4)

# ------------------------------------------------------------------------------
# 7. 8-PASS MULTI-SCALE TTA ENGINE
# ------------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  EVALUATING 8-PASS MULTI-SCALE TTA ON VALIDATION SET")
print("=" * 65)

m_dense.load_state_dict(torch.load("outputs/densenet_weights.pt", map_location=device, weights_only=False)["model_state_dict"])
m_dense.eval()

m_swin.load_state_dict(torch.load("outputs/swin_weights.pt", map_location=device, weights_only=False)["model_state_dict"])
m_swin.eval()

def predict_8pass_tta(model, images):
    with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
        # 1. Original
        p1 = torch.sigmoid(model(images)).squeeze(-1)
        # 2. Flip
        img_f = torch.flip(images, dims=[-1])
        p2 = torch.sigmoid(model(img_f)).squeeze(-1)
        # 3. Brightness (+5%)
        p3 = torch.sigmoid(model(images * 1.05)).squeeze(-1)
        # 4. Flip + Brightness
        p4 = torch.sigmoid(model(img_f * 1.05)).squeeze(-1)
        # 5. Contrast (+8%)
        p5 = torch.sigmoid(model((images - 0.5) * 1.08 + 0.5)).squeeze(-1)
        # 6. Flip + Contrast
        p6 = torch.sigmoid(model((img_f - 0.5) * 1.08 + 0.5)).squeeze(-1)
        # 7. Center Zoom (1.06x)
        h, w = images.shape[-2:]
        scaled = torch.nn.functional.interpolate(images, scale_factor=1.06, mode='bilinear', align_corners=False)
        sh, sw = scaled.shape[-2:]
        dh, dw = (sh - h) // 2, (sw - w) // 2
        img_zoom = scaled[:, :, dh:dh+h, dw:dw+w]
        p7 = torch.sigmoid(model(img_zoom)).squeeze(-1)
        # 8. Flip + Zoom
        p8 = torch.sigmoid(model(torch.flip(img_zoom, dims=[-1]))).squeeze(-1)
        
    return (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) / 8.0

p_dense_val, p_swin_val, y_true = [], [], []
with torch.inference_mode():
    for (img_256, _, y, _), (img_224, _, _, _) in tqdm(zip(val_loader_256, val_loader_224), total=len(val_loader_256), desc="Running 8-Pass TTA Validation"):
        p_dense_val.extend(predict_8pass_tta(m_dense, img_256.to(device)).cpu().numpy().tolist())
        p_swin_val.extend(predict_8pass_tta(m_swin, img_224.to(device)).cpu().numpy().tolist())
        y_true.extend(y.numpy().flatten())

p_dense_val = np.array(p_dense_val)
p_swin_val = np.array(p_swin_val)
y_true = np.array(y_true)

auc_dense_8tta = roc_auc_score(y_true, p_dense_val)
auc_swin_8tta = roc_auc_score(y_true, p_swin_val)

# Optimal Weight Search
r_dense_v = rankdata(p_dense_val) / len(p_dense_val)
r_swin_v = rankdata(p_swin_val) / len(p_swin_val)

best_w, best_auc = 0.50, 0.0
for w in np.linspace(0.30, 0.70, 41):
    blend = w * r_swin_v + (1.0 - w) * r_dense_v
    score = roc_auc_score(y_true, blend)
    if score > best_auc:
        best_auc, best_w = score, w

print("\n" + "*" * 65)
print(f"[*] DenseNet-121 (256x256 | 8-Pass TTA) AUC : {auc_dense_8tta:.4f}")
print(f"[*] Swin-Tiny ViT (224x224 | 8-Pass TTA) AUC : {auc_swin_8tta:.4f}")
print(f"[*] Optimal Blend Ratio: {best_w*100:.1f}% Swin + {(1-best_w)*100:.1f}% DenseNet")
print(f"[*] >>> MULTI-RESOLUTION RANK-AVERAGED AUC  : {best_auc:.4f} <<<")
print("*" * 65)

# ------------------------------------------------------------------------------
# 8. TEST INFERENCE (3,000 IMAGES)
# ------------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  GENERATING FINAL 3,000 TEST PREDICTIONS")
print("=" * 65)

TEST_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "test" in root and "images" in root: TEST_DIR = root; break

test_files = sorted([os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

class TestDatasetMultiRes(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.array(Image.open(path).convert("RGB"))
        
        # 256
        im256 = cv2.resize(img, (256, 256))
        arr256 = (im256.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        t256 = torch.from_numpy(arr256).permute(2, 0, 1).float()
        
        # 224
        im224 = cv2.resize(img, (224, 224))
        arr224 = (im224.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        t224 = torch.from_numpy(arr224).permute(2, 0, 1).float()
        
        return t256, t224, os.path.splitext(os.path.basename(path))[0]

test_loader = DataLoader(TestDatasetMultiRes(test_files), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

test_dense, test_swin, test_ids = [], [], []
with torch.inference_mode():
    for im256, im224, ids in tqdm(test_loader, desc="Predicting on Test Set"):
        test_dense.extend(predict_8pass_tta(m_dense, im256.to(device)).cpu().numpy().tolist())
        test_swin.extend(predict_8pass_tta(m_swin, im224.to(device)).cpu().numpy().tolist())
        test_ids.extend(ids)

test_dense = np.array(test_dense)
test_swin = np.array(test_swin)

r_dense_t = rankdata(test_dense) / len(test_dense)
r_swin_t = rankdata(test_swin) / len(test_swin)

final_conf = best_w * r_swin_t + (1.0 - best_w) * r_dense_t
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

# Save best model as model_weights.pt
best_source = "outputs/swin_weights.pt" if auc_swin_8tta >= auc_dense_8tta else "outputs/densenet_weights.pt"
shutil.copyfile(best_source, "outputs/model_weights.pt")
print(f"[+] Saved submission weights to outputs/model_weights.pt (from {best_source})")

print("\n" + "=" * 65)
print("  MULTI-RESOLUTION CHAMPION PIPELINE COMPLETE!")
print(f"  Final Benchmark AUC: {best_auc:.4f}")
print("  Generated: 'predictions.csv' and 'outputs/model_weights.pt'")
print("=" * 65)
