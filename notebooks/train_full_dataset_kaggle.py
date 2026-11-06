# ==============================================================================
# DEEPFAKE ML HACKATHON — FULL DATASET RETRAINING PIPELINE (100% DATA)
# ==============================================================================
# In competitive ML, after hyperparameter & architecture validation on folds:
#   - We train the winning model (Swin-Tiny Vision Transformer) on 100% of the
#     7,000 labeled images (no fold left out) for 10 epochs with Cosine Annealing.
#   - We also train DenseNet-121 on 100% of data to generate the ultimate
#     Rank-Averaged Super-Ensemble on the 3,000 unlabelled test images.
#   - 4-Pass Test-Time Augmentation (TTA) is applied to all test images.
#
# Generates:
#   - 'predictions.csv' (3,000 test predictions via 100%-data Rank-Averaged Ensemble)
#   - 'outputs/model_weights.pt' (Final winning Swin-Tiny weights trained on 100% data)
#   - 'outputs/predictions_swin.csv' (Standalone Swin-Tiny test predictions)
#   - 'outputs/predictions_densenet.csv' (Standalone DenseNet test predictions)
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
# 2. DATASET DISCOVERY — LOAD 100% OF TRAINING DATA (7,000 SAMPLES)
# ------------------------------------------------------------------------------
DATASET_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train_metadata.csv" in files:
        DATASET_DIR = root
        break

if DATASET_DIR is None:
    raise FileNotFoundError("Could not find train_metadata.csv in /kaggle/input! Make sure dataset is attached.")

print(f"[*] Dataset located at: {DATASET_DIR}")
train_meta = pd.read_csv(os.path.join(DATASET_DIR, "train_metadata.csv"))

# Attach full image paths
file_paths = []
for _, row in train_meta.iterrows():
    sub = "real" if row["label"] == 0 else "fake"
    file_paths.append(os.path.join(DATASET_DIR, "train", sub, f"{row['image_id']}.jpg"))
train_meta["file_path"] = file_paths

# FULL DATASET: All 7,000 samples utilized (No validation hold-out)
print(f"[*] Total Training Samples (100% of data): {len(train_meta)}")
print(f"[*] Class Distribution: Real={sum(train_meta['label'] == 0)}, Fake={sum(train_meta['label'] == 1)}")

# ------------------------------------------------------------------------------
# 3. AUGMENTATIONS & DATASET
# ------------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.3),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

class FullDeepfakeDataset(Dataset):
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

full_train_loader = DataLoader(
    FullDeepfakeDataset(train_meta, train_transforms),
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

# ------------------------------------------------------------------------------
# 4. MODEL ARCHITECTURES
# ------------------------------------------------------------------------------
class SwinClassifier(nn.Module):
    """Swin-Tiny Vision Transformer (Rank #1 Standalone Model)."""
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


class DenseNetClassifier(nn.Module):
    """DenseNet-121 CNN (Complementary Local Micro-Texture Expert)."""
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


# ------------------------------------------------------------------------------
# 5. FULL-DATASET TRAINING ENGINE (10 EPOCHS WITH COSINE ANNEALING)
# ------------------------------------------------------------------------------
def train_on_full_dataset(model, model_name, weights_path, epochs=10, lr=1.5e-4):
    print("\n" + "=" * 70)
    print(f"  RETRAINING {model_name.upper()} ON 100% DATASET ({len(train_meta)} SAMPLES)")
    print("=" * 70)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    
    epoch_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(full_train_loader, desc=f"[{model_name}] Epoch {epoch}/{epochs}", leave=False)
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(full_train_loader.dataset)
        epoch_losses.append(epoch_loss)
        cur_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Full Train Loss: {epoch_loss:.4f} | LR: {cur_lr:.6f}")
        
        # Save checkpoints at the optimal convergence window (Epochs 7, 8, 9, 10)
        if epoch in [7, 8, 9, 10]:
            ckpt_path = f"outputs/{model_name}_full_epoch_{epoch}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "train_loss": epoch_loss,
                "model_name": model_name,
                "best_threshold": 0.50
            }, ckpt_path)
            
    # Save final model weights
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epochs,
        "train_loss": epoch_losses[-1],
        "model_name": model_name,
        "best_threshold": 0.50
    }, weights_path)
    print(f"[*] Saved final 100%-data weights to: {weights_path}")
    return epoch_losses


# ------------------------------------------------------------------------------
# 6. RETRAIN SWIN-TINY (BEST ARCHITECTURE) & DENSENET-121 ON 100% DATA
# ------------------------------------------------------------------------------
# 1. Swin-Tiny (Rank #1 Standalone Architecture)
swin_model = SwinClassifier("swin_tiny_patch4_window7_224", pretrained=True).to(device)
swin_losses = train_on_full_dataset(swin_model, "swin_tiny", "outputs/swin_full_weights.pt", epochs=10, lr=1.5e-4)

# 2. DenseNet-121 (Texture Specialist Architecture)
densenet_model = DenseNetClassifier("densenet121", pretrained=True).to(device)
dense_losses = train_on_full_dataset(densenet_model, "densenet121", "outputs/densenet_full_weights.pt", epochs=10, lr=3e-4)

# ------------------------------------------------------------------------------
# 7. 4-PASS TTA INFERENCE ON 3,000 TEST IMAGES
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  RUNNING 4-PASS TTA TEST INFERENCE (3,000 IMAGES)")
print("=" * 70)

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
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        arr = (np.array(img, dtype=np.float32) / 255.0 - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).float(), os.path.splitext(os.path.basename(path))[0]

test_loader = DataLoader(TestDataset(test_files), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

def predict_4pass_tta(model, images):
    with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
        p1 = torch.sigmoid(model(images)).squeeze(-1)
        p2 = torch.sigmoid(model(torch.flip(images, dims=[-1]))).squeeze(-1)
        p3 = torch.sigmoid(model(images * 1.05)).squeeze(-1)
        p4 = torch.sigmoid(model(torch.flip(images, dims=[-1]) * 1.05)).squeeze(-1)
    return (p1 + p2 + p3 + p4) / 4.0

swin_model.eval()
densenet_model.eval()

test_swin_probs, test_dense_probs, test_ids = [], [], []
with torch.inference_mode():
    for images, ids in tqdm(test_loader, desc="Predicting on Test Set"):
        images = images.to(device)
        test_swin_probs.extend(predict_4pass_tta(swin_model, images).cpu().numpy().tolist())
        test_dense_probs.extend(predict_4pass_tta(densenet_model, images).cpu().numpy().tolist())
        test_ids.extend(ids)

test_swin_probs = np.array(test_swin_probs)
test_dense_probs = np.array(test_dense_probs)

# ------------------------------------------------------------------------------
# 8. ENSEMBLE RANK AVERAGING & OUTPUT EXPORT
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  COMPUTING RANK-AVERAGED ENSEMBLE ON 100%-DATA RETRAINED MODELS")
print("=" * 70)

# Mathematical Rank Averaging (ROC Curve Optimal)
r_swin = rankdata(test_swin_probs) / len(test_swin_probs)
r_dense = rankdata(test_dense_probs) / len(test_dense_probs)
final_conf = 0.55 * r_swin + 0.45 * r_dense
final_conf = np.clip(final_conf, 0.001, 0.999)
final_labels = (final_conf >= 0.50).astype(int)

# Export primary predictions.csv
predictions_df = pd.DataFrame({
    "image_id": test_ids,
    "label": final_labels,
    "confidence": np.round(final_conf, 4)
})
predictions_df.to_csv("predictions.csv", index=False)
print(f"[+] 'predictions.csv' generated ({len(predictions_df)} rows)!")
print(f"[+] Class Breakdown:\n{predictions_df['label'].value_counts().to_dict()}")

# Export standalone individual predictions
pd.DataFrame({"image_id": test_ids, "label": (test_swin_probs >= 0.5).astype(int), "confidence": np.round(test_swin_probs, 4)}).to_csv("outputs/predictions_swin.csv", index=False)
pd.DataFrame({"image_id": test_ids, "label": (test_dense_probs >= 0.5).astype(int), "confidence": np.round(test_dense_probs, 4)}).to_csv("outputs/predictions_densenet.csv", index=False)

# Copy the best single model weights (Swin-Tiny) to 'outputs/model_weights.pt'
shutil.copyfile("outputs/swin_full_weights.pt", "outputs/model_weights.pt")
print("[+] Top single model weights saved to 'outputs/model_weights.pt'")

# Plot training curves
plt.figure(figsize=(10, 4))
plt.plot(range(1, 11), swin_losses, label="Swin-Tiny (100% Data Loss)", marker="o", color="#2ca02c")
plt.plot(range(1, 11), dense_losses, label="DenseNet-121 (100% Data Loss)", marker="s", color="#1f77b4")
plt.title("Full Dataset (7,000 Images) Retraining Convergence")
plt.xlabel("Epoch")
plt.ylabel("Train Loss (BCEWithLogits)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=300)
print("[+] Updated retraining curve saved to 'outputs/training_curves.png'")

print("\n" + "=" * 70)
print("  FULL-DATASET RETRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("  Files Ready for Download:")
print("    1. predictions.csv")
print("    2. outputs/model_weights.pt")
print("    3. outputs/training_curves.png")
print("=================================================================")
