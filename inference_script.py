import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------------------------------------------------------------------
# MODEL ARCHITECTURES
# ------------------------------------------------------------------------------
class SwinClassifier(nn.Module):
    """Swin-Tiny Vision Transformer (Rank #1 Standalone Model)."""
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=False, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        in_features = self.backbone.num_features
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
    def __init__(self, model_name="densenet121", pretrained=False, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        in_features = self.backbone.num_features
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


class ConvNeXtClassifier(nn.Module):
    """ConvNeXt-Tiny matching training pipeline."""
    def __init__(self, model_name="convnext_tiny", pretrained=False, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        in_features = self.backbone.num_features
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


class DeepfakeClassifier(nn.Module):
    """Standard CNN classifier fallback (EfficientNet / ConvNeXt / ResNet)."""
    def __init__(self, model_name="efficientnet_b3", pretrained=False, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate
        )
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


# ------------------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------------------
class TestImageDataset(Dataset):
    """Dataset to stream test images directly from disk."""
    def __init__(self, image_files, image_size=224):
        self.image_files = image_files
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = self.image_files[idx]
        image_id = os.path.splitext(os.path.basename(file_path))[0]

        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                if img.size != (self.image_size, self.image_size):
                    img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
        except Exception:
            arr = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        return tensor, image_id


# ------------------------------------------------------------------------------
# MODEL LOADER (SUPPORTS BOTH SINGLE-MODEL & DUAL-MODEL ENSEMBLE CHECKPOINTS)
# ------------------------------------------------------------------------------
def load_checkpoint(weights_path, device):
    """Loads checkpoint from disk and instantiates model(s)."""
    if not os.path.exists(weights_path):
        for candidate in [
            "outputs/model_weights.pt",
            "model_weights.pt",
            "submission_package/outputs/model_weights.pt"
        ]:
            if os.path.exists(candidate):
                print(f"Notice: '{weights_path}' not found, resolving to '{candidate}'")
                weights_path = candidate
                break

    print(f"Loading weights from {weights_path}...")
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location=device)

    # Check if checkpoint is an ensemble bundle
    is_ensemble = False
    if isinstance(checkpoint, dict):
        if checkpoint.get("is_ensemble", False) or ("swin_state_dict" in checkpoint and "densenet_state_dict" in checkpoint):
            is_ensemble = True

    if is_ensemble:
        print("[+] Detected Super-Ensemble Checkpoint (Swin-Tiny ViT + DenseNet-121)!")
        threshold = checkpoint.get("best_threshold", 0.50)

        # 1. Swin-Tiny
        swin_model = SwinClassifier(model_name="swin_tiny_patch4_window7_224", pretrained=False)
        swin_model.load_state_dict(checkpoint["swin_state_dict"])
        swin_model.to(device)
        swin_model.eval()

        # 2. DenseNet-121
        dense_model = DenseNetClassifier(model_name="densenet121", pretrained=False)
        dense_model.load_state_dict(checkpoint["densenet_state_dict"])
        dense_model.to(device)
        dense_model.eval()

        return {
            "is_ensemble": True,
            "swin_model": swin_model,
            "dense_model": dense_model,
            "threshold": threshold
        }

    # Otherwise, single model checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_name = checkpoint.get("model_name", "swin_tiny")
        threshold = checkpoint.get("best_threshold", 0.50)
    else:
        state_dict = checkpoint
        model_name = "swin_tiny"
        threshold = 0.50

    model_name_str = str(model_name).lower()
    if "swin" in model_name_str:
        model = SwinClassifier("swin_tiny_patch4_window7_224", pretrained=False)
    elif "dense" in model_name_str:
        model = DenseNetClassifier("densenet121", pretrained=False)
    elif "convnext" in model_name_str:
        model = ConvNeXtClassifier("convnext_tiny", pretrained=False)
    else:
        model = DeepfakeClassifier(model_name=model_name, pretrained=False)

    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: strict state_dict load failed ({e}), attempting non-strict match...")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    return {
        "is_ensemble": False,
        "model": model,
        "threshold": threshold
    }


# ------------------------------------------------------------------------------
# 8-PASS TEST-TIME AUGMENTATION (TTA)
# ------------------------------------------------------------------------------
def predict_8pass_tta(model, images, device):
    """
    8-Pass Multi-Scale Test-Time Augmentation matching exact training pipeline:
    1. Original
    2. Horizontal Flip
    3. Brightness (+5%)
    4. Flip + Brightness
    5. Contrast Boost (+8%)
    6. Flip + Contrast
    7. Center Zoom (1.06x)
    8. Flip + Zoom
    """
    use_amp = (device.type == "cuda")
    autocast_ctx = torch.amp.autocast('cuda', enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=use_amp)
    with autocast_ctx:
        # 1. Original
        p1 = torch.sigmoid(model(images)).squeeze(-1)
        # 2. Horizontal Flip
        img_f = torch.flip(images, dims=[-1])
        p2 = torch.sigmoid(model(img_f)).squeeze(-1)
        # 3. Brightness (+5%)
        p3 = torch.sigmoid(model(images * 1.05)).squeeze(-1)
        # 4. Flip + Brightness
        p4 = torch.sigmoid(model(img_f * 1.05)).squeeze(-1)
        # 5. Contrast Boost (+8%)
        p5 = torch.sigmoid(model((images - 0.5) * 1.08 + 0.5)).squeeze(-1)
        # 6. Flip + Contrast
        p6 = torch.sigmoid(model((img_f - 0.5) * 1.08 + 0.5)).squeeze(-1)
        # 7. Center Zoom (1.06x)
        h, w = images.shape[-2:]
        scaled = nn.functional.interpolate(images, scale_factor=1.06, mode='bilinear', align_corners=False)
        sh, sw = scaled.shape[-2:]
        dh, dw = (sh - h) // 2, (sw - w) // 2
        img_zoom = scaled[:, :, dh:dh+h, dw:dw+w]
        p7 = torch.sigmoid(model(img_zoom)).squeeze(-1)
        # 8. Flip + Zoom
        p8 = torch.sigmoid(model(torch.flip(img_zoom, dims=[-1]))).squeeze(-1)

    return (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) / 8.0


# ------------------------------------------------------------------------------
# INFERENCE PIPELINE
# ------------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(test_dir, weights_path, output_path, batch_size=64, use_tta=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running inference on device: {device} | TTA (8-Pass Multi-Scale): {use_tta}")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    image_files.sort()

    total_images = len(image_files)
    print(f"[*] Found {total_images} test images in {test_dir}")
    if total_images == 0:
        raise ValueError(f"No test images found in {test_dir}!")

    model_bundle = load_checkpoint(weights_path, device)
    threshold = model_bundle["threshold"]

    dataset = TestImageDataset(image_files, image_size=224)
    workers = 4 if (device.type == "cuda" and os.name != "nt") else 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda")
    )

    all_image_ids = []

    if model_bundle["is_ensemble"]:
        swin_model = model_bundle["swin_model"]
        dense_model = model_bundle["dense_model"]
        all_swin_probs = []
        all_dense_probs = []

        print("[*] Evaluating Dual-Model Super-Ensemble (Swin-Tiny + DenseNet-121)...")
        for images, ids in tqdm(dataloader, desc="Ensemble Inference"):
            images = images.to(device, non_blocking=True)
            if use_tta:
                p_swin = predict_8pass_tta(swin_model, images, device)
                p_dense = predict_8pass_tta(dense_model, images, device)
            else:
                p_swin = torch.sigmoid(swin_model(images)).squeeze(-1)
                p_dense = torch.sigmoid(dense_model(images)).squeeze(-1)

            all_image_ids.extend(ids)
            all_swin_probs.extend(p_swin.cpu().numpy().tolist())
            all_dense_probs.extend(p_dense.cpu().numpy().tolist())

        # Mathematical Rank Averaging (ROC Curve Optimal)
        try:
            from scipy.stats import rankdata
        except ImportError:
            def rankdata(a):
                return np.argsort(np.argsort(a)) + 1.0

        r_swin = rankdata(all_swin_probs) / len(all_swin_probs)
        r_dense = rankdata(all_dense_probs) / len(all_dense_probs)
        confidences = 0.50 * r_swin + 0.50 * r_dense
        confidences = np.clip(confidences, 0.001, 0.999)

    else:
        model = model_bundle["model"]
        all_probs = []

        print("[*] Evaluating Standalone Model...")
        for images, ids in tqdm(dataloader, desc="Inference"):
            images = images.to(device, non_blocking=True)
            if use_tta:
                probs = predict_8pass_tta(model, images, device)
            else:
                probs = torch.sigmoid(model(images)).squeeze(-1)

            all_image_ids.extend(ids)
            all_probs.extend(probs.cpu().numpy().tolist())

        confidences = np.clip(np.array(all_probs, dtype=np.float32), 0.0, 1.0)

    labels = (confidences >= threshold).astype(int)

    df_preds = pd.DataFrame({
        "image_id": all_image_ids,
        "label": labels,
        "confidence": np.round(confidences, 4)
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_preds.to_csv(output_path, index=False)
    print(f"\n[+] Inference Complete! Saved {len(df_preds)} rows to {output_path}")
    print("\nSummary Statistics:")
    print(f"Total predictions: {len(df_preds)}")
    print(f"Class breakdown:\n{df_preds['label'].value_counts().to_dict()}")
    print(f"Confidence summary:\n{df_preds['confidence'].describe()}")
    print("\nFirst 5 rows:")
    print(df_preds.head())


def main():
    parser = argparse.ArgumentParser(description="Deepfake ML Hackathon — Test Set Inference Script")
    parser.add_argument("--test_dir", required=True, help="Path to test images folder")
    parser.add_argument("--weights", default="outputs/model_weights.pt", help="Path to model weight file (.pt)")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path (default: predictions.csv)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference (default: 64)")
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation (TTA)")
    args = parser.parse_args()

    run_inference(
        test_dir=args.test_dir,
        weights_path=args.weights,
        output_path=args.output,
        batch_size=args.batch_size,
        use_tta=(not args.no_tta)
    )


if __name__ == "__main__":
    main()
