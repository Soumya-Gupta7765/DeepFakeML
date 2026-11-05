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


class DeepfakeClassifier(nn.Module):
    """
    Deepfake classifier model architecture matching the training pipeline.
    """
    def __init__(self, model_name="efficientnet_b3", pretrained=False, drop_rate=0.3):
        super().__init__()
        # pretrained=False to avoid internet download during evaluation
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


class TestImageDataset(Dataset):
    """
    Dataset to stream test images directly from disk.
    """
    def __init__(self, image_files, image_size=224):
        self.image_files = image_files
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = self.image_files[idx]
        image_id = os.path.splitext(os.path.basename(file_path))[0]

        # Load RGB image
        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                if img.size != (self.image_size, self.image_size):
                    img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            # Fallback for corrupted images: return zero-tensor
            arr = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

        # Standard ImageNet normalization: (x - mean) / std
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        # HWC to CHW
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        return tensor, image_id


def load_model(weights_path, device):
    """
    Loads model checkpoint and restores weights.
    """
    print(f"Loading weights from {weights_path}...")
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        # Compatibility fallback for older PyTorch versions that do not accept weights_only
        checkpoint = torch.load(weights_path, map_location=device)

    # Detect model name & state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_name = checkpoint.get("model_name", "efficientnet_b3")
        threshold = checkpoint.get("best_threshold", 0.5)
        print(f"Detected checkpoint metadata: model={model_name}, optimal_threshold={threshold:.4f}")
    else:
        state_dict = checkpoint
        model_name = "efficientnet_b3"
        threshold = 0.5

    model = DeepfakeClassifier(model_name=model_name, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, threshold


@torch.inference_mode()
def run_inference(test_dir, weights_path, output_path, batch_size=64, num_workers=2, use_tta=True):
    # Determine compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device} | TTA (Test-Time Augmentation): {use_tta}")

    # Gather test images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    # Sort for deterministic ordering
    image_files.sort()

    total_images = len(image_files)
    print(f"Found {total_images} test images in {test_dir}")
    if total_images == 0:
        raise ValueError(f"No test images found in {test_dir}!")

    # Load model
    model, threshold = load_model(weights_path, device)

    # Create dataset & loader
    dataset = TestImageDataset(image_files, image_size=224)
    if os.name == "nt" or device.type == "cpu":
        num_workers = 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    all_image_ids = []
    all_confidences = []

    use_amp = (device.type == "cuda")
    print("Generating predictions...")
    for images, ids in tqdm(dataloader, desc="Inference"):
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Pass 1: Original Image
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig).squeeze(-1)

            if use_tta:
                # Pass 2: Horizontally Flipped Image
                images_flipped = torch.flip(images, dims=[-1])
                logits_flip = model(images_flipped)
                probs_flip = torch.sigmoid(logits_flip).squeeze(-1)
                probs = (probs_orig + probs_flip) / 2.0
            else:
                probs = probs_orig

        all_image_ids.extend(ids)
        all_confidences.extend(probs.cpu().numpy().tolist())

    # Build predictions dataframe
    confidences = np.array(all_confidences, dtype=np.float32)
    # Clip confidence to strict [0.0, 1.0] range
    confidences = np.clip(confidences, 0.0, 1.0)
    labels = (confidences >= threshold).astype(int)

    # Format exactly as required: image_id,label,confidence
    df_preds = pd.DataFrame({
        "image_id": all_image_ids,
        "label": labels,
        "confidence": np.round(confidences, 4)
    })

    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_preds.to_csv(output_path, index=False)

    print(f"\nInference Complete! Saved {len(df_preds)} rows to {output_path}")
    print("\nSummary Statistics:")
    print(f"Total predictions: {len(df_preds)}")
    print(f"Class breakdown:\n{df_preds['label'].value_counts()}")
    print(f"Confidence summary:\n{df_preds['confidence'].describe()}")
    print("\nFirst 5 rows:")
    print(df_preds.head())


def main():
    parser = argparse.ArgumentParser(description="Deepfake ML Hackathon — Test Set Inference Script")
    parser.add_argument("--test_dir", required=True, help="Path to test images folder")
    parser.add_argument("--weights", required=True, help="Path to model weight file (.pt)")
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

