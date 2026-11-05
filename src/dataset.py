import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

# Check for albumentations availability
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

# ImageNet statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224):
    """
    Data augmentations targeting synthetic artifact robustness:
    - Geometric shifts: Flips, rotation, scaling
    - Frequency/Compression degradation: JPEG compression, Gaussian blur
    - Photometric variability: Color jitter, coarse dropout
    """
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            # Degradation augmentations to prevent overfitting to generator artifacts
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        # Fallback basic numpy/PIL transform if albumentations is missing
        return None


def get_valid_transforms(image_size: int = 224):
    """
    Deterministic transforms for validation and test inference.
    """
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return None


def fallback_transform(image: np.ndarray, is_train: bool = False, image_size: int = 224) -> torch.Tensor:
    """Fallback manual normalization when albumentations is not installed."""
    if image.shape[0] != image_size or image.shape[1] != image_size:
        image = cv2.resize(image, (image_size, image_size))
    if is_train and np.random.rand() > 0.5:
        image = np.fliplr(image).copy()
    image = image.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    image = (image - mean) / std
    # HWC to CHW
    tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    return tensor


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for Deepfake Detection.
    Supports:
      - Training (returns image, label, image_id)
      - Validation (returns image, label, image_id)
      - Unlabelled Test (returns image, image_id)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        is_test: bool = False,
        image_size: int = 224
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        self.image_size = image_size
        self.has_label = "label" in self.df.columns and not self.is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        image_id = str(row["image_id"])

        # Load image via cv2 (RGB mode)
        image = cv2.imread(file_path)
        if image is None:
            # Fallback to PIL if cv2 read fails or path has unusual characters
            pil_img = Image.open(file_path).convert("RGB")
            image = np.array(pil_img)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            image_tensor = fallback_transform(
                image,
                is_train=(not self.is_test),
                image_size=self.image_size
            )

        if self.has_label:
            label = float(row["label"])
            label_tensor = torch.tensor([label], dtype=torch.float32)
            return image_tensor, label_tensor, image_id
        else:
            return image_tensor, image_id

