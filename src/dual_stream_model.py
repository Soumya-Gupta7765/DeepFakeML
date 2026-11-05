"""
Dual-Stream Spatial + Frequency (FFT) Architecture for Deepfake Detection
Numerically stable version preventing NaNs in FP16 mixed precision.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import timm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def extract_fft_spectrum(image_rgb: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """
    Computes a numerically safe 2D Fast Fourier Transform (FFT) log-magnitude spectrum.
    Guaranteed finite values (no NaNs, no Infs).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if gray.shape[0] != target_size or gray.shape[1] != target_size:
        gray = cv2.resize(gray, (target_size, target_size))

    # Float64 FFT to prevent numerical overflow
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    # log(abs + 1.0) ensures value is strictly non-negative and finite
    mag = 20.0 * np.log(np.abs(fshift) + 1.0)

    min_v, max_v = np.min(mag), np.max(mag)
    diff = max_v - min_v
    if diff > 1e-5:
        norm = (mag - min_v) / diff
    else:
        norm = np.zeros_like(mag)

    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
    fft_3ch = np.stack([norm] * 3, axis=-1)

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    fft_norm = (fft_3ch - mean) / std

    tensor = torch.from_numpy(fft_norm).permute(2, 0, 1).float()
    return torch.nan_to_num(tensor, nan=0.0)


class DualStreamClassifier(nn.Module):
    """
    Dual-Stream Network:
      - Spatial Stream: EfficientNet-B3 (1536 features)
      - Frequency Stream: ResNet-18 (512 features)
      - Fusion: Concatenation (2048) -> LayerNorm -> Regularized Head -> 1 Logit
    """
    def __init__(
        self,
        spatial_model: str = "efficientnet_b3",
        freq_model: str = "resnet18",
        pretrained: bool = True,
        drop_rate: float = 0.3,
        fusion_dim: int = 256
    ):
        super().__init__()
        self.spatial_backbone = timm.create_model(
            spatial_model,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate
        )
        spatial_dim = self.spatial_backbone.num_features

        self.freq_backbone = timm.create_model(
            freq_model,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate
        )
        freq_dim = self.freq_backbone.num_features

        total_features = spatial_dim + freq_dim

        # Stable Fusion Head using LayerNorm instead of BatchNorm1d
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(total_features, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.SiLU(),
            nn.Dropout(p=drop_rate / 2.0),
            nn.Linear(fusion_dim, 1)
        )

    def forward(self, x_spatial: torch.Tensor, x_freq: torch.Tensor) -> torch.Tensor:
        feat_spatial = self.spatial_backbone(x_spatial)
        feat_freq = self.freq_backbone(x_freq)

        fused = torch.cat([feat_spatial, feat_freq], dim=1)
        logits = self.fusion_head(fused)
        return logits

    def predict_proba(self, x_spatial: torch.Tensor, x_freq: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x_spatial, x_freq)
        return torch.sigmoid(logits)
