"""
Dual-Stream Spatial + Frequency (FFT) Architecture for Deepfake Detection
--------------------------------------------------------------------------
Stream 1 (Spatial Domain): EfficientNet-B3 extracting RGB facial textures & blending borders.
Stream 2 (Frequency Domain): ResNet-18 extracting Fast Fourier Transform (FFT) magnitude
spectra to detect periodic checkerboard noise & generator-specific frequency spikes.
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
    Computes the 2D Fast Fourier Transform (FFT) log-magnitude spectrum of an image.
    Outputs a 3-channel normalized frequency map: shape (3, target_size, target_size).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if gray.shape[0] != target_size or gray.shape[1] != target_size:
        gray = cv2.resize(gray, (target_size, target_size))

    # Compute 2D Fast Fourier Transform
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20.0 * np.log(np.abs(fshift) + 1e-8)

    # Normalize magnitude spectrum to [0, 1]
    min_val, max_val = np.min(magnitude_spectrum), np.max(magnitude_spectrum)
    if max_val - min_val > 1e-6:
        magnitude_norm = (magnitude_spectrum - min_val) / (max_val - min_val)
    else:
        magnitude_norm = np.zeros_like(magnitude_spectrum)

    # Replicate across 3 channels for standard CNN backbone ingestion
    fft_3ch = np.stack([magnitude_norm] * 3, axis=-1).astype(np.float32)

    # Normalize using ImageNet statistics
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    fft_norm = (fft_3ch - mean) / std

    # HWC -> CHW
    tensor = torch.from_numpy(fft_norm).permute(2, 0, 1).float()
    return tensor


class DualStreamClassifier(nn.Module):
    """
    Dual-Stream Network:
      - Spatial Stream: EfficientNet-B3 (1536 features)
      - Frequency Stream: ResNet-18 (512 features)
      - Fusion: Concatenation (2048 features) -> Regularized Head -> 1 Logit
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
        # Stream 1: Spatial Stream
        self.spatial_backbone = timm.create_model(
            spatial_model,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate
        )
        spatial_dim = self.spatial_backbone.num_features

        # Stream 2: Frequency (FFT) Stream
        self.freq_backbone = timm.create_model(
            freq_model,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate
        )
        freq_dim = self.freq_backbone.num_features

        total_features = spatial_dim + freq_dim

        # Fusion Classification Head
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(total_features, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.SiLU(),
            nn.Dropout(p=drop_rate / 2.0),
            nn.Linear(fusion_dim, 1)
        )

    def forward(self, x_spatial: torch.Tensor, x_freq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass accepting both spatial RGB tensors and FFT frequency tensors.
        Both inputs are shape (B, 3, 224, 224).
        """
        feat_spatial = self.spatial_backbone(x_spatial)  # (B, 1536)
        feat_freq = self.freq_backbone(x_freq)           # (B, 512)

        fused = torch.cat([feat_spatial, feat_freq], dim=1)  # (B, 2048)
        logits = self.fusion_head(fused)                     # (B, 1)
        return logits

    def predict_proba(self, x_spatial: torch.Tensor, x_freq: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x_spatial, x_freq)
        return torch.sigmoid(logits)
