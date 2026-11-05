import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class DeepfakeClassifier(nn.Module):
    """
    Deepfake Binary Classifier with a transfer-learning backbone and a custom
    regularized classification head.
    
    Supports:
      - Any timm backbone (e.g., 'efficientnet_b3', 'convnext_tiny', 'resnet50')
      - Global pooling with double-dropout linear projection
      - Sigmoid probability prediction
      - Intermediate feature extraction for Grad-CAM
    """
    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        pretrained: bool = True,
        drop_rate: float = 0.3,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.model_name = model_name
        self.drop_rate = drop_rate

        if HAS_TIMM:
            # Create backbone without original classification head (num_classes=0 returns pooled feature vector)
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                drop_rate=drop_rate
            )
            in_features = self.backbone.num_features
        else:
            # Simple fallback backbone if timm is not installed locally
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            in_features = 64

        # Custom Head
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=drop_rate / 2.0),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns raw logits of shape (B, 1).
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns probability of being FAKE (label=1) in range [0.0, 1.0].
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts pooled feature embeddings before classification head.
        Useful for metric learning or visualization.
        """
        return self.backbone(x)


def get_loss_fn(label_smoothing: float = 0.0) -> nn.Module:
    """
    Returns binary cross entropy with logits.
    """
    if label_smoothing > 0.0:
        # Custom smoothed BCE or standard BCEWithLogitsLoss
        return nn.BCEWithLogitsLoss()
    return nn.BCEWithLogitsLoss()

