"""
Grad-CAM Visualization Generator for Deepfake Detection
Generates attention heatmaps explaining what facial artifacts the model focuses on.
Can run on Kaggle GPU or any machine with PyTorch and OpenCV.
"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import timm

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DeepfakeClassifier(nn.Module):
    def __init__(self, model_name="efficientnet_b3", pretrained=False, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
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


class GradCAM:
    """
    Grad-CAM implementation hooking into the last convolutional feature map.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        prob = torch.sigmoid(output).item()

        # Backward on raw logit
        output.backward()

        # Channel-wise mean of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        # Weight activations by pooled gradients
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        return heatmap, prob


def apply_heatmap(heatmap, original_image, alpha=0.45):
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = np.float32(heatmap_colored) * alpha + np.float32(original_image) * (1 - alpha)
    return np.uint8(overlay)


def run_visualizations(weights_path, sample_images, output_path="outputs/gradcam_visualizations.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model = DeepfakeClassifier("efficientnet_b3", pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Target last conv layer in EfficientNet-B3 (conv_head)
    target_layer = model.backbone.conv_head
    grad_cam = GradCAM(model, target_layer)

    n_samples = len(sample_images)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for idx, (img_path, true_label) in enumerate(sample_images):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))

        norm_arr = (img_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(norm_arr).permute(2, 0, 1).unsqueeze(0).to(device)

        heatmap, pred_prob = grad_cam.generate_heatmap(tensor)
        cam_overlay = apply_heatmap(heatmap, img_resized)

        pred_label = "FAKE" if pred_prob >= 0.5 else "REAL"

        # Column 1: Original
        axes[idx, 0].imshow(img_resized)
        axes[idx, 0].set_title(f"Original ({true_label})", fontsize=11, fontweight="bold")
        axes[idx, 0].axis("off")

        # Column 2: Heatmap
        axes[idx, 1].imshow(heatmap, cmap="jet")
        axes[idx, 1].set_title("Grad-CAM Activation Map", fontsize=11)
        axes[idx, 1].axis("off")

        # Column 3: Overlay
        axes[idx, 2].imshow(cam_overlay)
        axes[idx, 2].set_title(f"Overlay | Pred: {pred_label} ({pred_prob:.2%})", fontsize=11, color="red" if pred_label == "FAKE" else "green")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Grad-CAM visualizations successfully saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="outputs/model_weights.pt")
    parser.add_argument("--output", default="outputs/gradcam_visualizations.png")
    args = parser.parse_args()

    # Pick 2 Real and 2 Fake samples from train
    real_dir = "hackathon_dataset/train/real"
    fake_dir = "hackathon_dataset/train/fake"

    samples = []
    if os.path.exists(real_dir) and os.path.exists(fake_dir):
        real_files = sorted(os.listdir(real_dir))[:2]
        fake_files = sorted(os.listdir(fake_dir))[:2]
        samples = [(os.path.join(real_dir, f), "REAL") for f in real_files] + \
                  [(os.path.join(fake_dir, f), "FAKE") for f in fake_files]
        run_visualizations(args.weights, samples, args.output)
    else:
        print("Sample directories not found locally; run this on Kaggle with dataset attached.")

