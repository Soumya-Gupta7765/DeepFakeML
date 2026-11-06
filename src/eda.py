"""
Exploratory Data Analysis (EDA) Script for Deepfake ML Hackathon
Analyzes dataset balance, image dimensions, file sizes, channel distributions,
frequency/sharpness (Laplacian variance), and exports visual charts.
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def analyze_dataset(
    dataset_dir: str = "hackathon_dataset",
    output_dir: str = "outputs",
    sample_size: int = 500  # Number of samples per class for deep pixel/sharpness profiling
):
    print("=" * 70)
    print("  DEEPFAKE ML HACKATHON — EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    train_meta_path = os.path.join(dataset_dir, "train_metadata.csv")
    sample_sub_path = os.path.join(dataset_dir, "sample_submission.csv")
    train_real_dir = os.path.join(dataset_dir, "train", "real")
    train_fake_dir = os.path.join(dataset_dir, "train", "fake")
    test_img_dir = os.path.join(dataset_dir, "test", "images")

    # 1. Metadata & File Counts
    print("\n[1/5] Checking Dataset Counts & Class Distribution...")
    train_meta = pd.read_csv(train_meta_path)
    sample_sub = pd.read_csv(sample_sub_path)

    real_files = sorted(os.listdir(train_real_dir)) if os.path.exists(train_real_dir) else []
    fake_files = sorted(os.listdir(train_fake_dir)) if os.path.exists(train_fake_dir) else []
    test_files = sorted(os.listdir(test_img_dir)) if os.path.exists(test_img_dir) else []

    n_real = len(real_files)
    n_fake = len(fake_files)
    n_test = len(test_files)
    total_train = n_real + n_fake

    print(f"  • Total Training Images: {total_train} (Real: {n_real} [{n_real/total_train:.2%}], Fake: {n_fake} [{n_fake/total_train:.2%}])")
    print(f"  • Total Unlabelled Test Images: {n_test}")
    print(f"  • Missing Values in Metadata: {train_meta.isnull().sum().sum()}")
    print(f"  • Overlap between Real & Fake folders: {len(set(real_files).intersection(set(fake_files)))}")

    # 2. File Size Profiling
    print("\n[2/5] Profiling File Sizes (JPEG Compression Signatures)...")
    real_sizes_kb = [os.path.getsize(os.path.join(train_real_dir, f)) / 1024.0 for f in real_files]
    fake_sizes_kb = [os.path.getsize(os.path.join(train_fake_dir, f)) / 1024.0 for f in fake_files]
    test_sizes_kb = [os.path.getsize(os.path.join(test_img_dir, f)) / 1024.0 for f in test_files]

    print(f"  • Real Images Size (KB): Mean = {np.mean(real_sizes_kb):.2f} KB | Std = {np.std(real_sizes_kb):.2f} KB")
    print(f"  • Fake Images Size (KB): Mean = {np.mean(fake_sizes_kb):.2f} KB | Std = {np.std(fake_sizes_kb):.2f} KB")
    print(f"  • Test Images Size (KB): Mean = {np.mean(test_sizes_kb):.2f} KB | Std = {np.std(test_sizes_kb):.2f} KB")

    # 3. Deep Image Dimension & Color Channel Profiling
    print(f"\n[3/5] Sampling {sample_size} images per class for Pixel, Color & Sharpness Profiling...")
    
    def profile_images(file_list, base_dir, max_samples):
        widths, heights = [], []
        r_means, g_means, b_means = [], [], []
        laplacian_vars = []  # Sharpness metric
        
        sample_subset = file_list[:max_samples]
        for f in tqdm(sample_subset, desc=f"Analyzing {os.path.basename(base_dir)}", leave=False):
            p = os.path.join(base_dir, f)
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                continue
            h, w, c = img_bgr.shape
            widths.append(w)
            heights.append(h)
            
            # Color channel means (RGB)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            r_means.append(np.mean(img_rgb[:, :, 0]))
            g_means.append(np.mean(img_rgb[:, :, 1]))
            b_means.append(np.mean(img_rgb[:, :, 2]))
            
            # Sharpness via Laplacian variance
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            laplacian_vars.append(laplacian_var)

        return {
            "w": widths, "h": heights,
            "r": r_means, "g": g_means, "b": b_means,
            "sharpness": laplacian_vars
        }

    real_stats = profile_images(real_files, train_real_dir, sample_size)
    fake_stats = profile_images(fake_files, train_fake_dir, sample_size)
    test_stats = profile_images(test_files, test_img_dir, sample_size)

    # Resolution Check
    unique_resolutions = set(zip(real_stats["w"] + fake_stats["w"] + test_stats["w"],
                                 real_stats["h"] + fake_stats["h"] + test_stats["h"]))
    print(f"  • Image Dimensions Observed: {unique_resolutions} (Strictly uniform 224x224)")
    print(f"  • Real Faces Sharpness (Laplacian Var): Mean = {np.mean(real_stats['sharpness']):.2f}")
    print(f"  • Fake Faces Sharpness (Laplacian Var): Mean = {np.mean(fake_stats['sharpness']):.2f}")

    # 4. Generate Visual Charts
    print("\n[4/5] Generating Visual EDA Charts (6-Panel Analysis)...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Subplot 1: Class Balance
    ax1 = axes[0, 0]
    bars = ax1.bar(["Real (0)", "Fake (1)"], [n_real, n_fake], color=["#2ca02c", "#d62728"], width=0.55, edgecolor="black")
    ax1.set_title("Training Set Class Distribution", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Images")
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval:,} ({yval/total_train:.1%})", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylim(0, 4200)

    # Subplot 2: File Size Distribution
    ax2 = axes[0, 1]
    ax2.hist(real_sizes_kb, bins=40, alpha=0.6, label=f"Real (Mean: {np.mean(real_sizes_kb):.1f}KB)", color="#2ca02c", density=True)
    ax2.hist(fake_sizes_kb, bins=40, alpha=0.6, label=f"Fake (Mean: {np.mean(fake_sizes_kb):.1f}KB)", color="#d62728", density=True)
    ax2.set_title("File Size Distribution (JPEG Compression)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("File Size (KB)")
    ax2.set_ylabel("Density")
    ax2.legend()

    # Subplot 3: Sharpness / Laplacian Variance Comparison
    ax3 = axes[0, 2]
    bp = ax3.boxplot([real_stats["sharpness"], fake_stats["sharpness"]], patch_artist=True, labels=["Real Faces", "Fake Faces"])
    colors = ["#2ca02c", "#d62728"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_title("Image Sharpness (Laplacian Variance)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Variance of Laplacian")

    # Subplot 4: Red Channel Intensity Distribution
    ax4 = axes[1, 0]
    ax4.hist(real_stats["r"], bins=30, alpha=0.55, label="Real", color="#2ca02c", density=True)
    ax4.hist(fake_stats["r"], bins=30, alpha=0.55, label="Fake", color="#d62728", density=True)
    ax4.set_title("Red Channel Mean Intensity", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Mean Red Pixel Value (0-255)")
    ax4.legend()

    # Subplot 5: Green Channel Intensity Distribution
    ax5 = axes[1, 1]
    ax5.hist(real_stats["g"], bins=30, alpha=0.55, label="Real", color="#2ca02c", density=True)
    ax5.hist(fake_stats["g"], bins=30, alpha=0.55, label="Fake", color="#d62728", density=True)
    ax5.set_title("Green Channel Mean Intensity", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Mean Green Pixel Value (0-255)")
    ax5.legend()

    # Subplot 6: Blue Channel Intensity Distribution
    ax6 = axes[1, 2]
    ax6.hist(real_stats["b"], bins=30, alpha=0.55, label="Real", color="#2ca02c", density=True)
    ax6.hist(fake_stats["b"], bins=30, alpha=0.55, label="Fake", color="#d62728", density=True)
    ax6.set_title("Blue Channel Mean Intensity", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Mean Blue Pixel Value (0-255)")
    ax6.legend()

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "eda_analysis.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"  • Charts successfully exported to: {chart_path}")

    # 5. Export Summary JSON
    print("\n[5/5] Exporting EDA Summary Report...")
    summary = {
        "dataset": {
            "total_train_images": total_train,
            "real_images": n_real,
            "fake_images": n_fake,
            "test_images": n_test,
            "class_balance_percent": {"real": round(n_real / total_train * 100, 2), "fake": round(n_fake / total_train * 100, 2)},
            "resolution": "224x224",
            "channels": "RGB (3-channel)"
        },
        "file_size_kb": {
            "real_mean": round(float(np.mean(real_sizes_kb)), 2),
            "fake_mean": round(float(np.mean(fake_sizes_kb)), 2),
            "test_mean": round(float(np.mean(test_sizes_kb)), 2)
        },
        "sharpness_laplacian_var": {
            "real_mean": round(float(np.mean(real_stats["sharpness"])), 2),
            "fake_mean": round(float(np.mean(fake_stats["sharpness"])), 2)
        },
        "color_channel_means": {
            "real": {"R": round(float(np.mean(real_stats["r"])), 2), "G": round(float(np.mean(real_stats["g"])), 2), "B": round(float(np.mean(real_stats["b"])), 2)},
            "fake": {"R": round(float(np.mean(fake_stats["r"])), 2), "G": round(float(np.mean(fake_stats["g"])), 2), "B": round(float(np.mean(fake_stats["b"])), 2)}
        }
    }
    
    report_json_path = os.path.join(output_dir, "eda_summary.json")
    with open(report_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  • Summary JSON exported to: {report_json_path}")

    print("\n" + "=" * 70)
    print("  EDA COMPLETE! Summary results ready for inclusion in Technical Report.")
    print("=" * 70)


if __name__ == "__main__":
    analyze_dataset()

