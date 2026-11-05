import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import DeepfakeDataset, get_train_transforms, get_valid_transforms
from src.model import DeepfakeClassifier, get_loss_fn
from src.utils import (
    seed_everything,
    AverageMeter,
    calculate_metrics,
    find_best_threshold
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True
) -> float:
    """Trains the model for one epoch."""
    model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))
        pbar.set_postfix(train_loss=f"{loss_meter.avg:.4f}")

    return loss_meter.avg


@torch.inference_mode()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> tuple[float, dict, np.ndarray, np.ndarray]:
    """Validates the model across the entire validation dataset."""
    model.eval()
    loss_meter = AverageMeter()
    all_targets = []
    all_probs = []

    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

        loss_meter.update(loss.item(), images.size(0))
        all_targets.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_targets = np.vstack(all_targets).ravel()
    all_probs = np.vstack(all_probs).ravel()

    # Find optimal threshold on validation predictions
    best_thresh, _ = find_best_threshold(all_targets, all_probs)
    metrics = calculate_metrics(all_targets, all_probs, threshold=best_thresh)

    return loss_meter.avg, metrics, all_targets, all_probs


def plot_and_save_curves(history: list[dict], output_path: str):
    """Generates and saves loss and AUC-ROC curves for report/deliverables."""
    df_hist = pd.DataFrame(history)
    epochs = df_hist["epoch"].values

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, df_hist["train_loss"], label="Train Loss", marker="o", color="#1f77b4")
    plt.plot(epochs, df_hist["val_loss"], label="Val Loss", marker="s", color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Loss Curves")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Validation AUC-ROC Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, df_hist["val_auc"], label="Val AUC-ROC", marker="^", color="#2ca02c")
    plt.plot(epochs, df_hist["val_f1"], label="Val F1", marker="d", color="#d62728", linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Performance (AUC-ROC & F1)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {output_path}")


def run_training(args):
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    use_amp = (device.type == "cuda")
    print(f"Device: {device} | Using Mixed Precision (AMP): {use_amp}")

    # 1. Load Data Splits
    print(f"Loading folds dataset from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    train_df = df[df["fold"] != args.fold].reset_index(drop=True)
    val_df = df[df["fold"] == args.fold].reset_index(drop=True)

    print(f"Fold {args.fold} Setup: Train samples = {len(train_df)} | Val samples = {len(val_df)}")

    # 2. Datasets and Loaders
    train_transform = get_train_transforms(image_size=args.image_size)
    val_transform = get_valid_transforms(image_size=args.image_size)

    train_dataset = DeepfakeDataset(train_df, transform=train_transform, is_test=False)
    val_dataset = DeepfakeDataset(val_df, transform=val_transform, is_test=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    # 3. Model & Optimization
    print(f"Initializing model backbone: {args.model_name}...")
    model = DeepfakeClassifier(
        model_name=args.model_name,
        pretrained=args.pretrained,
        drop_rate=args.drop_rate
    )
    model.to(device)

    criterion = get_loss_fn()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_auc = 0.0
    best_threshold = 0.5
    best_weights_path = os.path.join(args.output_dir, f"{args.model_name}_fold{args.fold}_best.pt")
    history = []

    print("\n" + "=" * 70)
    print(f"Starting Training: {args.epochs} Epochs | Model: {args.model_name} | Fold: {args.fold}")
    print("=" * 70)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp
        )

        val_loss, metrics, _, _ = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": metrics["auc"],
            "val_f1": metrics["f1"],
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "best_threshold": metrics["threshold"],
            "lr": current_lr,
            "time_sec": elapsed
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {metrics['auc']:.4f} | "
            f"Val F1: {metrics['f1']:.4f} (thresh={metrics['threshold']:.2f}) | "
            f"Val Acc: {metrics['accuracy']:.4f}"
        )

        # Checkpoint: Save model if validation AUC-ROC improves
        if metrics["auc"] > best_auc:
            improvement = metrics["auc"] - best_auc
            best_auc = metrics["auc"]
            best_threshold = metrics["threshold"]

            checkpoint = {
                "epoch": epoch,
                "model_name": args.model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "best_threshold": best_threshold,
                "metrics": metrics,
                "args": vars(args)
            }
            torch.save(checkpoint, best_weights_path)
            print(f"  --> [*] Best model saved! (AUC: {best_auc:.4f}, +{improvement:.4f}) to {best_weights_path}")

    total_time = (time.time() - start_time) / 60.0
    print("\n" + "=" * 70)
    print(f"Training Complete in {total_time:.2f} minutes.")
    print(f"Highest Validation AUC-ROC: {best_auc:.4f} (Optimal Threshold: {best_threshold:.2f})")
    print(f"Best Weights: {best_weights_path}")
    print("=" * 70)

    # 4. Save history and generate curves
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(args.output_dir, f"{args.model_name}_fold{args.fold}_history.csv")
    history_df.to_csv(history_csv, index=False)
    print(f"Training history saved to: {history_csv}")

    curves_png = os.path.join(args.output_dir, f"{args.model_name}_fold{args.fold}_curves.png")
    plot_and_save_curves(history, curves_png)

    # Also save a standard copy named 'model_weights.pt' in output_dir for convenient submission
    standard_weights_path = os.path.join(args.output_dir, "model_weights.pt")
    if os.path.exists(best_weights_path):
        import shutil
        shutil.copyfile(best_weights_path, standard_weights_path)
        print(f"Standard submission weights copied to: {standard_weights_path}")


def get_args():
    parser = argparse.ArgumentParser(description="Deepfake ML Hackathon Model Training")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold index (0-4)")
    parser.add_argument("--model_name", type=str, default="efficientnet_b3", help="timm model architecture")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Load ImageNet pretrained weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for Cosine Annealing")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for regularization")
    parser.add_argument("--drop_rate", type=float, default=0.3, help="Head dropout rate")
    parser.add_argument("--image_size", type=int, default=224, help="Input resolution (224)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num workers (e.g., 2 or 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Target device: 'cuda' or 'cpu'")
    parser.add_argument("--data_file", type=str, default="hackathon_dataset/train_folds.csv", help="Path to folds CSV")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for weights and logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_training(args)

