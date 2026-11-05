"""
Ensemble Blender: Combines predictions from multiple backbones (e.g., EfficientNet-B3 + ConvNeXt-Tiny).
Calculates weighted probability averages and saves the updated predictions.csv.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def blend_predictions(pred_files: list[str], weights: list[float] = None, output_file: str = "predictions_ensemble.csv"):
    if weights is None:
        weights = [1.0 / len(pred_files)] * len(pred_files)
    
    # Normalize weights
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    
    print(f"Blending {len(pred_files)} prediction files with weights: {weights}")
    
    dfs = [pd.read_csv(f) for f in pred_files]
    
    # Verify ID alignment
    base_ids = dfs[0]["image_id"].tolist()
    for i, df in enumerate(dfs[1:], start=1):
        if df["image_id"].tolist() != base_ids:
            raise ValueError(f"Image IDs in {pred_files[i]} do not match {pred_files[0]}!")
            
    # Calculate weighted probabilities
    blended_probs = np.zeros(len(base_ids), dtype=np.float64)
    for df, w in zip(dfs, weights):
        blended_probs += df["confidence"].values * w
        
    blended_probs = np.clip(blended_probs, 0.0, 1.0)
    blended_labels = (blended_probs >= 0.50).astype(int)
    
    out_df = pd.DataFrame({
        "image_id": base_ids,
        "label": blended_labels,
        "confidence": np.round(blended_probs, 4)
    })
    
    out_df.to_csv(output_file, index=False)
    print(f"\nBlended predictions successfully exported to: {output_file}")
    print(f"Total rows: {len(out_df)}")
    print(f"Class distribution:\n{out_df['label'].value_counts().to_dict()}")
    print("\nFirst 5 rows:")
    print(out_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, help="List of prediction CSV paths to blend")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Weights for each file")
    parser.add_argument("--output", default="predictions_ensemble.csv", help="Output blended predictions CSV path")
    args = parser.parse_args()
    
    blend_predictions(args.files, args.weights, args.output)
