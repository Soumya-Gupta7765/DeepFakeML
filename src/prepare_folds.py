import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def prepare_folds(
    metadata_path: str = "hackathon_dataset/train_metadata.csv",
    train_dir: str = "hackathon_dataset/train",
    output_path: str = "hackathon_dataset/train_folds.csv",
    n_splits: int = 5,
    random_state: int = 42
):
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Verify and attach relative file path
    file_paths = []
    exists_count = 0
    for _, row in df.iterrows():
        subfolder = "real" if row["label"] == 0 else "fake"
        rel_path = os.path.join(train_dir, subfolder, f"{row['image_id']}.jpg")
        file_paths.append(rel_path.replace("\\", "/"))
        if os.path.exists(rel_path):
            exists_count += 1

    df["file_path"] = file_paths
    print(f"Total entries: {len(df)} | Verified files on disk: {exists_count}")

    # Stratified K-Fold split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    # Summary
    print("\nFold distribution:")
    fold_dist = pd.crosstab(df["fold"], df["label"], margins=True)
    print(fold_dist)

    df.to_csv(output_path, index=False)
    print(f"\nSaved folds to {output_path}")

if __name__ == "__main__":
    prepare_folds()

