import os
import shutil
import hashlib
import pandas as pd

def get_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

SUB_DIR = "submission_package"
if os.path.exists(SUB_DIR):
    shutil.rmtree(SUB_DIR)
os.makedirs(SUB_DIR, exist_ok=True)
os.makedirs(os.path.join(SUB_DIR, "outputs"), exist_ok=True)

files_to_copy = [
    ("predictions.csv", "predictions.csv"),
    ("inference_script.py", "inference_script.py"),
    ("requirements.txt", "requirements.txt"),
    ("TECHNICAL_REPORT.md", "TECHNICAL_REPORT.md"),
    ("README.md", "README.md"),
    ("outputs/model_weights.pt", "outputs/model_weights.pt"),
    ("outputs/densenet_weights.pt", "outputs/densenet_weights.pt"),
    ("outputs/gradcam_visualizations.png", "outputs/gradcam_visualizations.png"),
    ("outputs/model_weights.onnx", "outputs/model_weights.onnx"),
    ("outputs/model_weights.onnx.data", "outputs/model_weights.onnx.data"),
    ("outputs/model_weights_traced.pt", "outputs/model_weights_traced.pt"),
]

manifest = []

print("=" * 70)
print("PACKAGING OFFICIAL SUBMISSION BUNDLE")
print("=" * 70)

for src, dst in files_to_copy:
    if os.path.exists(src):
        dest_path = os.path.join(SUB_DIR, dst)
        shutil.copy2(src, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        sha = get_sha256(dest_path)[:12]
        manifest.append({
            "File": dst,
            "Size (MB)": f"{size_mb:.2f} MB",
            "SHA256 (Prefix)": sha,
            "Status": "READY"
        })
        print(f"[+] Packaged: {dst:<35} | {size_mb:>8.2f} MB | SHA: {sha}")
    else:
        print(f"[-] WARNING: Missing {src}")

df_manifest = pd.DataFrame(manifest)
df_manifest.to_csv(os.path.join(SUB_DIR, "MANIFEST.csv"), index=False)
print("\n" + "=" * 70)
print(f"Submission Package successfully prepared in '{SUB_DIR}/'!")
print("=" * 70)

