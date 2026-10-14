"""audit_labels_cleanlab.py — Detect mislabeled samples with Cleanlab.

Pipeline:
    1. Load fixmatch_v2 (ResNet18) checkpoint.
    2. Extract 512-dim penultimate features for all train images.
    3. Fit a Logistic Regression classifier with cross-validation.
    4. Run Cleanlab's `find_label_issues` to rank suspicious samples.
    5. Save JSON + gallery of top-20 suspects.

Requirement:
    pip install cleanlab

Output:
    artifacts/label_audit/
        ├── cleanlab_issues.json
        ├── suspicious_samples.png
        └── report.md
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

# Cleanlab import (with helpful error message)
try:
    from cleanlab.filter import find_label_issues
    from cleanlab.count import estimate_cv_predicted_probabilities
except ImportError:
    raise ImportError(
        "cleanlab is not installed. Please run:\n"
        "    pip install cleanlab\n"
        "then re-run this script."
    )

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from config import (
    CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS,
    TRAIN_DIR, VAL_FRACTION, SEED,
)
from data import IndexedImageDataset, eval_transform, stratified_val_split
from audit import list_split
from analysis import load_checkpoint


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(ARTIFACTS, "label_audit")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, loader, device):
    """Return (features, labels) for a loader, using penultimate layer."""
    model.eval()
    all_feats, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        # ResNet18 forward up to avgpool
        f = model.conv1(x)
        f = model.bn1(f); f = model.relu(f); f = model.maxpool(f)
        f = model.layer1(f); f = model.layer2(f)
        f = model.layer3(f); f = model.layer4(f)
        f = model.avgpool(f).flatten(1)
        all_feats.append(f.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Gallery of suspicious samples
# ---------------------------------------------------------------------------
def save_suspect_gallery(items, issue_indices, save_path, n=20):
    """Show the top-N most suspicious images with their (suspected) wrong label."""
    fig, axes = plt.subplots(4, 5, figsize=(18, 14))
    axes = axes.flatten()

    for i, idx in enumerate(issue_indices[:n]):
        path, cls = items[idx]
        try:
            img = Image.open(path).convert("RGB").resize((224, 224))
            axes[i].imshow(img)
            axes[i].set_title(
                f"#{i+1}\nLabel: {cls}\n{os.path.basename(path)[:18]}",
                fontsize=7, color="red"
            )
            axes[i].axis("off")
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error:\n{e}", ha="center")
            axes[i].axis("off")

    for j in range(len(issue_indices[:n]), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Top-{n} Suspicious Label Samples (Cleanlab)",
                 fontsize=14)
    fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  Saved gallery → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("LABEL AUDIT WITH CLEANLAB")
    print("=" * 70)

    # --- 1) Load data ---
    print("\n[1/5] Loading training data ...")
    items = list_split(TRAIN_DIR)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    items = [(p, c) for p, c in items if c in class_to_idx]
    print(f"  Total labeled items: {len(items)}")

    # Use all labeled data (no val split) for label auditing
    ds = IndexedImageDataset(items, class_to_idx, eval_transform())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    # --- 2) Load model and extract features ---
    print("\n[2/5] Extracting features with fixmatch_v2 ...")
    model, _ = load_checkpoint("fixmatch_v2", "resnet")
    feats, labels = extract_features(model, loader, DEVICE)
    print(f"  Features: {feats.shape}, Labels: {labels.shape}")

    # --- 3) Cross-validated probabilities ---
    print("\n[3/5] Computing cross-validated probabilities ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(
        max_iter=1000, C=1.0,
        n_jobs=1, random_state=SEED,
    )
    # get_cross_validated_predicted_probabilities returns (N, C)
    pred_probs = estimate_cv_predicted_probabilities(
        X=feats, labels=labels,
        clf=clf, cv_n_folds=5, seed=SEED,
    )
    print(f"  Predicted probs shape: {pred_probs.shape}")

    # --- 4) Find label issues ---
    print("\n[4/5] Finding label issues with Cleanlab ...")
    issue_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    print(f"  Found {len(issue_indices)} suspicious samples")

    # --- 5) Save results ---
    print("\n[5/5] Saving results ...")

    # Build a compact list of issues
    issues_list = []
    for rank, idx in enumerate(issue_indices[:50], start=1):
        path, cls = items[idx]
        # Predicted class = argmax of pred_probs
        pred_cls = CLASSES[int(np.argmax(pred_probs[idx]))]
        given_prob = float(pred_probs[idx][class_to_idx[cls]])
        pred_prob = float(pred_probs[idx].max())
        issues_list.append({
            "rank": rank,
            "path": path,
            "given_label": cls,
            "model_prediction": pred_cls,
            "given_label_prob": given_prob,
            "predicted_label_prob": pred_prob,
        })

    with open(os.path.join(OUT_DIR, "cleanlab_issues.json"), "w") as f:
        json.dump({
            "n_total": len(items),
            "n_issues": len(issue_indices),
            "issue_rate": len(issue_indices) / len(items),
            "top_50": issues_list,
        }, f, indent=2)

    # Gallery of top-20
    save_suspect_gallery(
        items, issue_indices,
        os.path.join(OUT_DIR, "suspicious_samples.png"),
        n=20,
    )

    # Markdown report
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("# Label Audit Report (Cleanlab)\n\n")
        f.write(f"- Total labeled samples: **{len(items)}**\n")
        f.write(f"- Suspicious samples: **{len(issue_indices)}** "
                f"({100 * len(issue_indices) / len(items):.2f}%)\n\n")
        f.write("## Top-20 Suspicious Samples\n\n")
        f.write("| # | Path | Given | Model says | P(given) | P(pred) |\n")
        f.write("|---|---|---|---|---:|---:|\n")
        for it in issues_list[:20]:
            f.write(f"| {it['rank']} | {os.path.basename(it['path'])} | "
                    f"{it['given_label']} | {it['model_prediction']} | "
                    f"{it['given_label_prob']:.3f} | "
                    f"{it['predicted_label_prob']:.3f} |\n")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total: {len(items)}")
    print(f"  Issues: {len(issue_indices)} "
          f"({100 * len(issue_indices) / len(items):.2f}%)")
    print(f"\n  Output: {OUT_DIR}")
    print(f"  ⚠ Inspect the gallery and cross-check with investigate_ambulance.py")


if __name__ == "__main__":
    main()