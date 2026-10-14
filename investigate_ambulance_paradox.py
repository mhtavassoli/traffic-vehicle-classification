"""investigate_ambulance_paradox.py — Why does the model predict blue vans as ambulance?

After FixMatch v2, ambulance recall jumped to 0.99 in test3 (which contains 92% blue vans mislabeled).
This script investigates:
    1. Does merged train contain blue vans labeled as ambulance?
    2. What pseudo-labels does fixmatch_v2 assign to unclean samples?
    3. Which unclean samples get pseudo-labeled as 'ambulance' with high confidence?

Output:
    artifacts/ambulance_paradox/
        ├── unclean_pseudo_ambulance.png
        ├── train_vs_unclean_comparison.png
        └── report.json
"""
import os
import json
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from config import (
    CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS,
    TRAIN_DIR, UNCLEAN_DIR,
)
from data import IndexedImageDataset, eval_transform
from audit import list_split
from analysis import load_checkpoint


OUT_DIR = os.path.join(ARTIFACTS, "ambulance_paradox")
os.makedirs(OUT_DIR, exist_ok=True)


def rgb_stats(path):
    """Return mean R, G, B and blue dominance for an image."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((128, 128))
        arr = np.asarray(im, dtype=np.float32)
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2
    center = arr[cy - h // 4:cy + h // 4, cx - w // 4:cx + w // 4]
    mean_rgb = center.reshape(-1, 3).mean(axis=0)
    r, g, b = center[..., 0], center[..., 1], center[..., 2]
    blue_ratio = float(((b > r + 20) & (b > g + 20)).mean())
    return mean_rgb, blue_ratio


@torch.no_grad()
def compute_pseudo_labels(model, items, class_to_idx, device, threshold=0.95):
    """For each unclean image, compute pseudo-label and confidence."""
    ds = IndexedImageDataset(items, class_to_idx, eval_transform())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    results = []
    model.eval()
    for batch_idx, (x, _) in enumerate(loader):
        x = x.to(device)
        probs = F.softmax(model(x), dim=1)
        conf, pred = probs.max(dim=1)
        for i in range(x.size(0)):
            item_idx = batch_idx * 64 + i
            if item_idx >= len(items):
                break
            path, orig_cls = items[item_idx]
            results.append({
                "path": path,
                "original_label": orig_cls,
                "pseudo_label": CLASSES[pred[i].item()],
                "confidence": float(conf[i].item()),
                "above_threshold": bool(conf[i].item() >= threshold),
            })
    return results


def save_gallery(paths, save_path, n=12, title=""):
    """Save a gallery of n images."""
    paths = paths[:n]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    for idx, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB").resize((224, 224))
            axes[idx].imshow(img)
            axes[idx].set_title(os.path.basename(p)[:22], fontsize=7)
            axes[idx].axis("off")
        except Exception as e:
            axes[idx].text(0.5, 0.5, str(e), ha="center")
            axes[idx].axis("off")
    for i in range(len(paths), len(axes)):
        axes[i].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  Saved gallery -> {save_path}")


def main():
    print("=" * 70)
    print("AMBULANCE PARADOX INVESTIGATION")
    print("=" * 70)

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    # --- 1) Check merged train / ambulance for blue vans ---
    print("\n[1/4] Analyzing merged train / ambulance ...")
    train_items = list_split(TRAIN_DIR)
    train_amb = [(p, c) for p, c in train_items if c == "ambulance"]
    print(f"  Total train ambulance: {len(train_amb)}")

    train_stats = []
    for p, c in train_amb:
        mean_rgb, blue_ratio = rgb_stats(p)
        train_stats.append({
            "path": p,
            "blue_ratio": blue_ratio,
            "mean_B": float(mean_rgb[2]),
        })

    # Sort by blue_ratio (highest first)
    train_stats.sort(key=lambda x: -x["blue_ratio"])
    top_blue_train = [s["path"] for s in train_stats[:12]]
    save_gallery(top_blue_train,
                 os.path.join(OUT_DIR, "train_most_blue.png"),
                 title="Train / ambulance — most BLUE images (top-12)")

    print(f"  Train blue_ratio stats: "
          f"min={min(s['blue_ratio'] for s in train_stats):.3f}, "
          f"mean={np.mean([s['blue_ratio'] for s in train_stats]):.3f}, "
          f"max={max(s['blue_ratio'] for s in train_stats):.3f}")

    # --- 2) Compute pseudo-labels on unclean ---
    print("\n[2/4] Loading fixmatch_v2 and computing pseudo-labels ...")
    model, _ = load_checkpoint("fixmatch_v2", "resnet")

    unclean_items = list_split(UNCLEAN_DIR)
    # Keep only known classes for dataset
    unclean_known = [(p, c) for p, c in unclean_items if c in class_to_idx]
    print(f"  Total unclean (known classes): {len(unclean_known)}")

    pseudo_results = compute_pseudo_labels(
        model, unclean_known, class_to_idx, DEVICE, threshold=0.95
    )

    # --- 3) Analyze ---
    print("\n[3/4] Analyzing pseudo-labels ...")
    above = [r for r in pseudo_results if r["above_threshold"]]
    pseudo_as_ambulance = [r for r in above if r["pseudo_label"] == "ambulance"]
    print(f"  Above threshold (>=0.95): {len(above)}/{len(pseudo_results)} "
          f"({100*len(above)/len(pseudo_results):.1f}%)")
    print(f"  Pseudo-labeled as 'ambulance': {len(pseudo_as_ambulance)}")

    # Count: which original labels got pseudo-labeled as ambulance?
    orig_to_amb = Counter(r["original_label"] for r in pseudo_as_ambulance)
    print(f"  Originals that became 'ambulance':")
    for orig, cnt in orig_to_amb.most_common():
        print(f"    {orig}: {cnt}")

    # Save gallery
    if pseudo_as_ambulance:
        paths = [r["path"] for r in pseudo_as_ambulance]
        # Sort by confidence descending
        sorted_p = sorted(pseudo_as_ambulance, key=lambda x: -x["confidence"])
        paths_sorted = [r["path"] for r in sorted_p]
        save_gallery(paths_sorted,
                     os.path.join(OUT_DIR, "unclean_pseudo_ambulance.png"),
                     title="Unclean samples with pseudo-label='ambulance'")

    # --- 4) Report ---
    print("\n[4/4] Saving report ...")
    report = {
        "train_ambulance": {
            "n": len(train_amb),
            "blue_ratio_mean": float(np.mean([s["blue_ratio"] for s in train_stats])),
            "blue_ratio_max": float(max(s["blue_ratio"] for s in train_stats)),
        },
        "unclean": {
            "n_total": len(pseudo_results),
            "n_above_threshold": len(above),
            "n_pseudo_ambulance": len(pseudo_as_ambulance),
            "originals_became_ambulance": dict(orig_to_amb),
        },
        "top_5_pseudo_ambulance": [
            {"path": r["path"], "orig": r["original_label"],
             "conf": r["confidence"]}
            for r in sorted(pseudo_as_ambulance,
                            key=lambda x: -x["confidence"])[:5]
        ],
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    if len(pseudo_as_ambulance) > 20:
        print("  ⚠ HYPOTHESIS 1 CONFIRMED: FixMatch pseudo-labels assigned")
        print(f"    {len(pseudo_as_ambulance)} unclean samples to 'ambulance'.")
        print("    Inspect unclean_pseudo_ambulance.png to see if they are blue vans.")
    else:
        print("  ⚠ Pseudo-labels are not the main cause.")
        print("    Investigate class weighting / class balance.")


if __name__ == "__main__":
    main()