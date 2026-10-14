"""investigate_ambulance.py — Diagnose the ambulance anomaly.

After FixMatch v2, ambulance recall jumped from 0.36 to 0.99 in test3.
This script tests two hypotheses:
    H1 (good): training set contains REAL ambulances → model learned correctly.
    H2 (bad):  training set contains BLUE NISSAN VANS labeled as ambulance
               → model learned wrong labels.

We compare color statistics of:
    - merged train / ambulance/*.jpg  (labeled data)
    - test3 / ambulance/*.jpg         (anomalous test images)

Output:
    artifacts/ambulance_investigation/
        ├── train_ambulance_samples.png
        ├── test3_ambulance_samples.png
        ├── side_by_side.png
        ├── color_stats.json
        └── report.md
"""
import os
import json
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from config import TRAIN_DIR, ARTIFACTS, TESTING_DATA_ROOTS
from audit import list_split


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(ARTIFACTS, "ambulance_investigation")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1) Locate test3 (the anomalous folder)
# ---------------------------------------------------------------------------
def find_test3_ambulance_folder():
    """Return the path to test3/ambulance/ across all testing roots."""
    for root in TESTING_DATA_ROOTS:
        candidate = os.path.join(root, "test3", "ambulance")
        if os.path.isdir(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# 2) Color analysis
# ---------------------------------------------------------------------------
def rgb_stats(img_path):
    """Return mean R, G, B and 'blue dominance' ratio for an image."""
    with Image.open(img_path) as im:
        im = im.convert("RGB").resize((128, 128))
        arr = np.asarray(im, dtype=np.float32)  # (128, 128, 3)

    # Center crop (vehicle is usually centered)
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2
    ch, cw = h // 4, w // 4
    center = arr[cy - ch:cy + ch, cx - cw:cx + cw]

    mean_rgb = center.reshape(-1, 3).mean(axis=0)

    # Blue dominance: how many pixels have B > R + 20 and B > G + 20
    r, g, b = center[..., 0], center[..., 1], center[..., 2]
    blue_mask = (b > r + 20) & (b > g + 20)
    blue_ratio = float(blue_mask.mean())

    return {
        "mean_R": float(mean_rgb[0]),
        "mean_G": float(mean_rgb[1]),
        "mean_B": float(mean_rgb[2]),
        "blue_ratio": blue_ratio,
    }


# ---------------------------------------------------------------------------
# 3) Gallery helpers
# ---------------------------------------------------------------------------
def save_gallery(paths, save_path, title_prefix="", n=12):
    """Save an n-image gallery (3x4 grid)."""
    paths = paths[:n]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    for idx, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB").resize((224, 224))
            axes[idx].imshow(img)
            axes[idx].set_title(os.path.basename(p)[:24], fontsize=7)
            axes[idx].axis("off")
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error:\n{e}", ha="center")
            axes[idx].axis("off")

    for i in range(len(paths), len(axes)):
        axes[i].axis("off")

    fig.suptitle(title_prefix, fontsize=12)
    fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  Saved gallery → {save_path}")


# ---------------------------------------------------------------------------
# 4) Main investigation
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("AMBULANCE INVESTIGATION")
    print("=" * 70)

    # --- Train ambulance images ---
    print("\n[1/4] Scanning merged train / ambulance ...")
    all_train = list_split(TRAIN_DIR)
    train_amb = [p for p, c in all_train if c == "ambulance"]
    print(f"  Total ambulance in merged train: {len(train_amb)}")

    if not train_amb:
        print("  ✗ No ambulance images found in train. Aborting.")
        return

    # --- Test3 ambulance images ---
    print("\n[2/4] Locating test3/ambulance ...")
    test3_dir = find_test3_ambulance_folder()
    if test3_dir is None:
        print("  ✗ test3/ambulance not found in any TESTING_DATA_ROOT.")
        test3_amb = []
    else:
        test3_amb = [os.path.join(test3_dir, f)
                     for f in sorted(os.listdir(test3_dir))
                     if os.path.isfile(os.path.join(test3_dir, f))]
        print(f"  test3/ambulance: {test3_dir}")
        print(f"  Total test3 ambulance images: {len(test3_amb)}")

    # --- Random samples ---
    print("\n[3/4] Building galleries ...")
    random.seed(42)
    train_samples = random.sample(train_amb, min(12, len(train_amb)))
    save_gallery(train_samples,
                 os.path.join(OUT_DIR, "train_ambulance_samples.png"),
                 title_prefix="TRAIN / ambulance (merged V1+V2)")

    if test3_amb:
        test3_samples = test3_amb[:12]
        save_gallery(test3_samples,
                     os.path.join(OUT_DIR, "test3_ambulance_samples.png"),
                     title_prefix="TEST3 / ambulance (suspected blue vans)")

    # --- Color statistics ---
    print("\n[4/4] Computing color statistics ...")
    train_stats = [rgb_stats(p) for p in train_amb]
    test3_stats = [rgb_stats(p) for p in test3_amb] if test3_amb else []

    def aggregate(stats):
        if not stats:
            return {}
        arr = np.array([[s["mean_R"], s["mean_G"], s["mean_B"]]
                        for s in stats])
        return {
            "n": len(stats),
            "mean_R": float(arr[:, 0].mean()),
            "mean_G": float(arr[:, 1].mean()),
            "mean_B": float(arr[:, 2].mean()),
            "mean_blue_ratio": float(np.mean([s["blue_ratio"] for s in stats])),
            "std_blue_ratio": float(np.std([s["blue_ratio"] for s in stats])),
        }

    agg_train = aggregate(train_stats)
    agg_test3 = aggregate(test3_stats)

    # --- Report ---
    report = {
        "train": agg_train,
        "test3": agg_test3,
        "hypotheses": {
            "H1_train_is_real_ambulance": None,   # fill below
            "H2_train_is_blue_vans": None,
        },
    }

    # Decision rule: if test3 blue_ratio > 0.15 and train blue_ratio < 0.10
    # → they are visually different → H1 supported (model learned correctly)
    # If both blue_ratio > 0.15 → similar → H2 supported (label leak)
    if agg_train and agg_test3:
        train_blue = agg_train["mean_blue_ratio"]
        test3_blue = agg_test3["mean_blue_ratio"]
        if test3_blue > 0.15 and train_blue < 0.10:
            verdict = "H1 (train contains real ambulances — model learned correctly)"
            report["hypotheses"]["H1_train_is_real_ambulance"] = True
            report["hypotheses"]["H2_train_is_blue_vans"] = False
        elif train_blue > 0.15 and test3_blue > 0.15:
            verdict = "H2 (train contains blue vans — model learned WRONG labels!)"
            report["hypotheses"]["H1_train_is_real_ambulance"] = False
            report["hypotheses"]["H2_train_is_blue_vans"] = True
        else:
            verdict = "Inconclusive — inspect galleries manually"
        report["verdict"] = verdict
    else:
        report["verdict"] = "Missing data — could not compare"

    with open(os.path.join(OUT_DIR, "color_stats.json"), "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("# Ambulance Investigation Report\n\n")
        f.write("## Statistics\n\n")
        f.write(f"### Merged train / ambulance (n={agg_train.get('n', 0)})\n")
        f.write(f"- Mean RGB: ({agg_train.get('mean_R', 0):.1f}, "
                f"{agg_train.get('mean_G', 0):.1f}, "
                f"{agg_train.get('mean_B', 0):.1f})\n")
        f.write(f"- Blue ratio: {agg_train.get('mean_blue_ratio', 0):.3f}\n\n")
        f.write(f"### test3 / ambulance (n={agg_test3.get('n', 0)})\n")
        f.write(f"- Mean RGB: ({agg_test3.get('mean_R', 0):.1f}, "
                f"{agg_test3.get('mean_G', 0):.1f}, "
                f"{agg_test3.get('mean_B', 0):.1f})\n")
        f.write(f"- Blue ratio: {agg_test3.get('mean_blue_ratio', 0):.3f}\n\n")
        f.write(f"## Verdict\n\n{report['verdict']}\n")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Train ambulance:  n={agg_train.get('n', 0)}, "
          f"blue_ratio={agg_train.get('mean_blue_ratio', 0):.3f}")
    print(f"  Test3 ambulance:  n={agg_test3.get('n', 0)}, "
          f"blue_ratio={agg_test3.get('mean_blue_ratio', 0):.3f}")
    print(f"\n  Verdict: {report['verdict']}")
    print(f"\n  Output dir: {OUT_DIR}")
    print(f"  ⚠ Inspect the galleries manually to confirm!")


if __name__ == "__main__":
    main()