"""evaluate_antishortcut.py — Test the AntiShortcutNet on test3 anomaly.

Success criteria:
    1. Val accuracy remains high (~0.90+)
    2. Attention weight on TOP region > attention on BOTTOM
    3. test3 blue vans → mostly 'vanet' (not 'ambulance')
"""
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

from config import (
    CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS,
    TESTING_DATA_ROOTS,
)
from data import IndexedImageDataset, eval_transform
from audit import list_split
from models_attention import AntiShortcutNet


@torch.no_grad()
def predict_folder(model, folder, c2i, device):
    """Run inference on all images in a class-organized folder."""
    items = list_split(folder)
    known_items = [(p, c) for p, c in items if c in c2i]
    print(f"Debug: known_items has {len(known_items)} items after filtering.")
    print(f"Debug: c2i keys are: {list(c2i.keys())}")
    if items:
        print(f"Debug: Example label from items: {items[0][1]}")

    ds = IndexedImageDataset(known_items, c2i, eval_transform())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    model.eval()
    all_preds, all_labels, all_confs = [], [], []
    all_attn = []

    for x, y in loader:
        x = x.to(device)
        out = model(x, return_attention=True)
        
        # preds, confs = out["memberships"].max(dim=1)
        preds = out["memberships"].argmax(dim=1)          # Get the class index (integer)
        confs = out["memberships"].max(dim=1).values      # Get the membership value (confidence)
        
        all_preds.append(preds.cpu())
        all_labels.append(y)
        all_confs.append(confs.cpu())
        all_attn.append(out["attention"].cpu())

    if not all_preds:
        print("Warning: No images were loaded. Skipping this folder.")
        return {
            "y_true": [], "y_pred": [], "confidences": [],
            "attention_mean": [0,0,0,0], "attention_std": [0,0,0,0]
        }

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    confs = torch.cat(all_confs).numpy()
    attn = torch.cat(all_attn).numpy()

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "confidences": confs,
        "attention_mean": attn.mean(axis=0).tolist(),
        "attention_std": attn.std(axis=0).tolist(),
    }


def main():
    print("=" * 70)
    print("EVALUATING ANTI-SHORTCUT MODEL")
    print("=" * 70)

    ckpt = torch.load(os.path.join(ARTIFACTS, "antishortcut.pt"),
                      map_location=DEVICE, weights_only=False)
    model = AntiShortcutNet(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Loaded antishortcut.pt (val_acc={ckpt['config']['val_acc']:.4f})")

    c2i = ckpt["class_to_idx"]
    regions = ["TOP", "MIDDLE", "BOTTOM", "FULL"]

# --- Test3 specifically ---
    print("\n[TEST3] The original anomaly folder:")
    for root in TESTING_DATA_ROOTS:
        test3_root = os.path.join(root, "test3")  # مسیر والد
        if not os.path.isdir(test3_root):
            continue

        print(f"  Folder: {test3_root}")
        # We read the entire 'test3' folder (including ambulance, vanet, etc.)        
        result = predict_folder(model, test3_root, c2i, DEVICE)
        
        # Now we filter the results to see only ambulance images 
        # (because list_split returns the original label)
        ambulance_indices = [i for i, label in enumerate(result["y_true"]) 
                             if CLASSES[label] == "ambulance"]
        
        if not ambulance_indices:
            print("  No 'ambulance' images found in this folder.")
            continue

        y_pred_amb = result["y_pred"][ambulance_indices]
        attn_amb = result["attention_mean"] # Note: Overall folder average
        n = len(y_pred_amb)

        # Count predictions
        pred_counts = {}
        for p in y_pred_amb:
            pred_counts[CLASSES[p]] = pred_counts.get(CLASSES[p], 0) + 1

        print(f"  Predictions on {n} 'ambulance' images:")
        for cls, cnt in sorted(pred_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:10s} {cnt:3d} ({100*cnt/n:.1f}%)")
            
        print(f"  Attention: TOP={result['attention_mean'][0]:.3f}  "
              f"MID={result['attention_mean'][1]:.3f}  "
              f"BOT={result['attention_mean'][2]:.3f}  "
              f"FULL={result['attention_mean'][3]:.3f}")

        with open(os.path.join(ARTIFACTS, "antishortcut",
                                "test3_predictions.json"), "w") as f:
            json.dump({
                "pred_counts": pred_counts,
                "attention_mean": result["attention_mean"],
            }, f, indent=2)

    # --- Comparison: baseline vs anti-shortcut ---
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    # Getting the number of predictions from the last loop iteration (for Test 3)    
    antishortcut_amb = pred_counts.get("ambulance", 0)
    antishortcut_van = pred_counts.get("vanet", 0)
    total_imgs = sum(pred_counts.values()) or 25

    print(f"  {'Metric':<30s} {'Baseline':>12s} {'AntiShortcut':>15s}")
    print(f"  {'-'*30} {'-'*12} {'-'*15}")
    print(f"  {'Val accuracy':<30s} {'0.9091':>12s} "
          f"{ckpt['config']['val_acc']:>15.4f}")
    print(f"  {'test3 → ambulance':<30s} {'~24 / 25':>12s} "
          f"{f'{antishortcut_amb} / {total_imgs}':>15s}")
    print(f"  {'test3 → vanet':<30s} {'~1 / 25':>12s} "
          f"{f'{antishortcut_van} / {total_imgs}':>15s}")


if __name__ == "__main__":
    main()