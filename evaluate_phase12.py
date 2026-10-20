"""evaluate_phase12.py — Evaluate Phase12Net (MIMO + Topology) on test3 anomaly.

Success criteria:
    1. Val accuracy remains high (~0.92)
    2. The remaining 9 stubborn blue vans are reduced.
    3. We can inspect how the MIMO combiner weights the Attention vs Topology experts.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS,
    TESTING_DATA_ROOTS,
)
from data import IndexedImageDataset, eval_transform
from audit import list_split
from models_phase12 import Phase12Net


@torch.no_grad()
def predict_folder(model, folder, c2i, device):
    """Run inference on all images in a class-organized folder."""
    items = list_split(folder)
    known_items = [(p, c) for p, c in items if c in c2i]
    
    if not known_items:
        return None

    ds = IndexedImageDataset(known_items, c2i, eval_transform())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    model.eval()
    all_preds, all_labels, all_confs = [], [], []
    all_attn, all_exp_weights = [], []

    for x, y in loader:
        x = x.to(device)
        out = model(x, return_all=True)
        
        preds = out["memberships"].argmax(dim=1)
        confs = out["memberships"].max(dim=1).values
        
        all_preds.append(preds.cpu())
        all_labels.append(y)
        all_confs.append(confs.cpu())
        all_attn.append(out["attention"].cpu())
        all_exp_weights.append(out["expert_weights"].cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    confs = torch.cat(all_confs).numpy()
    attn = torch.cat(all_attn).numpy()
    exp_w = torch.cat(all_exp_weights).numpy() # (N, 2) -> [Attn Weight, Topo Weight]

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "confidences": confs,
        "attention_mean": attn.mean(axis=0).tolist(),
        "expert_weights_mean": exp_w.mean(axis=0).tolist(),
        "expert_weights_std": exp_w.std(axis=0).tolist(),
    }


def main():
    print("=" * 70)
    print("EVALUATING PHASE 12 (MIMO + TOPOLOGY)")
    print("=" * 70)

    ckpt_path = os.path.join(ARTIFACTS, "phase12.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. Run train_phase12.py first.")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = Phase12Net(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Loaded phase12.pt (val_acc={ckpt['config']['val_acc']:.4f})")

    c2i = ckpt["class_to_idx"]

    # --- Test3 specifically ---
    print("\n[TEST3] The original anomaly folder:")
    
    # We will store metrics for the final comparison table
    total_amb_imgs = 0
    pred_amb_count = 0
    pred_van_count = 0
    
    for root in TESTING_DATA_ROOTS:
        test3_root = os.path.join(root, "test3")
        if not os.path.isdir(test3_root):
            continue

        result = predict_folder(model, test3_root, c2i, DEVICE)
        if result is None:
            continue

        # Filter results to see only 'ambulance' images (the blue vans)
        ambulance_indices = [i for i, label in enumerate(result["y_true"]) 
                             if CLASSES[label] == "ambulance"]
        
        if not ambulance_indices:
            continue

        y_pred_amb = result["y_pred"][ambulance_indices]
        n = len(y_pred_amb)
        total_amb_imgs = n

        # Count predictions
        pred_counts = {}
        for p in y_pred_amb:
            pred_counts[CLASSES[p]] = pred_counts.get(CLASSES[p], 0) + 1

        pred_amb_count = pred_counts.get("ambulance", 0)
        pred_van_count = pred_counts.get("vanet", 0)

        print(f"  Folder: {test3_root}")
        print(f"  Predictions on {n} 'ambulance' (blue van) images:")
        for cls, cnt in sorted(pred_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:10s} {cnt:3d} ({100*cnt/n:.1f}%)")
            
        print(f"  Attention (Top/Mid/Bot/Full): "
              f"{result['attention_mean'][0]:.3f} / "
              f"{result['attention_mean'][1]:.3f} / "
              f"{result['attention_mean'][2]:.3f} / "
              f"{result['attention_mean'][3]:.3f}")
              
        print(f"  Expert Weights (Attention vs Topology): "
              f"{result['expert_weights_mean'][0]:.3f} vs {result['expert_weights_mean'][1]:.3f}")

        # Save JSON
        os.makedirs(os.path.join(ARTIFACTS, "phase12"), exist_ok=True)
        with open(os.path.join(ARTIFACTS, "phase12", "test3_eval.json"), "w") as f:
            json.dump({
                "pred_counts": pred_counts,
                "attention_mean": result["attention_mean"],
                "expert_weights_mean": result["expert_weights_mean"],
            }, f, indent=2)

    # --- Comparison Table ---
    print("\n" + "=" * 70)
    print("PROGRESS COMPARISON (test3 blue vans)")
    print("=" * 70)
    
    print(f"  {'Metric':<30s} {'Baseline':>12s} {'Phase 11':>12s} {'Phase 12':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Val accuracy':<30s} {'0.9091':>12s} {'0.9225':>12s} {ckpt['config']['val_acc']:>12.4f}")
    
    # Baseline and Phase 11 numbers are known from previous runs
    print(f"  {'test3 → ambulance (errors)':<30s} {'24 / 25':>12s} {'9 / 25':>12s} {f'{pred_amb_count} / {total_amb_imgs}':>12s}")
    print(f"  {'test3 → vanet (correct)':<30s} {'1 / 25':>12s} {'16 / 25':>12s} {f'{pred_van_count} / {total_amb_imgs}':>12s}")


if __name__ == "__main__":
    main()