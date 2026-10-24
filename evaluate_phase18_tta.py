"""evaluate_phase18_tta.py — Evaluate Phase 17 Model with Test-Time Augmentation (TTA).

Focus: Improve inference accuracy without retraining.
TTA Technique: We pass the original image AND a horizontally flipped version 
to the model, then average their probabilities. This reduces variance and 
often fixes edge cases (like the blue vans).
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, TESTING_DATA_ROOTS
from data import IndexedImageDataset, eval_transform
from audit import list_split
from models_phase16 import Phase16Net

REJECT_THRESHOLD = 0.15

@torch.no_grad()
def tta_predict_batch(model, x):
    """Applies TTA on a batch of images."""
    # 1. Original image prediction
    out_orig = model(x, return_all=True)
    probs_orig = out_orig["memberships"]
    
    # 2. Horizontally Flipped image prediction
    # dims=[3] flips the width dimension (N, C, H, W)
    x_flipped = torch.flip(x, dims=[3])
    out_flip = model(x_flipped, return_all=True)
    probs_flip = out_flip["memberships"]
    
    # 3. Average the probabilities
    probs_avg = (probs_orig + probs_flip) / 2.0
    
    # We also average the expert weights for consistency
    exp_w_avg = (out_orig["expert_weights"] + out_flip["expert_weights"]) / 2.0
    
    return probs_avg, exp_w_avg

@torch.no_grad()
def predict_folder_with_tta(model, folder, c2i, device):
    items = list_split(folder)
    known_items = [(p, c) for p, c in items if c in c2i]
    
    if not known_items:
        return None

    ds = IndexedImageDataset(known_items, c2i, eval_transform())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    model.eval()
    all_results = []
    all_exp_weights = []

    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        
        # Use TTA instead of direct prediction
        probs, exp_weights = tta_predict_batch(model, x)
        
        top2_probs, top2_idx = torch.topk(probs, 2, dim=1)
        
        for j in range(x.size(0)):
            idx_in_known = i * loader.batch_size + j
            if idx_in_known >= len(known_items):
                break
                
            img_path = known_items[idx_in_known][0]
            true_cls = CLASSES[y[j].item()]
            pred_cls = CLASSES[top2_idx[j, 0].item()]
            
            top1_prob = top2_probs[j, 0].item()
            top2_prob = top2_probs[j, 1].item()
            margin = top1_prob - top2_prob
            
            status = "CONFIDENT" if margin >= REJECT_THRESHOLD else "NEEDS_REVIEW"
                
            all_results.append({
                "path": os.path.basename(img_path),
                "true_label": true_cls,
                "pred_label": pred_cls,
                "confidence": round(top1_prob, 4),
                "margin": round(margin, 4),
                "status": status,
                "expert_attn": round(exp_weights[j, 0].item(), 4),
                "expert_topo": round(exp_weights[j, 1].item(), 4)
            })

        all_exp_weights.append(exp_weights.cpu())

    exp_w = torch.cat(all_exp_weights).numpy()

    return {
        "detailed_results": all_results,
        "expert_weights_mean": exp_w.mean(axis=0).tolist(),
        "expert_weights_std": exp_w.std(axis=0).tolist(),
    }

def main():
    print("=" * 75)
    print("EVALUATING PHASE 18 (TEST-TIME AUGMENTATION ON PHASE 17 MODEL)")
    print("=" * 75)

    ckpt_path = os.path.join(ARTIFACTS, "phase17.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = Phase16Net(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded phase17.pt with TTA enabled.")

    c2i = ckpt["class_to_idx"]
    out_dir = os.path.join(ARTIFACTS, "phase18_eval")
    os.makedirs(out_dir, exist_ok=True)

    print("\n[TEST3] The original anomaly folder (Blue Vans):")
    
    total_amb_images = 0
    pred_amb_count = 0
    pred_van_count = 0
    review_count = 0
    
    for root_idx, root in enumerate(TESTING_DATA_ROOTS):
        test3_root = os.path.join(root, "test3")
        if not os.path.isdir(test3_root):
            continue

        result = predict_folder_with_tta(model, test3_root, c2i, DEVICE)
        if result is None:
            continue

        if root_idx == 0:
            json_path = os.path.join(out_dir, "test3_tta_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"  ✓ Detailed TTA results saved to: {json_path}")

        ambulance_results = [r for r in result["detailed_results"] if r["true_label"] == "ambulance"]
        
        if not ambulance_results:
            continue

        total_amb_images += len(ambulance_results)
        
        pred_counts = {}
        for r in ambulance_results:
            pred_counts[r["pred_label"]] = pred_counts.get(r["pred_label"], 0) + 1
            if r["status"] == "NEEDS_REVIEW":
                review_count += 1

        pred_amb_count += pred_counts.get("ambulance", 0)
        pred_van_count += pred_counts.get("vanet", 0)

        print(f"  Folder {root_idx+1}: Predictions on {len(ambulance_results)} 'ambulance' (blue van) images:")
        for cls, cnt in sorted(pred_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:10s} {cnt:3d} ({100*cnt/len(ambulance_results):.1f}%)")

    print("\n  Overall Expert Weights Mean (TTA): Attn={:.3f}, Topo={:.3f}".format(
        result['expert_weights_mean'][0], result['expert_weights_mean'][1]))
    print(f"  NEEDS_REVIEW (Margin < {REJECT_THRESHOLD}): {review_count} / {total_amb_images}")

    print("\n" + "=" * 95)
    print("TTA IMPACT COMPARISON (test3 blue vans)")
    print("=" * 95)
    
    print(f"  {'Metric':<28s} {'Phase 14':>10s} {'Phase 16':>10s} {'Phase 17':>10s} {'Phase 18 (TTA)':>15s}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*15}")
    print(f"  {'test3 → ambulance (errors)':<28s} {'0 / 25':>10s} {'18 / 50':>10s} {'18 / 50':>10s} {f'{pred_amb_count} / {total_amb_images}':>15s}")
    print(f"  {'test3 → vanet (correct)':<28s} {'25 / 25':>10s} {'32 / 50':>10s} {'32 / 50':>10s} {f'{pred_van_count} / {total_amb_images}':>15s}")
    print(f"  {'test3 → NEEDS_REVIEW':<28s} {'0 / 25':>10s} {'0 / 50':>10s} {'0 / 50':>10s} {f'{review_count} / {total_amb_images}':>15s}")

if __name__ == "__main__":
    main()