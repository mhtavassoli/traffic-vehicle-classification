"""evaluate_phase13.py — Evaluate Phase13Net (Fixed Softmax & Reject Option)"""
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
from models_phase13 import Phase13Net

# Reject threshold (Margin Threshold). Since we use Softmax, 0.25 is reasonable.
REJECT_THRESHOLD = 0.25

@torch.no_grad()
def predict_folder_with_reject(model, folder, c2i, device):
    items = list_split(folder)
    known_items = [(p, c) for p, c in items if c in c2i]
    
    if not known_items:
        return None

    ds = IndexedImageDataset(known_items, c2i, eval_transform())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    model.eval()
    
    all_results = []
    all_attn = []
    all_exp_weights = []

    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        out = model(x, return_all=True)
        
        # Bug fix: convert fuzzy memberships to actual probabilities using Softmax
        memberships = out["memberships"]
        probs = torch.softmax(memberships, dim=1)
        
        # Find the top-2 classes
        top2_probs, top2_idx = torch.topk(probs, 2, dim=1)
        
        for j in range(x.size(0)):
            idx_in_known = i * loader.batch_size + j
            img_path = known_items[idx_in_known][0]
            
            true_cls = CLASSES[y[j].item()]
            pred_cls = CLASSES[top2_idx[j, 0].item()]
            
            top1_prob = top2_probs[j, 0].item()
            top2_prob = top2_probs[j, 1].item()
            margin = top1_prob - top2_prob
            
            # Reject Option system
            status = "CONFIDENT"
            if margin < REJECT_THRESHOLD:
                status = "NEEDS_REVIEW"
                
            all_results.append({
                "path": os.path.basename(img_path),
                "true_label": true_cls,
                "pred_label": pred_cls,
                "confidence": round(top1_prob, 4),
                "margin": round(margin, 4),
                "status": status,
                "expert_attn": round(out["expert_weights"][j, 0].item(), 4),
                "expert_topo": round(out["expert_weights"][j, 1].item(), 4)
            })

        all_attn.append(out["attention"].cpu())
        all_exp_weights.append(out["expert_weights"].cpu())

    attn = torch.cat(all_attn).numpy()
    exp_w = torch.cat(all_exp_weights).numpy()

    return {
        "detailed_results": all_results,
        "attention_mean": attn.mean(axis=0).tolist(),
        "expert_weights_mean": exp_w.mean(axis=0).tolist(),
        "expert_weights_std": exp_w.std(axis=0).tolist(),
    }

def main():
    print("=" * 75)
    print("EVALUATING PHASE 13 (MIMO + DYNAMIC ROUTING + REJECT OPTION)")
    print("=" * 75)

    ckpt_path = os.path.join(ARTIFACTS, "phase13.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = Phase13Net(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Loaded phase13.pt (val_acc={ckpt['config']['val_acc']:.4f})")

    c2i = ckpt["class_to_idx"]
    out_dir = os.path.join(ARTIFACTS, "phase13")
    os.makedirs(out_dir, exist_ok=True)

    print("\n[TEST3] The original anomaly folder (Blue Vans):")
    
    # Overall stats for the final table
    total_amb_imgs = 0
    pred_amb_count = 0
    pred_van_count = 0
    review_count = 0
    
    for root_idx, root in enumerate(TESTING_DATA_ROOTS):
        test3_root = os.path.join(root, "test3")
        if not os.path.isdir(test3_root):
            continue

        result = predict_folder_with_reject(model, test3_root, c2i, DEVICE)
        if result is None:
            continue

        # Save the full results to a JSON file (only for the first folder, to avoid overwriting the original file)
        if root_idx == 0:
            json_path = os.path.join(out_dir, "test3_detailed_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"  ✓ Detailed results saved to: {json_path}")

        # Filter ambulance images
        ambulance_results = [r for r in result["detailed_results"] if r["true_label"] == "ambulance"]
        
        if not ambulance_results:
            continue

        total_amb_imgs += len(ambulance_results)
        
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
            
    # Print overall statistics
    print("\n  Overall Expert Weights Mean: Attn={:.3f}, Topo={:.3f}".format(
        result['expert_weights_mean'][0], result['expert_weights_mean'][1]))
    print("  Overall Expert Weights STD : Attn={:.3f}, Topo={:.3f} (Dynamic Routing)".format(
        result['expert_weights_std'][0], result['expert_weights_std'][1]))
    print(f"  NEEDS_REVIEW (Margin < {REJECT_THRESHOLD}): {review_count} / {total_amb_imgs}")

    # --- Comparison Table ---
    print("\n" + "=" * 75)
    print("PROGRESS COMPARISON (test3 blue vans)")
    print("=" * 75)
    
    print(f"  {'Metric':<32s} {'Baseline':>10s} {'Phase 11':>10s} {'Phase 12':>10s} {'Phase 13':>10s}")
    print(f"  {'-'*32} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Val accuracy':<32s} {'0.9091':>10s} {'0.9225':>10s} {'0.9300':>10s} {ckpt['config']['val_acc']:>10.4f}")
    print(f"  {'test3 → ambulance (errors)':<32s} {'24 / 25':>10s} {'9 / 25':>10s} {'9 / 25':>10s} {f'{pred_amb_count} / {total_amb_imgs}':>10s}")
    print(f"  {'test3 → vanet (correct)':<32s} {'1 / 25':>10s} {'16 / 25':>10s} {'16 / 25':>10s} {f'{pred_van_count} / {total_amb_imgs}':>10s}")
    print(f"  {'test3 → NEEDS_REVIEW':<32s} {'0 / 25':>10s} {'0 / 25':>10s} {'0 / 25':>10s} {f'{review_count} / {total_amb_imgs}':>10s}")

if __name__ == "__main__":
    main()