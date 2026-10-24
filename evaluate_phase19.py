"""evaluate_phase19.py — Evaluate Phase19Net (Calibrated Quantum Model).

Focus: See if the learned Temperature (0.66) fixed the overconfidence on test3 blue vans.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, TESTING_DATA_ROOTS
from data import IndexedImageDataset, eval_transform
from audit import list_split
from models_phase19 import Phase19Net # Import Phase 19

REJECT_THRESHOLD = 0.25 # We kept the 0.25 threshold from the API server to force reviews

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
    all_exp_weights = []

    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        out = model(x, return_all=True)
        
        # The Phase 19 head outputs calibrated probabilities directly
        probs = out["memberships"]
        
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
                "expert_attn": round(out["expert_weights"][j, 0].item(), 4),
                "expert_topo": round(out["expert_weights"][j, 1].item(), 4)
            })

        all_exp_weights.append(out["expert_weights"].cpu())

    exp_w = torch.cat(all_exp_weights).numpy()

    return {
        "detailed_results": all_results,
        "expert_weights_mean": exp_w.mean(axis=0).tolist(),
        "expert_weights_std": exp_w.std(axis=0).tolist(),
    }

def main():
    print("=" * 75)
    print("EVALUATING PHASE 19 (CALIBRATED QUANTUM MODEL)")
    print("=" * 75)

    ckpt_path = os.path.join(ARTIFACTS, "phase19.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. Run train_phase19.py first.")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = Phase19Net(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    temp = ckpt["config"].get("temperature", 1.0)
    print(f"Loaded phase19.pt (Learned Temperature: {temp:.4f})")

    c2i = ckpt["class_to_idx"]
    out_dir = os.path.join(ARTIFACTS, "phase19_eval")
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

        result = predict_folder_with_reject(model, test3_root, c2i, DEVICE)
        if result is None:
            continue

        if root_idx == 0:
            json_path = os.path.join(out_dir, "test3_detailed_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"  ✓ Detailed results saved to: {json_path}")

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

    print("\n  Overall Expert Weights Mean: Attn={:.3f}, Topo={:.3f}".format(
        result['expert_weights_mean'][0], result['expert_weights_mean'][1]))
    print(f"  NEEDS_REVIEW (Margin < {REJECT_THRESHOLD}): {review_count} / {total_amb_images}")

if __name__ == "__main__":
    main()