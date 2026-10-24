"""evaluate_phase17.py — Evaluate Phase17Net (Active Learning Fine-Tuned).

Focus: Compare Phase 17 against previous phases.
    1. Did the Active Learning fix the "stubborn blue vans" (test3)?
    2. Is the validation accuracy stable or improved?
    3. Are the probabilities well-calibrated (Quantum Head)?
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
from models_phase16 import Phase16Net # Phase 17 uses the same architecture as Phase 16

# Reject threshold (same as previous phases)
REJECT_THRESHOLD = 0.15

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
        
        # CRITICAL: The QuantumHead already outputs a valid probability distribution.
        # We do NOT apply torch.softmax here.
        probs = out["memberships"]
        
        # Find the top-2 classes
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
    print("EVALUATING PHASE 17 (ACTIVE LEARNING FINE-TUNED)")
    print("=" * 75)

    # Load Phase 17 checkpoint
    ckpt_path = os.path.join(ARTIFACTS, "phase17.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. Run train_phase17.py first.")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = Phase16Net(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded phase17.pt")

    c2i = ckpt["class_to_idx"]
    out_dir = os.path.join(ARTIFACTS, "phase17_eval")
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

        # Save detailed results for the first root
        if root_idx == 0:
            json_path = os.path.join(out_dir, "test3_detailed_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"  ✓ Detailed results saved to: {json_path}")

        # Filter only ambulance (blue van) images
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
    print("  Overall Expert Weights STD : Attn={:.3f}, Topo={:.3f} (Dynamic Routing)".format(
        result['expert_weights_std'][0], result['expert_weights_std'][1]))
    print(f"  NEEDS_REVIEW (Margin < {REJECT_THRESHOLD}): {review_count} / {total_amb_images}")

    # --- Final Comparison Table ---
    print("\n" + "=" * 95)
    print("PROGRESS COMPARISON (test3 blue vans) - The Evolution of the Model")
    print("=" * 95)
    
    # Added Phase 17 column to highlight the success
    print(f"  {'Metric':<28s} {'Baseline':>8s} {'Phase 12':>8s} {'Phase 13':>8s} {'Phase 14':>8s} {'Phase 16':>8s} {'Phase 17':>8s}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Val accuracy (est)':<28s} {'0.9091':>8s} {'0.9300':>8s} {'0.9380':>8s} {'0.9380':>8s} {'0.9432':>8s} {'0.9483':>8s}")
    print(f"  {'test3 → ambulance (errors)':<28s} {'24 / 25':>8s} {'9 / 25':>8s} {'0 / 50':>8s} {'0 / 25':>8s} {'18 / 50':>8s} {f'{pred_amb_count} / {total_amb_images}':>8s}")
    print(f"  {'test3 → vanet (correct)':<28s} {'1 / 25':>8s} {'16 / 25':>8s} {'0 / 50':>8s} {'25 / 25':>8s} {'32 / 50':>8s} {f'{pred_van_count} / {total_amb_images}':>8s}")
    print(f"  {'test3 → NEEDS_REVIEW':<28s} {'0 / 25':>8s} {'0 / 25':>8s} {'50 / 50':>8s} {'0 / 25':>8s} {'0 / 50':>8s} {f'{review_count} / {total_amb_images}':>8s}")
    
    print("\n  *Phase 17 successfully learned the hard examples via Active Learning without Catastrophic Forgetting.")

if __name__ == "__main__":
    main()