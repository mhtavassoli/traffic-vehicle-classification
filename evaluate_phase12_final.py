"""evaluate_phase12_final.py — Final evaluation of the balanced Phase 12 model.

Focus: Analyze the 9 remaining "stubborn blue vans" that Phase 11 failed on.
Does the Topology Expert now succeed where Attention failed?

Features:
    1. Run inference on test3 blue vans.
    2. Track Expert Weights (Attention vs Topology) per image.
    3. Visualize what the Topology Expert "sees" via its edge features.
    4. Generate a comparison report (Baseline vs Phase 11 vs Phase 12-Final).
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

from config import CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, TESTING_DATA_ROOTS
from data import IndexedImageDataset, eval_transform
from audit import list_split
from models_phase12 import Phase12Net


OUT_DIR = os.path.join(ARTIFACTS, "phase12_final")
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def predict_folder_detailed(model, folder, c2i, device):
    """Run inference and record expert weights per image."""
    items = []
    folder_name = os.path.basename(folder)

    # Check whether the given folder is a class itself or the root directory
    if folder_name in c2i:
        # If the folder is a class, list the images inside it directly
        for f in os.listdir(folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                items.append((os.path.join(folder, f), folder_name))
    else:
        # If the folder is the root directory, use the main function
        items = list_split(folder)
    
    known_items = [(p, c) for p, c in items if c in c2i]
    ds = IndexedImageDataset(known_items, c2i, eval_transform())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model.eval()
    results = []

    for x, y in loader:
        x = x.to(device)
        out = model(x, return_all=True)
        memberships = out["memberships"]
        preds = memberships.argmax(dim=1)
        confs = memberships.max(dim=1).values
        attn = out["attention"]
        expert_w = out["expert_weights"] # (B, 2) -> [Attn, Topo]

        for i in range(x.size(0)):
            results.append({
                "path": known_items[len(results)][0],
                "true_label": CLASSES[y[i].item()],
                "pred_label": CLASSES[preds[i].item()],
                "confidence": float(confs[i].item()),
                "expert_attn": float(expert_w[i, 0].item()),
                "expert_topo": float(expert_w[i, 1].item()),
            })

    return results

@torch.no_grad()
def visualize_topology_expert(model, folder, c2i, device):
    """Extract features from the Topology Expert to see what edges it finds."""
    items = []
    folder_name = os.path.basename(folder)
    
    if folder_name in c2i:
        for f in os.listdir(folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                items.append((os.path.join(folder, f), folder_name))
    else:
        items = list_split(folder)

    known_items = [(p, c) for p, c in items if c in c2i][:3] # just 3 samples

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    model.eval()

    for idx, (path, _) in enumerate(known_items):
        img = Image.open(path).convert("RGB")
        x = eval_transform()(img).unsqueeze(0).to(device)

        fmap = model.features(x)
        edge_feats = F.relu(model.topology_expert.edge_conv(fmap))
        edge_map = edge_feats.mean(dim=1, keepdim=True)
        edge_map = F.interpolate(edge_map, size=(128, 128),
                                 mode="bilinear", align_corners=False)
        edge_map = edge_map[0, 0].cpu().numpy()

        axes[idx, 0].imshow(img.resize((128, 128)))
        axes[idx, 0].set_title(f"Original: {os.path.basename(path)[:20]}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(edge_map, cmap="hot")
        axes[idx, 1].set_title("Topology Expert (Sobel Edges)")
        axes[idx, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "topology_expert_edges.png"),
                bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved -> {OUT_DIR}/topology_expert_edges.png")


def main():
    print("=" * 70)
    print("EVALUATING PHASE 12 FINAL (BALANCED MIMO)")
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

    # --- Evaluate on test3 ---
    print("\n[TEST3] The 9 stubborn blue vans:")
    all_results = []

    for root in TESTING_DATA_ROOTS:
        test3_dir = os.path.join(root, "test3", "ambulance")
        if not os.path.isdir(test3_dir):
            continue

        print(f"  Folder: {test3_dir}")
        results = predict_folder_detailed(model, test3_dir, c2i, DEVICE)
        all_results.extend(results)

        # Count
        pred_counts = {}
        for r in results:
            pred_counts[r["pred_label"]] = pred_counts.get(r["pred_label"], 0) + 1

        n = len(results)
        print(f"  Predictions on {n} 'ambulance' images:")
        for cls, cnt in sorted(pred_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:10s} {cnt:3d} ({100*cnt/n:.1f}%)")

        # Average expert weights for these samples
        avg_attn = np.mean([r["expert_attn"] for r in results])
        avg_topo = np.mean([r["expert_topo"] for r in results])
        print(f"  Average Expert Weights for these {n} images:")
        print(f"    Attention: {avg_attn:.3f}")
        print(f"    Topology:  {avg_topo:.3f}")

    # --- Visualize Topology Expert ---
    print("\n[VISUALIZATION] Generating Topology Expert edge maps...")
    for root in TESTING_DATA_ROOTS:
        test3_dir = os.path.join(root, "test3", "ambulance")
        if os.path.isdir(test3_dir):
            visualize_topology_expert(model, test3_dir, c2i, DEVICE)
            break

    # --- Save Report ---
    with open(os.path.join(OUT_DIR, "test3_detailed_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Accurate calculation of statistics based on all obtained results
    total_images = len(all_results)
    correct_ambulance = sum(1 for r in all_results if r["pred_label"] == "ambulance")
    incorrect_ambulance = sum(1 for r in all_results if r["pred_label"] == "vanet")

    print("\n" + "=" * 70)
    print("PROGRESS COMPARISON (test3 blue vans)")
    print("=" * 70)
    print(f"  {'Metric':<30s} {'Baseline':>12s} {'Phase 11':>12s} {'Phase 12':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Val accuracy':<30s} {'0.9091':>12s} {'0.9225':>12s} "
          f"{ckpt['config']['val_acc']:>12.4f}")
    print(f"  {'test3 errors (ambulance)':<30s} {'24/25':>12s} {'9/25':>12s} "
          f"{pred_counts.get('ambulance', 0)}/25")
    print(f"  {'test3 correct (vanet)':<30s} {'1/25':>12s} {'16/25':>12s} "
          f"{pred_counts.get('vanet', 0)}/25")


if __name__ == "__main__":
    main()