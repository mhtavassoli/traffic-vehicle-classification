"""analysis_phase3.py — Phase 3: Fixed UQ, Cross-Model Gallery, OOD Analysis."""
import os, json, numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import (CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, BATCH_SIZE,
                    TRAIN_DIR, UNCLEAN_DIR, VAL_FRACTION, SEED)
from data import IndexedImageDataset, eval_transform, stratified_val_split
from audit import list_split, run_audit, build_cleaned_index
from analysis import load_checkpoint


# ============================================================
# 1) MC Dropout — Fixed (ResNet now has Dropout)
# ============================================================
def mc_dropout_predict(model, x, n_samples=50):
    """Proper MC Dropout: keep model.train() during inference."""
    model.train()  # Dropout ON, BatchNorm uses batch stats
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(torch.softmax(model(x), dim=1))
    preds = torch.stack(preds)
    mean = preds.mean(0)
    std = preds.std(0)
    model.eval()
    return mean, std


def mc_dropout_analysis(model, loader, n_samples=50, device=DEVICE):
    """Full MC Dropout analysis on a loader."""
    model.train()
    all_conf, all_unc, all_correct, all_pred, all_true = [], [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.stack([
                torch.softmax(model(x), dim=1) for _ in range(n_samples)
            ])
            mean = preds.mean(0)
            std = preds.std(0)
            conf, pred = mean.max(1)
            unc = std.mean(1)
            correct = (pred == y)
            all_conf.append(conf.cpu()); all_unc.append(unc.cpu())
            all_correct.append(correct.cpu()); all_pred.append(pred.cpu())
            all_true.append(y.cpu())
    model.eval()
    return {
        "conf": torch.cat(all_conf),
        "uncertainty": torch.cat(all_unc),
        "correct": torch.cat(all_correct),
        "pred": torch.cat(all_pred),
        "true": torch.cat(all_true),
    }


# ============================================================
# 2) Cross-Model Error Gallery
# ============================================================
def save_cross_model_errors(models_dict, dataset, val_loader,
                            save_path=os.path.join(ARTIFACTS, "cross_model_errors.png")):
    """
    Find images missed by 2+ models, and save their visual gallery.
    """
    from collections import defaultdict
    per_image = defaultdict(list)
    all_preds = {}

    for name, model in models_dict.items():
        model.eval()
        preds_all = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                preds_all.append(model(x).argmax(1).cpu())
        all_preds[name] = torch.cat(preds_all)

    # Collect labels
    labels = []
    for _, y in val_loader:
        labels.append(y)
    labels = torch.cat(labels)

    # Find multi-error images
    multi_error_indices = []
    for i in range(len(labels)):
        miss_count = sum(
            (all_preds[name][i].item() != labels[i].item())
            for name in models_dict
        )
        if miss_count >= 2:
            multi_error_indices.append(i)

    print(f"  Found {len(multi_error_indices)} images missed by 2+ models")

    # Visualize
    n = min(len(multi_error_indices), 12)
    if n == 0:
        print("  No cross-model errors to show.")
        return multi_error_indices

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for idx, img_idx in enumerate(multi_error_indices[:n]):
        img, true_label = dataset[img_idx]
        ax = axes[idx // 4, idx % 4]
        # Denormalize
        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        ax.imshow(np.clip(img_np, 0, 1))
        preds_str = " | ".join(
            f"{name[:6]}:{CLASSES[all_preds[name][img_idx].item()][:4]}"
            for name in models_dict
        )
        ax.set_title(f"True: {CLASSES[true_label.item()]}\n{preds_str}",
                     fontsize=8, color="red")
        ax.axis("off")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {save_path}")
    return multi_error_indices


# ============================================================
# 3) OOD Analysis on unclean
# ============================================================
def ood_analysis(model, unclean_items, class_to_idx, threshold=0.70):
    """
    Analyze 'unclean' as Out-Of-Distribution.
    - Known classes → should be classified normally
    - 'neysan' (unseen class) → should have LOW confidence
    """
    ds = IndexedImageDataset(unclean_items, class_to_idx, eval_transform())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    results = {"known": {"correct": 0, "total": 0, "low_conf": 0},
               "neysan": {"total": 0, "low_conf": 0, "predicted_classes": []}}

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            probs = torch.softmax(model(x), 1)
            conf, pred = probs.max(1)

            for i in range(x.size(0)):
                true_cls = CLASSES[y[i].item()] if y[i].item() < len(CLASSES) else "neysan"
                if true_cls == "neysan":
                    results["neysan"]["total"] += 1
                    if conf[i] < threshold:
                        results["neysan"]["low_conf"] += 1
                    results["neysan"]["predicted_classes"].append(
                        CLASSES[pred[i].item()])
                else:
                    results["known"]["total"] += 1
                    if pred[i] == y[i]:
                        results["known"]["correct"] += 1
                    if conf[i] < threshold:
                        results["known"]["low_conf"] += 1

    # Compute metrics
    if results["neysan"]["total"] > 0:
        results["neysan"]["review_rate"] = (
            results["neysan"]["low_conf"] / results["neysan"]["total"])
    if results["known"]["total"] > 0:
        results["known"]["accuracy"] = (
            results["known"]["correct"] / results["known"]["total"])
        results["known"]["review_rate"] = (
            results["known"]["low_conf"] / results["known"]["total"])

    return results


# ============================================================
# 4) Fuzzy Threshold
# ============================================================
def fuzzy_threshold(conf, low=0.3, high=0.9):
    """
    Fuzzy membership function for 'needs_review' decision.
    Returns a value in [0, 1]: 0 = definitely accept, 1 = definitely review.
    Uses a smooth sigmoid-like transition instead of hard cutoff.
    """
    # Smooth step: 0 at high, 1 at low, transition in between
    x = (conf - low) / (high - low)  # 0 at low, 1 at high
    x = np.clip(x, 0, 1)
    # Smoothstep function: 3x^2 - 2x^3
    review_score = 1 - (3 * x**2 - 2 * x**3)
    return review_score


def fuzzy_route(conf, low=0.3, high=0.9):
    """Three-band routing with fuzzy boundaries."""
    score = fuzzy_threshold(conf, low, high)
    if score < 0.2:
        return "ACCEPT"
    elif score > 0.8:
        return "REVIEW"
    else:
        return "VERIFY"


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("PHASE 3: Fixed UQ + Cross-Model Gallery + OOD + Fuzzy")
    print("=" * 70)

    # Build val set
    items = list_split(TRAIN_DIR)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    _, va_items, _, _ = stratified_val_split(items, class_to_idx,
                                             VAL_FRACTION, SEED)
    ds_va = IndexedImageDataset(va_items, class_to_idx, eval_transform())
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Val size: {len(ds_va)}")

    # Load models (NOTE: resnet needs to be re-trained with Dropout!)
    models = {}
    for name, arch in [("cnn_baseline", "cnn"),
                       ("resnet_feat", "resnet"),
                       ("resnet_ft", "resnet")]:
        try:
            m, _ = load_checkpoint(name, arch)
            models[name] = m
            print(f"  Loaded {name}")
        except Exception as e:
            print(f"  Skipping {name}: {e}")

    # 3.1) MC Dropout (only works on models with Dropout)
    print("\n>>> MC Dropout analysis")
    if "cnn_baseline" in models:
        res = mc_dropout_analysis(models["cnn_baseline"], va_loader, n_samples=50)
        corr = torch.corrcoef(torch.stack([res["uncertainty"],
                                            (~res["correct"]).float()]))[0, 1]
        print(f"  cnn_baseline: mean_unc={res['uncertainty'].mean():.4f}, "
              f"corr(unc, error)={corr:.3f}")
        print(f"  → Positive corr = uncertainty predicts errors ✓")

    # 3.2) Cross-model error gallery
    print("\n>>> Cross-model error gallery")
    if len(models) >= 2:
        multi_errors = save_cross_model_errors(models, ds_va, va_loader)
        with open(os.path.join(ARTIFACTS, "cross_model_indices.json"), "w") as f:
            json.dump(multi_errors, f)

    # 3.3) OOD analysis on unclean
    print("\n>>> OOD analysis on unclean")
    unclean_items = list_split(UNCLEAN_DIR)
    # Filter to known classes + neysan
    # Note: unclean may have 9 classes (8 known + neysan)
    if unclean_items:
        ood = ood_analysis(models.get("resnet_ft", list(models.values())[0]),
                           unclean_items, class_to_idx)
        print(f"  Known classes: acc={ood['known'].get('accuracy', 'N/A'):.3f}, "
              f"review_rate={ood['known'].get('review_rate', 'N/A'):.3f}")
        print(f"  Neysan (unseen): review_rate={ood['neysan'].get('review_rate', 'N/A'):.3f}")

    # 3.4) Fuzzy threshold demo
    print("\n>>> Fuzzy threshold demo")
    for conf in [0.2, 0.5, 0.7, 0.95]:
        route = fuzzy_route(conf, low=0.3, high=0.9)
        print(f"  conf={conf:.2f} → route={route}")

    print("\n✓ Phase 3 complete")


if __name__ == "__main__":
    main()