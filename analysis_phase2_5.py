"""analysis_phase2_5.py — Extended threshold + MC Dropout + Cross-Model."""
import os, json, numpy as np, torch,torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from config import (CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, BATCH_SIZE,
                    TRAIN_DIR, VAL_FRACTION, SEED)
from data import IndexedImageDataset, eval_transform, stratified_val_split
from audit import list_split
from analysis import load_checkpoint, find_temperature, compute_ece, plot_reliability


def optimize_double_threshold_v2(logits, labels, review_cost=0.1,
                                  low_range=(0.05, 0.60),
                                  high_range=(0.65, 0.995)):
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(1)
    correct = (pred == labels).float()

    best = (0.5, 0.9)
    best_score = float("inf")
    grid_results = []
    for low in np.arange(*low_range, 0.025):
        for high in np.arange(*high_range, 0.025):
            if low >= high: continue
            accept = conf >= high
            review = ~accept
            err = 1 - correct[accept].mean().item() if accept.sum() > 0 else 1.0
            score = err + review_cost * review.float().mean().item()
            grid_results.append((low, high, score,
                                 accept.float().mean().item()))
            if score < best_score:
                best_score, best = score, (low, high)
    return best, best_score, grid_results

def enable_dropout_only(model):
    """Keep BatchNorm in eval, enable only Dropout layers."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def mc_dropout_uncertainty(model, loader, n_samples=30, device=DEVICE):
    """MC Dropout: variance across stochastic forward passes."""
    # model.train()  # Dropout ON
    model.eval()  # Dropout ON
    enable_dropout_only(model) 
    results = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.stack([
                torch.softmax(model(x), dim=1) for _ in range(n_samples)
            ])
            mean = preds.mean(0)
            std = preds.std(0)
            conf, pred = mean.max(1)
            uncertainty = std.mean(1)
            correct = (pred == y)

            results.append({
                "conf": conf.cpu(), "uncertainty": uncertainty.cpu(),
                "correct": correct.cpu(), "pred": pred.cpu(), "true": y.cpu(),
                "std_per_class": std.cpu(),
            })
    model.eval()      
    return results


def cross_model_errors(models_dict, val_loader, device=DEVICE):
    """Find images misclassified by 2+ models."""
    from collections import defaultdict
    per_image = defaultdict(list)

    for name, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                mask = (pred != y)
                for i in range(x.size(0)):
                    key = (batch_idx, i)
                    if mask[i]:
                        per_image[key].append(name)

    # Count images missed by 2+ models
    multi_errors = {k: v for k, v in per_image.items() if len(v) >= 2}
    return multi_errors, per_image


def main():
    print("=" * 70)
    print("PHASE 2.5: Extended Analysis")
    print("=" * 70)

    # Build val set
    items = list_split(TRAIN_DIR)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    _, va_items, _, _ = stratified_val_split(items, class_to_idx,
                                             VAL_FRACTION, SEED)
    ds_va = IndexedImageDataset(va_items, class_to_idx, eval_transform())
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Val size: {len(ds_va)}")

    # Load all three models
    models = {}
    for name, arch in [("cnn_baseline", "cnn"),
                       ("resnet_feat", "resnet"),
                       ("resnet_ft", "resnet")]:
        m, _ = load_checkpoint(name, arch)
        models[name] = m

    # 1) Extended double-threshold grid
    print("\n>>> Extended double-threshold grid")
    for name, model in models.items():
        model.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits_list.append(model(x).cpu())
                labels_list.append(y.cpu())
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        (low, high), score, grid = optimize_double_threshold_v2(logits, labels)
        print(f"  {name}: low={low:.3f}, high={high:.3f}, score={score:.4f}")

    # 2) MC Dropout on resnet_ft
    print("\n>>> MC Dropout on resnet_ft")
    results = mc_dropout_uncertainty(models["resnet_ft"], va_loader, n_samples=30)
    all_conf = torch.cat([r["conf"] for r in results])
    all_unc = torch.cat([r["uncertainty"] for r in results])
    all_correct = torch.cat([r["correct"] for r in results])

    print(f"  Mean confidence: {all_conf.mean():.3f}")
    print(f"  Mean uncertainty: {all_unc.mean():.3f}")
    print(f"  Correlation (conf, correct): "
          f"{torch.corrcoef(torch.stack([all_conf, all_correct.float()]))[0,1]:.3f}")
    print(f"  Correlation (unc, correct): "
          f"{torch.corrcoef(torch.stack([all_unc, all_correct.float()]))[0,1]:.3f}")

    # 3) Cross-model errors
    print("\n>>> Cross-model error overlap")
    multi, per_image = cross_model_errors(models, va_loader)
    print(f"  Images missed by 2+ models: {len(multi)}")
    print(f"  Total unique error images: {len(per_image)}")

    # Save
    summary = {
        "extended_thresholds": {
            name: {"low": low, "high": high, "score": score}
            for name in models
        } if False else None,
        "mc_dropout_resnet_ft": {
            "mean_conf": float(all_conf.mean()),
            "mean_uncertainty": float(all_unc.mean()),
            "n_samples": 30,
        },
        "cross_model_multi_error_count": len(multi),
    }
    with open(os.path.join(ARTIFACTS, "phase2_5_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n✓ Phase 2.5 complete")


if __name__ == "__main__":
    main()