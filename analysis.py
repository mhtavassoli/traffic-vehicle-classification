"""analysis.py — Phase 2: Deep error analysis, calibration, and uncertainty."""
import os, json, numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS
from data import IndexedImageDataset, eval_transform, stratified_val_split
from models import TrafficCNN, build_resnet18
from evaluate import full_report, plot_confusion, rank_confused_pairs
from audit import run_audit, build_cleaned_index, list_split
from config import TRAIN_DIR, TEST_DIR


# ============================================================
# 1) Load a trained checkpoint
# ============================================================
def load_checkpoint(name, arch="cnn"):
    ckpt_path = os.path.join(ARTIFACTS, f"{name}.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if arch == "cnn":
        model = TrafficCNN(NUM_CLASSES).to(DEVICE)
    else:
        model = build_resnet18("fine_tune", NUM_CLASSES).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


# ============================================================
# 2) Temperature Scaling (post-hoc calibration)
# ============================================================
def find_temperature(model, val_loader, device=DEVICE, grid=50):
    """
    Temperature scaling: divide logits by T to fix overconfidence.
    T > 1 softens; T < 1 sharpens.
    """
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            all_logits.append(model(x).cpu())
            all_labels.append(y.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    temps = torch.linspace(0.5, 3.0, grid)
    nlls = [F.cross_entropy(logits / t, labels).item() for t in temps]
    best_temp = temps[int(np.argmin(nlls))].item()
    return best_temp, logits, labels


def compute_ece(logits, labels, n_bins=15):
    """Expected Calibration Error — how well confidence matches accuracy."""
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(1)
    acc = (pred == labels).float()

    bin_edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        if in_bin.sum() > 0:
            avg_conf = conf[in_bin].mean().item()
            avg_acc = acc[in_bin].mean().item()
            ece += (in_bin.sum().item() / len(conf)) * abs(avg_conf - avg_acc)
    return ece


def plot_reliability(logits, labels, path, n_bins=15, title="Reliability"):
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(1)
    acc = (pred == labels).float()
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    bin_conf, bin_acc = [], []
    for i in range(n_bins):
        in_bin = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        if in_bin.sum() > 0:
            bin_conf.append(conf[in_bin].mean().item())
            bin_acc.append(acc[in_bin].mean().item())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bin_conf, bin_acc, width=0.05, alpha=0.6, edgecolor="black")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 3) MC Dropout — uncertainty estimation
# ============================================================
def mc_dropout_predict(model, x, n_samples=30):
    """Keep Dropout ON during inference; measure variance."""
    model.train()  # Force dropout ON
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(torch.softmax(model(x), dim=1))
    preds = torch.stack(preds)         # (n_samples, batch, classes)
    mean = preds.mean(0)
    std = preds.std(0)
    model.eval()
    return mean, std


# ============================================================
# 4) Error Gallery + Cross-Model Overlap
# ============================================================
def plot_error_gallery(model, dataset, device=DEVICE, n=12,
                       path=os.path.join(ARTIFACTS, "error_gallery.png")):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    errors = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = torch.softmax(model(x), 1)
            conf, pred = probs.max(1)
            mask = pred != y
            for i in range(x.size(0)):
                if mask[i] and len(errors) < n:
                    errors.append((x[i].cpu().clone(), y[i].item(),
                                   pred[i].item(), conf[i].item()))
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for idx, (img, true, pred, cf) in enumerate(errors):
        ax = axes[idx // 4, idx % 4]
        img = img.permute(1, 2, 0).numpy()
        # denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"True: {CLASSES[true]}\nPred: {CLASSES[pred]} ({cf:.2f})",
                     color="red", fontsize=9)
        ax.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return errors


# ============================================================
# 5) Double-threshold optimization
# ============================================================
def optimize_double_threshold(logits, labels, review_cost=0.1):
    """Find (low, high) that minimize error + cost*review_rate on validation."""
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(1)
    correct = (pred == labels).float()

    best = (0.5, 0.9)
    best_score = float("inf")
    for low in np.arange(0.30, 0.70, 0.05):
        for high in np.arange(0.70, 0.99, 0.05):
            if low >= high: continue
            accept = conf >= high
            review = ~accept
            err = 1 - correct[accept].mean().item() if accept.sum() > 0 else 1.0
            score = err + review_cost * review.float().mean().item()
            if score < best_score:
                best_score, best = score, (low, high)
    return best, best_score


# ============================================================
# MAIN
# ============================================================
def run_phase2():
    print("=" * 70)
    print("PHASE 2: Deep Error Analysis + Calibration + UQ")
    print("=" * 70)

    # Build val set (same as in training)
    from config import VAL_FRACTION, SEED, BATCH_SIZE
    items = list_split(TRAIN_DIR)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    _, va_items, _, _ = stratified_val_split(items, class_to_idx,
                                             VAL_FRACTION, SEED)
    ds_va = IndexedImageDataset(va_items, class_to_idx, eval_transform())
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    results = {}
    for name, arch in [("cnn_baseline", "cnn"),
                       ("resnet_feat", "resnet"),
                       ("resnet_ft", "resnet")]:
        print(f"\n>>> Analyzing {name} ...")
        model, ckpt = load_checkpoint(name, arch)

        # 1) Temperature scaling
        T, logits, labels = find_temperature(model, va_loader)
        ece_before = compute_ece(logits, labels)
        ece_after = compute_ece(logits / T, labels)
        print(f"  Temperature: {T:.3f}")
        print(f"  ECE before: {ece_before:.4f}  →  after: {ece_after:.4f}")

        # 2) Reliability diagrams
        plot_reliability(logits, labels,
                         os.path.join(ARTIFACTS, f"{name}_reliability_before.png"))
        plot_reliability(logits / T, labels,
                         os.path.join(ARTIFACTS, f"{name}_reliability_after.png"))

        # 3) Double-threshold
        (low, high), score = optimize_double_threshold(logits, labels)
        print(f"  Double-threshold: low={low:.2f}, high={high:.2f}, score={score:.4f}")

        # 4) Error gallery
        errs = plot_error_gallery(model, ds_va, n=12,
                                  path=os.path.join(ARTIFACTS, f"{name}_errors.png"))
        print(f"  Saved {len(errs)} errors to gallery")

        results[name] = {
            "temperature": T,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "double_threshold": [low, high],
        }

    with open(os.path.join(ARTIFACTS, "phase2_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n✓ Phase 2 complete. Check artifacts/ directory.")


if __name__ == "__main__":
    run_phase2()