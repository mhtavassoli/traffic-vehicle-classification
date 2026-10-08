"""analysis_phase4.py — Phase 4: OOD Detection + MIMO + Fuzzy."""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import (CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, BATCH_SIZE,
                    TRAIN_DIR, UNCLEAN_DIR, VAL_FRACTION, SEED)
from data import IndexedImageDataset, eval_transform, stratified_val_split
from audit import list_split
from analysis import load_checkpoint


# ============================================================
# 1) Energy-Based OOD Detection
# ============================================================
def energy_score(logits, T=1.0):
    """
    Energy = -T * logsumexp(logits / T)
    Higher energy = more in-distribution.
    Lower energy = more OOD.
    Reference: Liu et al., NeurIPS 2020.
    """
    return -T * torch.logsumexp(logits / T, dim=1)


def fit_energy_threshold(model, val_loader, device=DEVICE, percentile=5):
    """Find the 5th percentile of in-distribution energies (validation)."""
    model.eval()
    all_energies = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            logits = model(x)
            all_energies.append(energy_score(logits).cpu())
    energies = torch.cat(all_energies)
    threshold = torch.quantile(energies, percentile / 100.0)
    return threshold.item(), energies


# ============================================================
# 2) Mahalanobis OOD Detection
# ============================================================
def fit_mahalanobis(model, train_loader, device=DEVICE):
    """
    Fit Gaussian per class on penultimate features.
    All tensors stay on `device` for speed.
    Reference: Lee et al., NeurIPS 2018.
    """
    model.eval()
    # Grab features from the layer before fc
    def get_features(x):
        # ResNet: features = avgpool output
        x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
        x = model.avgpool(x)
        return x.flatten(1)

    feats_per_class = {c: [] for c in range(NUM_CLASSES)}
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            feats = get_features(x)          # ← stays on device
            for i in range(x.size(0)):
                feats_per_class[y[i].item()].append(feats[i])

    # Compute per-class mean and shared covariance
    means = []
    for c in range(NUM_CLASSES):
        f = torch.stack(feats_per_class[c])
        means.append(f.mean(0))
    means = torch.stack(means)  # (C, D) on device

    # Shared covariance
    all_feats = []
    for c in range(NUM_CLASSES):
        f = torch.stack(feats_per_class[c])
        all_feats.append(f - means[c])
    all_feats = torch.cat(all_feats)
    cov = (all_feats.T @ all_feats) / all_feats.size(0)
    cov += torch.eye(cov.size(0), device=device) * 1e-4  # regularization, device match
    precision = torch.linalg.inv(cov)

    return means, precision, get_features


def mahalanobis_score(features, means, precision):
    """Min Mahalanobis distance across class means. Auto-moves to features' device."""
    means = means.to(features.device)
    precision = precision.to(features.device)

    # features: (B, D), means: (C, D), precision: (D, D)
    diff = features.unsqueeze(1) - means.unsqueeze(0)  # (B, C, D)
    # (B, C, D) @ (D, D) -> (B, C, D)
    m = torch.einsum("bcd,de->bce", diff, precision)
    # -> (B, C)
    dist = torch.einsum("bcd,bcd->bc", m, diff)
    return dist.min(1).values  # (B,)


# ============================================================
# 3) Fuzzy Three-Band Decision
# ============================================================
def fuzzy_three_band(conf, energy, mahal,
                     energy_thresh, mahal_thresh,
                     low=0.3, high=0.9):
    """
    Combine softmax confidence + energy + mahalanobis into a fuzzy decision.
    Returns: route ∈ {ACCEPT, VERIFY, REVIEW}, and a fuzzy membership score.
    """
    # Membership for "in-distribution" (higher = more likely ID)
    m_conf = float(np.clip((conf - low) / (high - low), 0, 1))
    m_energy = float(np.clip((energy - energy_thresh) / abs(energy_thresh), 0, 1))
    m_mahal = float(np.clip(1 - mahal / (mahal_thresh * 2 + 1e-6), 0, 1))

    # Fuzzy AND: minimum (conservative)
    id_score = min(m_conf, m_energy, m_mahal)

    # Smooth route
    if id_score > 0.7:
        return "ACCEPT", id_score
    elif id_score < 0.3:
        return "REVIEW", id_score
    else:
        return "VERIFY", id_score


# ============================================================
# 4) Evaluate OOD Detection with AUROC
# ============================================================
def evaluate_ood(model, val_loader, unclean_items, class_to_idx,
                 means, precision, get_features, energy_thresh,
                 device=DEVICE):
    """Compare softmax, energy, and mahalanobis on OOD detection."""
    from sklearn.metrics import roc_auc_score

    # In-distribution scores (validation)
    id_conf, id_energy, id_mahal = [], [], []
    model.eval()
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, 1)
            conf = probs.max(1).values
            energy = energy_score(logits)
            feats = get_features(x)
            mahal = mahalanobis_score(feats, means, precision)
            id_conf.append(conf.cpu()); id_energy.append(energy.cpu())
            id_mahal.append(mahal.cpu())

    id_conf = torch.cat(id_conf); id_energy = torch.cat(id_energy)
    id_mahal = torch.cat(id_mahal)

    # OOD scores (neysan)
    ood_conf, ood_energy, ood_mahal = [], [], []
    unknown_items = [(p, c) for p, c in unclean_items if c not in class_to_idx]
    if unknown_items:
        dummy_idx = {c: 0 for c in class_to_idx}
        for cls in set(c for _, c in unknown_items):
            dummy_idx[cls] = 0
        ds_ood = IndexedImageDataset(unknown_items, dummy_idx, eval_transform())
        loader = DataLoader(ds_ood, batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, 1)
                conf = probs.max(1).values
                energy = energy_score(logits)
                feats = get_features(x)
                mahal = mahalanobis_score(feats, means, precision)
                ood_conf.append(conf.cpu()); ood_energy.append(energy.cpu())
                ood_mahal.append(mahal.cpu())

    ood_conf = torch.cat(ood_conf); ood_energy = torch.cat(ood_energy)
    ood_mahal = torch.cat(ood_mahal)

    # AUROC: 1 = perfect, 0.5 = random
    # For confidence: OOD should have LOWER conf → use -conf as score
    auroc_conf = roc_auc_score(
        [0] * len(id_conf) + [1] * len(ood_conf),
        (-id_conf).tolist() + (-ood_conf).tolist()
    )
    # For energy: OOD should have LOWER energy → use -energy as score
    auroc_energy = roc_auc_score(
        [0] * len(id_energy) + [1] * len(ood_energy),
        (-id_energy).tolist() + (-ood_energy).tolist()
    )
    # For mahalanobis: OOD should have HIGHER mahal
    auroc_mahal = roc_auc_score(
        [0] * len(id_mahal) + [1] * len(ood_mahal),
        id_mahal.tolist() + ood_mahal.tolist()
    )
    
    # Move everything to the CPU before returning.
    return {
        "auroc_softmax": float(auroc_conf),
        "auroc_energy": float(auroc_energy),
        "auroc_mahalanobis": float(auroc_mahal),
        "id_energy_mean": float(id_energy.mean().item()),
        "ood_energy_mean": float(ood_energy.mean().item()),
        "id_mahal_mean": float(id_mahal.mean().item()),
        "ood_mahal_mean": float(ood_mahal.mean().item()),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("PHASE 4: OOD Detection + Energy + Mahalanobis + Fuzzy")
    print("=" * 70)

    # Build val set
    items = list_split(TRAIN_DIR)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    tr_items, va_items, _, _ = stratified_val_split(items, class_to_idx,
                                                    VAL_FRACTION, SEED)
    ds_tr = IndexedImageDataset(tr_items, class_to_idx, eval_transform())
    ds_va = IndexedImageDataset(va_items, class_to_idx, eval_transform())
    tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=False)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Val size: {len(ds_va)}, Train size: {len(ds_tr)}")

    # Load best model
    model, _ = load_checkpoint("resnet_ft_v2", "resnet")
    print("✓ Loaded resnet_ft_v2")

    # --- 4.1) Energy threshold on validation ---
    print("\n>>> Fitting energy threshold on validation (5th percentile)...")
    e_thresh, id_energies = fit_energy_threshold(model, va_loader, percentile=5)
    print(f"  Energy threshold: {e_thresh:.4f}")
    print(f"  ID energy range: [{id_energies.min():.2f}, {id_energies.max():.2f}]")

    # --- 4.2) Mahalanobis fit on train ---
    print("\n>>> Fitting Mahalanobis on training features...")
    means, precision, get_features = fit_mahalanobis(model, tr_loader)
    print(f"  Class means: {means.shape}, Precision: {precision.shape}")

    # --- 4.3) Evaluate OOD detection ---
    print("\n>>> Evaluating OOD detection...")
    unclean_items = list_split(UNCLEAN_DIR)
    results = evaluate_ood(model, va_loader, unclean_items, class_to_idx,
                          means, precision, get_features, e_thresh)
    print(f"  AUROC (softmax conf):     {results['auroc_softmax']:.3f}")
    print(f"  AUROC (energy):           {results['auroc_energy']:.3f}")
    print(f"  AUROC (mahalanobis):      {results['auroc_mahalanobis']:.3f}")
    print(f"  ID  energy mean: {results['id_energy_mean']:.3f}, "
          f"OOD energy mean: {results['ood_energy_mean']:.3f}")
    print(f"  ID  mahal mean:  {results['id_mahal_mean']:.3f}, "
          f"OOD mahal mean:  {results['ood_mahal_mean']:.3f}")

    # --- 4.4) Fuzzy Three-Band Demo ---
    print("\n>>> Fuzzy three-band demo")
    # Simulate some confidences
    fake_energy = e_thresh
    fake_mahal = results['id_mahal_mean']
    for conf in [0.3, 0.5, 0.7, 0.95]:
        route, score = fuzzy_three_band(
            conf, fake_energy, fake_mahal,
            e_thresh, results['id_mahal_mean']
        )
        print(f"  conf={conf:.2f} → route={route} (score={score:.2f})")

    # --- Save ---
    with open(os.path.join(ARTIFACTS, "phase4_ood.json"), "w") as f:
        json.dump({**results, "energy_threshold": e_thresh}, f, indent=2)
    print("\n✓ Phase 4 complete")


if __name__ == "__main__":
    main()