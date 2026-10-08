"""analysis_phase5.py — Phase 5: Five Alternative OOD Methods + MIMO."""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from config import (CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, BATCH_SIZE,
                    TRAIN_DIR, UNCLEAN_DIR, VAL_FRACTION, SEED)
from data import IndexedImageDataset, eval_transform, stratified_val_split
from audit import list_split
from analysis import load_checkpoint


# ============================================================
# Helper: extract penultimate features
# ============================================================
def make_feature_extractor(model):
    def get_features(x):
        x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
        x = model.avgpool(x)
        return x.flatten(1)
    return get_features


# ============================================================
# Method 1: ODIN (Temperature + Input Perturbation)
# ============================================================
def odin_score(model, x, temperature=1000.0, epsilon=0.0014):
    """
    ODIN: Liang et al., ICLR 2018.
    1. Apply temperature scaling to logits
    2. Perturb input toward higher confidence
    3. Use max softmax as OOD score
    """
    x = x.clone().detach().requires_grad_(True)
    logits = model(x) / temperature
    pred = logits.argmax(1)
    loss = F.cross_entropy(logits, pred)
    loss.backward()
    x_adv = x - epsilon * x.grad.sign()
    x_adv = x_adv.detach()
    with torch.no_grad():
        logits_adv = model(x_adv) / temperature
    return F.softmax(logits_adv, 1).max(1).values


# ============================================================
# Method 2: ReAct (Rectified Activations)
# ============================================================
def react_score(model, x, percentile=90):
    """
    ReAct: Sun et al., ICLR 2021.
    Clip activations at `percentile` of in-distribution features.
    """
    # We'll compute this after extracting features
    pass


def fit_react_threshold(feats_id, percentile=90):
    """Find percentile threshold from ID features."""
    flat = feats_id.flatten().cpu().numpy()
    return float(np.percentile(flat, percentile))


# ============================================================
# Method 3: KNN-based OOD
# ============================================================
def fit_knn_features(model, train_loader, device=DEVICE):
    """Store all training features for KNN distance computation."""
    get_features = make_feature_extractor(model)
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            feats = get_features(x)
            all_feats.append(feats.cpu())
            all_labels.append(y)
    return torch.cat(all_feats), torch.cat(all_labels)


def knn_ood_score(features, train_feats, k=5):
    """Distance to k-th nearest neighbor in training features."""
    # features: (B, D), train_feats: (N, D)
    # Compute pairwise L2 distances
    dist = torch.cdist(features.cpu().float(), train_feats.float(), p=2)  # (B, N)
    # k-th smallest
    kth = dist.topk(k, dim=1, largest=False).values[:, -1]
    return kth


# ============================================================
# Method 4: ViM (Virtual Logit Matching)
# ============================================================
def fit_vim(model, train_loader, device=DEVICE, n_components=50):
    """
    ViM: Wang et al., CVPR 2022.
    Decompose features into principal subspace + residual.
    OOD score = residual norm.
    """
    feats, _ = fit_knn_features(model, train_loader, device)
    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(feats.numpy())
    # Project into principal subspace and back to get residual
    feats_projected = pca.inverse_transform(pca.transform(feats.numpy()))
    residual = feats.numpy() - feats_projected
    # Store the subspace and mean residual norm
    return {
        "pca": pca,
        "mean_residual": float(np.linalg.norm(residual, axis=1).mean()),
    }


def vim_score(features, vim_fit):
    pca = vim_fit["pca"]
    feat_np = features.cpu().numpy()
    feat_proj = pca.inverse_transform(pca.transform(feat_np))
    residual = feat_np - feat_proj
    return torch.from_numpy(np.linalg.norm(residual, axis=1)).float()


# ============================================================
# Method 5: Mahalanobis with PCA
# ============================================================
def fit_mahalanobis_pca(model, train_loader, device=DEVICE, n_components=50):
    """Mahalanobis on PCA-reduced features."""
    feats, labels = fit_knn_features(model, train_loader, device)
    pca = PCA(n_components=n_components, random_state=SEED)
    feats_np = pca.fit_transform(feats.numpy())
    feats_reduced = torch.from_numpy(feats_np).float()

    means = []
    for c in range(NUM_CLASSES):
        mask = (labels == c)
        means.append(feats_reduced[mask].mean(0))
    means = torch.stack(means)

    # Shared covariance in reduced space (50×50 — well-conditioned!)
    diffs = []
    for c in range(NUM_CLASSES):
        mask = (labels == c)
        diffs.append(feats_reduced[mask] - means[c])
    diffs = torch.cat(diffs)
    cov = (diffs.T @ diffs) / diffs.size(0)
    cov += torch.eye(cov.size(0)) * 1e-3
    precision = torch.linalg.inv(cov)

    return {"pca": pca, "means": means, "precision": precision}


def mahalanobis_pca_score(features, fit):
    pca = fit["pca"]
    feats_np = pca.transform(features.cpu().numpy())
    feats_red = torch.from_numpy(feats_np).float()

    diff = feats_red.unsqueeze(1) - fit["means"].unsqueeze(0)
    m = torch.einsum("bcd,de->bce", diff, fit["precision"])
    dist = torch.einsum("bcd,bcd->bc", m, diff)
    return dist.min(1).values


# ============================================================
# Evaluate all methods
# ============================================================
def evaluate_methods(model, train_loader, val_loader, unclean_items,
                     class_to_idx, device=DEVICE):
    print("  Extracting features...")
    get_features = make_feature_extractor(model)

    # ID features from validation
    id_conf, id_energy, id_feats = [], [], []
    model.eval()
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, 1)
            id_conf.append(probs.max(1).values.cpu())
            id_energy.append((-torch.logsumexp(logits, dim=1)).cpu())
            id_feats.append(get_features(x).cpu())

    id_conf = torch.cat(id_conf)
    id_energy = torch.cat(id_energy)
    id_feats = torch.cat(id_feats)

    # OOD features (neysan)
    unknown_items = [(p, c) for p, c in unclean_items if c not in class_to_idx]
    dummy_idx = {c: 0 for c in class_to_idx}
    for cls in set(c for _, c in unknown_items):
        dummy_idx[cls] = 0

    ds_ood = IndexedImageDataset(unknown_items, dummy_idx, eval_transform())
    loader = DataLoader(ds_ood, batch_size=BATCH_SIZE, shuffle=False)

    ood_conf, ood_energy, ood_feats = [], [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, 1)
            ood_conf.append(probs.max(1).values.cpu())
            ood_energy.append((-torch.logsumexp(logits, dim=1)).cpu())
            ood_feats.append(get_features(x).cpu())

    ood_conf = torch.cat(ood_conf)
    ood_energy = torch.cat(ood_energy)
    ood_feats = torch.cat(ood_feats)

    # Fit methods on training data
    print("  Fitting KNN...")
    train_feats, train_labels = fit_knn_features(model, train_loader, device)
    print("  Fitting ViM...")
    vim_fit = fit_vim(model, train_loader, device, n_components=50)
    print("  Fitting Mahalanobis+PCA...")
    maha_fit = fit_mahalanobis_pca(model, train_loader, device, n_components=50)
    print("  Computing ReAct threshold...")
    react_thresh = fit_react_threshold(train_feats, percentile=90)

    # Scores
    labels_combined = [0] * len(id_feats) + [1] * len(ood_feats)

    # 1. Softmax
    scores_softmax = (-id_conf).tolist() + (-ood_conf).tolist()
    auroc_softmax = roc_auc_score(labels_combined, scores_softmax)

    # 2. Energy (FIXED direction: +energy for OOD)
    scores_energy = id_energy.tolist() + ood_energy.tolist()
    auroc_energy = roc_auc_score(labels_combined, scores_energy)

    # 3. KNN
    id_knn = knn_ood_score(id_feats, train_feats, k=5)
    ood_knn = knn_ood_score(ood_feats, train_feats, k=5)
    scores_knn = id_knn.tolist() + ood_knn.tolist()
    auroc_knn = roc_auc_score(labels_combined, scores_knn)

    # 4. ViM
    id_vim = vim_score(id_feats, vim_fit)
    ood_vim = vim_score(ood_feats, vim_fit)
    scores_vim = id_vim.tolist() + ood_vim.tolist()
    auroc_vim = roc_auc_score(labels_combined, scores_vim)

    # 5. Mahalanobis + PCA
    id_maha = mahalanobis_pca_score(id_feats, maha_fit)
    ood_maha = mahalanobis_pca_score(ood_feats, maha_fit)
    scores_maha = id_maha.tolist() + ood_maha.tolist()
    auroc_maha = roc_auc_score(labels_combined, scores_maha)

    return {
        "softmax": auroc_softmax,
        "energy": auroc_energy,
        "knn": auroc_knn,
        "vim": auroc_vim,
        "mahalanobis_pca": auroc_maha,
        "react_threshold": react_thresh,
        "id_energy_mean": id_energy.mean().item(),
        "ood_energy_mean": ood_energy.mean().item(),
        "id_feats_mean_norm": id_feats.norm(dim=1).mean().item(),
        "ood_feats_mean_norm": ood_feats.norm(dim=1).mean().item(),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("PHASE 5: Five Alternative OOD Methods")
    print("=" * 70)

    items = list_split(TRAIN_DIR)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    tr_items, va_items, _, _ = stratified_val_split(items, class_to_idx,
                                                    VAL_FRACTION, SEED)
    ds_tr = IndexedImageDataset(tr_items, class_to_idx, eval_transform())
    ds_va = IndexedImageDataset(va_items, class_to_idx, eval_transform())
    tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=False)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    model, _ = load_checkpoint("resnet_ft_v2", "resnet")
    print("✓ Loaded resnet_ft_v2")

    unclean_items = list_split(UNCLEAN_DIR)
    results = evaluate_methods(model, tr_loader, va_loader,
                              unclean_items, class_to_idx)

    print("\n" + "=" * 70)
    print("OOD DETECTION RESULTS (AUROC, higher = better)")
    print("=" * 70)
    methods = [
        ("Softmax conf", results["softmax"]),
        ("Energy", results["energy"]),
        ("KNN (k=5)", results["knn"]),
        ("ViM (50 PCA)", results["vim"]),
        ("Mahalanobis+PCA", results["mahalanobis_pca"]),
    ]
    for name, auroc in methods:
        marker = "✓" if auroc > 0.75 else ("~" if auroc > 0.6 else "✗")
        print(f"  {name:20s}  AUROC = {auroc:.3f}  {marker}")

    print(f"\nReAct threshold (90th pct): {results['react_threshold']:.3f}")
    print(f"ID  energy mean: {results['id_energy_mean']:.3f}")
    print(f"OOD energy mean: {results['ood_energy_mean']:.3f}")
    print(f"ID  feature norm: {results['id_feats_mean_norm']:.2f}")
    print(f"OOD feature norm: {results['ood_feats_mean_norm']:.2f}")

    with open(os.path.join(ARTIFACTS, "phase5_ood.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n✓ Phase 5 complete")


if __name__ == "__main__":
    main()