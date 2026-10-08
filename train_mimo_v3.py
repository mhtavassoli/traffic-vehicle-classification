"""train_mimo_v3.py — Two-Stage MIMO: classifier first, then OOD experts."""
import os, json, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score
from config import (CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, BATCH_SIZE,
                    TRAIN_DIR, UNCLEAN_DIR, VAL_FRACTION, SEED)
from data import IndexedImageDataset, train_transform, eval_transform, stratified_val_split
from audit import list_split


# ============================================================
# Stage 1: Standard Classifier (proven to work)
# ============================================================
def build_classifier():
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    for name, p in net.named_parameters():
        p.requires_grad = name.startswith("layer4") or name.startswith("fc")
    return net


def extract_features(model, x):
    model.eval()
    with torch.no_grad():
        f = model.conv1(x); f = model.bn1(f); f = model.relu(f); f = model.maxpool(f)
        f = model.layer1(f); f = model.layer2(f); f = model.layer3(f); f = model.layer4(f)
        f = model.avgpool(f)
    return f.flatten(1)


# ============================================================
# Stage 2: OOD Experts (frozen backbone)
# ============================================================
class OODExperts(nn.Module):
    """Three OOD experts on top of frozen features."""
    def __init__(self, in_features=512, num_classes=NUM_CLASSES):
        super().__init__()
        # Expert A: Prototype
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, in_features) * 0.01
        )
        # Expert B: Reconstruction
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, in_features),
        )
        # Expert C: Confidence predictor
        self.conf_head = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, features):
        # Prototype distances
        diffs = features.unsqueeze(1) - self.prototypes.unsqueeze(0)
        proto_dists = diffs.norm(dim=-1)  # (B, C)

        # Reconstruction
        z = self.encoder(features)
        recon = self.decoder(z)
        recon_err = (features - recon).norm(dim=1, keepdim=True)

        # Confidence
        conf = self.conf_head(features)

        return {
            "proto_dists": proto_dists,
            "recon_error": recon_err,
            "confidence": conf,
        }


def train_ood_experts(features, labels, epoch, lr=1e-3):
    """Train OOD experts on cached features. Fast because backbone is frozen."""
    device = features.device
    experts = OODExperts().to(device)
    opt = torch.optim.AdamW(experts.parameters(), lr=lr, weight_decay=1e-4)

    N = features.size(0)
    batch_size = 64
    for ep in range(epoch):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            f_batch = features[idx]
            y_batch = labels[idx]
            opt.zero_grad()
            out = experts(f_batch)

            # Loss 1: prototype should be close to correct class prototype
            true_proto = out["proto_dists"][torch.arange(len(y_batch)), y_batch]
            l_proto = true_proto.mean()

            # Loss 2: reconstruction should be small
            l_recon = out["recon_error"].mean()

            # Loss 3: confidence should predict "is this a classifiable sample?"
            # All training samples are classifiable → confidence should be high
            l_conf = F.binary_cross_entropy(
                out["confidence"],
                torch.ones_like(out["confidence"])
            )

            loss = l_proto + 0.5 * l_recon + 0.5 * l_conf
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  [OOD experts] ep {ep+1}/{epoch} | loss {total_loss:.3f}")
    return experts


# ============================================================
# Evaluate: combine classifier + OOD experts
# ============================================================
def evaluate_combined(classifier, experts, loader, device=DEVICE):
    """Return classification acc + OOD scores."""
    classifier.eval(); experts.eval()
    all_labels = []
    all_proto_scores = []
    all_recon_scores = []
    all_conf_scores = []
    correct = 0; total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feats = extract_features(classifier, x)
            logits = classifier.fc(feats)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            out = experts(feats)
            # OOD score for each sample = min distance to any prototype
            min_proto = out["proto_dists"].min(1).values
            all_proto_scores.append(min_proto.cpu())
            all_recon_scores.append(out["recon_error"].squeeze(1).cpu())
            all_conf_scores.append(out["confidence"].squeeze(1).cpu())
            all_labels.append(y.cpu())

    return {
        "accuracy": correct / total,
        "proto_scores": torch.cat(all_proto_scores),
        "recon_scores": torch.cat(all_recon_scores),
        "conf_scores": torch.cat(all_conf_scores),
        "labels": torch.cat(all_labels),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("MIMO v3: Two-Stage (Classifier + OOD Experts on frozen features)")
    print("=" * 70)

    items = list_split(TRAIN_DIR)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    tr_items, va_items, _, _ = stratified_val_split(items, class_to_idx,
                                                    VAL_FRACTION, SEED)

    # ---------- STAGE 1: Classifier ----------
    print("\n[STAGE 1] Training classifier (like resnet_ft_v2)...")
    ds_tr = IndexedImageDataset(tr_items, class_to_idx, train_transform(True))
    ds_va = IndexedImageDataset(va_items, class_to_idx, eval_transform())
    tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    clf = build_classifier().to(DEVICE)

    # Load resnet_ft_v2 if exists (we already trained it!)
    ckpt_path = os.path.join(ARTIFACTS, "resnet_ft_v2.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt["state_dict"]
        new_state = {}
        for k, v in state.items():
            if k in ("fc.weight", "fc.bias"):
                new_state[f"fc.1.{k.split('.')[1]}"] = v
            else:
                new_state[k] = v
        clf.load_state_dict(new_state, strict=False)
        print(f"  ✓ Loaded existing resnet_ft_v2 weights (skip Stage 1)")
    else:
        print(f"  ✗ No resnet_ft_v2 checkpoint. Please run retrain_resnet_ft.py first.")
        return

    # Verify accuracy
    clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in va_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = clf(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"  Classifier val acc: {correct/total:.4f}")

    # ---------- STAGE 2: Extract features ----------
    print("\n[STAGE 2] Extracting features from train set...")
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for x, y in tr_loader:
            x = x.to(DEVICE)
            feats = extract_features(clf, x)
            all_feats.append(feats)
            all_labels.append(y.to(DEVICE))
    train_feats = torch.cat(all_feats)
    train_labels = torch.cat(all_labels)
    print(f"  Train features shape: {train_feats.shape}")

    # ---------- STAGE 3: Train OOD experts ----------
    print("\n[STAGE 3] Training OOD experts on frozen features...")
    experts = train_ood_experts(train_feats, train_labels, epoch=20, lr=1e-3)

    # ---------- EVALUATE ----------
    print("\n[EVAL] Classifier + OOD experts on validation...")
    results = evaluate_combined(clf, experts, va_loader)
    print(f"  Val accuracy: {results['accuracy']:.4f}")

    # ---------- OOD DETECTION on unclean ----------
    print("\n[EVAL] OOD detection on neysan...")
    unclean_items = list_split(UNCLEAN_DIR)
    known_items = [(p, c) for p, c in unclean_items if c in class_to_idx]
    unknown_items = [(p, c) for p, c in unclean_items if c not in class_to_idx]
    print(f"  Known: {len(known_items)}, Unknown (neysan): {len(unknown_items)}")

    # ID samples (validation, we already have scores)
    id_proto = results["proto_scores"].numpy()
    id_recon = results["recon_scores"].numpy()
    id_conf  = results["conf_scores"].numpy()

    # OOD samples (neysan)
    dummy_idx = {c: 0 for c in class_to_idx}
    for cls in set(c for _, c in unknown_items):
        dummy_idx[cls] = 0
    ds_ood = IndexedImageDataset(unknown_items, dummy_idx, eval_transform())
    ood_loader = DataLoader(ds_ood, batch_size=BATCH_SIZE, shuffle=False)

    ood_proto, ood_recon, ood_conf = [], [], []
    clf.eval(); experts.eval()
    with torch.no_grad():
        for x, _ in ood_loader:
            x = x.to(DEVICE)
            feats = extract_features(clf, x)
            out = experts(feats)
            ood_proto.append(out["proto_dists"].min(1).values.cpu())
            ood_recon.append(out["recon_error"].squeeze(1).cpu())
            ood_conf.append(out["confidence"].squeeze(1).cpu())

    ood_proto = torch.cat(ood_proto).numpy()
    ood_recon = torch.cat(ood_recon).numpy()
    ood_conf  = torch.cat(ood_conf).numpy()

    # AUROC
    labels_combined = [0] * len(id_proto) + [1] * len(ood_proto)

    auroc_proto = roc_auc_score(labels_combined,
                                 np.concatenate([id_proto, ood_proto]))
    auroc_recon = roc_auc_score(labels_combined,
                                 np.concatenate([id_recon, ood_recon]))
    # For confidence: ID should have HIGH conf, OOD should have LOW
    auroc_conf = roc_auc_score(labels_combined,
                                np.concatenate([-id_conf, -ood_conf]))

    print(f"\n  === OOD Detection AUROC ===")
    print(f"  Prototype min-dist:  {auroc_proto:.3f}")
    print(f"  Reconstruction err:  {auroc_recon:.3f}")
    print(f"  Confidence:          {auroc_conf:.3f}")

    # Combined fuzzy
    # Normalize scores to [0, 1]
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    id_combined = norm(id_proto) + norm(id_recon) - norm(id_conf)
    ood_combined = norm(ood_proto) + norm(ood_recon) - norm(ood_conf)
    auroc_combined = roc_auc_score(labels_combined,
                                    np.concatenate([id_combined, ood_combined]))
    print(f"  → Combined (fuzzy):  {auroc_combined:.3f}  ⭐")

    # Save
    summary = {
        "classifier_val_acc": results["accuracy"],
        "auroc_prototype": auroc_proto,
        "auroc_reconstruction": auroc_recon,
        "auroc_confidence": auroc_conf,
        "auroc_combined_fuzzy": auroc_combined,
    }
    with open(os.path.join(ARTIFACTS, "phase7_twostage_mimo.json"), "w") as f:
        json.dump(summary, f, indent=2)

    torch.save({
        "classifier_state": clf.state_dict(),
        "experts_state": experts.state_dict(),
        "class_to_idx": class_to_idx,
    }, os.path.join(ARTIFACTS, "twostage_mimo.pt"))

    print("\n✓ Phase 7 complete")


if __name__ == "__main__":
    main()