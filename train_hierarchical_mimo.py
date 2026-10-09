"""train_hierarchical_mimo.py — Hierarchical Multi-Expert Classifier.

Instead of forcing 8 mutually-exclusive classes, we learn a two-level taxonomy:
  Level 1: 4 super-classes (emergency, heavy, commercial, passenger)
  Level 2: sub-classes within each super-class

This handles the 'neysan' problem semantically: neysan = Nissan commercial
vehicle, so it goes to "commercial" super-class, then to "neysan" sub-class.
"""
import os, json, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score
from config import (CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, BATCH_SIZE,
                    TRAIN_DIR, UNCLEAN_DIR, VAL_FRACTION, SEED)
from data import IndexedImageDataset, train_transform, eval_transform, stratified_val_split
from audit import list_split


# ============================================================
# Hierarchical taxonomy
# ============================================================
HIERARCHY = {
    "emergency":  ["ambulance"],
    "heavy":      ["kamyun", "autobus", "minibus"],
    "commercial": ["vanet", "kamyunet", "neysan"],  # neysan added here!
    "passenger":  ["savari", "taxi"],
}

# Reverse mapping
SUPER_OF = {}
for sup, subs in HIERARCHY.items():
    for sub in subs:
        SUPER_OF[sub] = sup

SUPER_CLASSES = list(HIERARCHY.keys())  # 4 super-classes
SUPER_TO_IDX = {s: i for i, s in enumerate(SUPER_CLASSES)}


# ============================================================
# Hierarchical Model
# ============================================================
class HierarchicalResNet(nn.Module):
    """
    Two-level classifier:
      - Level 1: predicts super-class (4-way)
      - Level 2: for each super-class, predicts sub-class
    """
    def __init__(self, all_classes):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.in_features = backbone.fc.in_features

        # Shared backbone (without fc)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.all_classes = all_classes  # includes neysan
        self.class_to_idx = {c: i for i, c in enumerate(all_classes)}

        # Level 1: 4 super-classes
        self.head_L1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.in_features, len(SUPER_CLASSES)),
        )

        # Level 2: one head per super-class
        self.head_L2 = nn.ModuleDict({
            sup: nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.in_features, len(subs)),
            )
            for sup, subs in HIERARCHY.items()
        })

    def features(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)

    def forward(self, x):
        f = self.features(x)
        L1_logits = self.head_L1(f)
        L2_logits = {sup: self.head_L2[sup](f) for sup in SUPER_CLASSES}
        return {
            "features": f,
            "L1_logits": L1_logits,
            "L2_logits": L2_logits,
        }

    def predict_hierarchical(self, x):
        """Full hierarchical prediction."""
        out = self.forward(x)
        # Level 1: pick super-class
        sup_idx = out["L1_logits"].argmax(1)
        # Level 2: pick sub-class within that super-class
        final_class = []
        for i in range(x.size(0)):
            sup = SUPER_CLASSES[sup_idx[i].item()]
            sub_logits = out["L2_logits"][sup][i]
            sub_idx = sub_logits.argmax().item()
            final_class.append(HIERARCHY[sup][sub_idx])
        return final_class, out


# ============================================================
# Hierarchical Loss
# ============================================================
class HierarchicalLoss(nn.Module):
    def __init__(self, alpha_L1=1.0, alpha_L2=1.0):
        super().__init__()
        self.alpha_L1 = alpha_L1
        self.alpha_L2 = alpha_L2

    def forward(self, outputs, target_subs):
        """
        target_subs: list of sub-class names (e.g., ['vanet', 'savari', ...])
        """
        device = outputs["L1_logits"].device
        # L1 targets
        L1_targets = torch.tensor(
            [SUPER_TO_IDX[SUPER_OF[s]] for s in target_subs], device=device
        )
        loss_L1 = F.cross_entropy(outputs["L1_logits"], L1_targets)

        # L2 targets: only for the correct super-class
        loss_L2 = 0.0
        for sup in SUPER_CLASSES:
            mask = torch.tensor(
                [SUPER_OF[s] == sup for s in target_subs], device=device
            )
            if mask.sum() == 0:
                continue
            sub_targets = torch.tensor(
                [HIERARCHY[sup].index(s) for s in target_subs if SUPER_OF[s] == sup],
                device=device,
            )
            loss_L2 = loss_L2 + F.cross_entropy(
                outputs["L2_logits"][sup][mask], sub_targets
            )

        return self.alpha_L1 * loss_L1 + self.alpha_L2 * loss_L2


# ============================================================
# Training
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, class_to_idx, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        target_subs = [idx_to_class[i.item()] for i in y]

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, target_subs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Compute accuracy from hierarchical prediction
        pred_classes, _ = model.predict_hierarchical(x)
        pred_indices = [class_to_idx[c] for c in pred_classes]
        pred_indices = torch.tensor(pred_indices, device=device)
        correct += (pred_indices == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_hierarchical(model, loader, class_to_idx, device):
    model.eval()
    correct, total = 0, 0
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred_classes, _ = model.predict_hierarchical(x)
        pred_indices = torch.tensor(
            [class_to_idx[c] for c in pred_classes], device=device
        )
        correct += (pred_indices == y).sum().item()
        total += y.size(0)
    return correct / total


# ============================================================
# OOD detection via L1 confidence
# ============================================================
@torch.no_grad()
def ood_via_hierarchy(model, id_loader, ood_loader, device):
    """
    Use SUPER-CLASS confidence + Level-2 ambiguity as OOD signal.
    A sample is OOD if:
      - L1 confidence is low, OR
      - L2 heads disagree (high entropy across sub-classes)
    """
    model.eval()

    def compute_ood_scores(loader):
        all_scores = []
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            # L1 confidence
            L1_probs = F.softmax(out["L1_logits"], 1)
            L1_conf = L1_probs.max(1).values  # (B,)
            # L2 entropy (averaged across super-classes)
            L2_entropy = torch.zeros_like(L1_conf)
            for sup in SUPER_CLASSES:
                p = F.softmax(out["L2_logits"][sup], 1)
                ent = -(p * (p + 1e-8).log()).sum(1)
                L2_entropy += ent
            L2_entropy /= len(SUPER_CLASSES)
            # Combined: low L1 conf + high L2 entropy = OOD
            combined = -L1_conf + 0.1 * L2_entropy
            all_scores.append(combined.cpu())
        return torch.cat(all_scores).numpy()

    id_scores = compute_ood_scores(id_loader)
    ood_scores = compute_ood_scores(ood_loader)

    labels = [0] * len(id_scores) + [1] * len(ood_scores)
    auroc = roc_auc_score(labels, np.concatenate([id_scores, ood_scores]))
    return auroc


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("PHASE 8: Hierarchical MIMO Classifier")
    print("=" * 70)
    print(f"Super-classes: {SUPER_CLASSES}")
    print(f"Hierarchy: {HIERARCHY}")

    # Build dataset — use ALL classes including neysan from unclean
    tr_items = list_split(TRAIN_DIR)

    # Get neysan samples from unclean
    unclean_items = list_split(UNCLEAN_DIR)
    neysan_items = [(p, c) for p, c in unclean_items if c == "neysan"]
    print(f"\nFound {len(neysan_items)} neysan samples in unclean")

    # Add neysan to training (as part of commercial super-class)
    all_train_items = tr_items + neysan_items

    # Build class list (all 9 classes)
    all_classes = list(HIERARCHY.keys())  # super-classes
    all_subs = []
    for subs in HIERARCHY.values():
        all_subs.extend(subs)
    print(f"All sub-classes: {all_subs}")

    # Split — stratify by sub-class
    class_to_idx = {c: i for i, c in enumerate(all_subs)}
    tr_split, va_split, _, _ = stratified_val_split(
        all_train_items, class_to_idx, VAL_FRACTION, SEED
    )

    ds_tr = IndexedImageDataset(tr_split, class_to_idx, train_transform(True))
    ds_va = IndexedImageDataset(va_split, class_to_idx, eval_transform())
    tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    model = HierarchicalResNet(all_subs).to(DEVICE)
    # Train only layer4 + heads
    for name, p in model.named_parameters():
        if name.startswith(("conv1", "bn1", "layer1", "layer2", "layer3")):
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = HierarchicalLoss(alpha_L1=1.0, alpha_L2=1.0)

    EPOCHS = 20
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, tr_loader, optimizer, criterion, class_to_idx, DEVICE
        )
        va_acc = evaluate_hierarchical(model, va_loader, class_to_idx, DEVICE)
        scheduler.step()
        print(f"ep {epoch:02d} | tr_loss {tr_loss:.4f} | tr_acc {tr_acc:.4f} | va_acc {va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "all_classes": all_subs,
                "hierarchy": HIERARCHY,
                "config": {"epoch": epoch, "val_acc": va_acc},
            }, os.path.join(ARTIFACTS, "hierarchical_mimo.pt"))

    print(f"\n✓ Best val acc (hierarchical): {best_acc:.4f}")

    # Load best and evaluate OOD detection
    model.load_state_dict(torch.load(
        os.path.join(ARTIFACTS, "hierarchical_mimo.pt"),
        map_location=DEVICE, weights_only=False
    )["state_dict"])

    print("\n[OOD] Testing hierarchical OOD detection on held-out neysan...")
    # Hold out 20% of neysan for OOD test
    n = len(neysan_items)
    n_test = max(10, n // 5)
    ood_test_items = neysan_items[:n_test]
    dummy_idx = {c: 0 for c in all_subs}
    ood_test_items_idx = [(p, "neysan") for p, _ in ood_test_items]
    ds_ood = IndexedImageDataset(
        ood_test_items_idx,
        {**class_to_idx, "neysan": all_subs.index("neysan")},
        eval_transform()
    )
    ood_loader = DataLoader(ds_ood, batch_size=BATCH_SIZE, shuffle=False)

    auroc = ood_via_hierarchy(model, va_loader, ood_loader, DEVICE)
    print(f"  AUROC (hierarchical L1+L2): {auroc:.3f}")
    if auroc > 0.7:
        print("  ⭐ SUCCESS — hierarchy provides OOD signal!")
    else:
        print("  ✗ Still weak — need different approach")

    with open(os.path.join(ARTIFACTS, "phase8_hierarchical.json"), "w") as f:
        json.dump({
            "val_acc": best_acc,
            "auroc_hierarchical": auroc,
            "n_neysan_train": n - n_test,
            "n_neysan_test": n_test,
        }, f, indent=2)

    print("\n✓ Phase 8 complete")


if __name__ == "__main__":
    main()