"""train_fixmatch_v2.py — FixMatch with class weighting + parallel evaluation.

Building on resnet_ft_v2, this script uses FixMatch (Sohn et al., NeurIPS 2020)
to leverage the unlabeled `unclean` split for semi-supervised learning.

Key improvements over train_fixmatch.py:
  1. Class weighting: emphasise `kamyun` and `kamyunet` (weak classes).
  2. Warm start: load resnet_ft_v2.pt and continue training.
  3. Parallel evaluation: run on ALL TESTING_DATA_ROOTS from config.
  4. Curriculum on lambda_u: warm-up unsupervised loss for stability.

Reference: https://arxiv.org/abs/2001.07685
"""
import os, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from collections import Counter

from config import (
    CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS,
    TRAIN_DIR, UNCLEAN_DIR, VAL_FRACTION, SEED,
    IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE,
    TESTING_DATA_ROOTS, TEST_FOLDERS_BY_ROOT, build_output_dir,
)
from data import (
    IndexedImageDataset, train_transform, eval_transform,
    stratified_val_split,
)
from audit import list_split
from analysis import load_checkpoint


# ============================================================
# 1) Reproducibility
# ============================================================
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2) Weak + Strong augmentations (FixMatch core)
# ============================================================
def weak_augment():
    """Weak: just resize + normalize (used to generate pseudo-labels)."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def strong_augment():
    """Strong: aggressive augmentation (used to train on pseudo-labels).

    RandAugment-like pipeline with rotation, color, translation, and erasing.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


# ============================================================
# 3) Unlabeled dataset — returns two views per image
# ============================================================
class UnlabeledDataset(Dataset):
    """Returns (weak_view, strong_view) for each unlabeled image."""

    def __init__(self, items, weak_tf, strong_tf):
        self.items = items
        self.weak = weak_tf
        self.strong = strong_tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, _ = self.items[i]  # label is ignored
        img = Image.open(path).convert("RGB")
        return self.weak(img), self.strong(img)


# ============================================================
# 4) Class-weighted FixMatch loss
# ============================================================
class FixMatchLoss(nn.Module):
    """
    Total loss = supervised (weighted CE) + lambda_u * unsupervised (masked CE).

    The supervised loss uses class weights to emphasise weak classes
    (kamyun, kamyunet) that were identified in earlier phases.
    """

    def __init__(self, class_weights, confidence_threshold=0.95):
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights))
        self.conf_thresh = confidence_threshold

    def forward(self, logits_l, y_l, logits_weak, logits_strong):
        # --- Supervised loss (class-weighted CE) ---
        
        # weight = self.class_weights
        # FIX: Move class weights to the same device as the input logits (e.g., GPU)
        # to prevent "Expected all tensors to be on the same device" RuntimeError.
        weight = self.class_weights.to(logits_l.device)
        
        loss_sup = F.cross_entropy(logits_l, y_l,weight=weight)

        # --- Unsupervised loss (FixMatch) ---
        with torch.no_grad():
            probs_weak = F.softmax(logits_weak, dim=1)
            max_probs, pseudo_labels = probs_weak.max(dim=1)
            mask = max_probs >= self.conf_thresh

        if mask.sum() == 0:
            return loss_sup, 0.0, 0.0

        loss_unsup = (F.cross_entropy(logits_strong, pseudo_labels,
                                       reduction="none") * mask.float()).mean()
        mask_ratio = mask.float().mean().item()
        return loss_sup, loss_unsup, mask_ratio


# ============================================================
# 5) Build loaders
# ============================================================
def build_loaders(batch_size_l=16, batch_size_u=32, num_workers=2):
    """
    Build labeled, unlabeled, and validation loaders.

    The unlabeled set = unclean (without neysan and without val overlap).
    """
    # --- Labeled training data ---
    items = list_split(TRAIN_DIR)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    # Filter to known classes (skip neysan if present)
    items = [(p, c) for p, c in items if c in class_to_idx]

    # Stratified split: 80/20
    tr_items, va_items, _, _ = stratified_val_split(
        items, class_to_idx, VAL_FRACTION, SEED
    )

    # --- Unlabeled data (from unclean) ---
    unclean_raw = list_split(UNCLEAN_DIR)
    unlabeled_items = [(p, c) for p, c in unclean_raw if c in class_to_idx]

    # Remove any unclean item whose content exists in train/val (avoid leakage)
    # Use file path as a cheap proxy (content hash would be more robust)
    train_paths = {p for p, _ in tr_items} | {p for p, _ in va_items}
    unlabeled_items = [(p, c) for p, c in unlabeled_items if p not in train_paths]

    print(f"  Labeled:   {len(tr_items)}")
    print(f"  Unlabeled: {len(unlabeled_items)}")
    print(f"  Val:       {len(va_items)}")

    # --- Datasets ---
    ds_l = IndexedImageDataset(tr_items, class_to_idx, train_transform(True))
    ds_u = UnlabeledDataset(unlabeled_items, weak_augment(), strong_augment())
    ds_v = IndexedImageDataset(va_items, class_to_idx, eval_transform())

    # --- Loaders ---
    loader_l = DataLoader(ds_l, batch_size=batch_size_l, shuffle=True,
                          drop_last=True, num_workers=num_workers,
                          pin_memory=True)
    loader_u = DataLoader(ds_u, batch_size=batch_size_u, shuffle=True,
                          drop_last=True, num_workers=num_workers,
                          pin_memory=True)
    loader_v = DataLoader(ds_v, batch_size=64, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    return loader_l, loader_u, loader_v, class_to_idx, tr_items, va_items


# ============================================================
# 6) Compute class weights for weak classes
# ============================================================
def compute_class_weights(tr_items, class_to_idx, weak_classes=None,
                          weak_multiplier=2.0):
    """
    Compute inverse-frequency weights, then boost weak classes.

    weak_classes: list of class names to upweight (e.g., ["kamyun", "kamyunet"])
    weak_multiplier: factor by which to multiply their weight
    """
    counts = Counter(c for _, c in tr_items)
    total = sum(counts.values())
    weights = []
    for cls in CLASSES:
        n = counts.get(cls, 1)
        w = total / (NUM_CLASSES * n)  # inverse frequency
        if weak_classes and cls in weak_classes:
            w *= weak_multiplier
        weights.append(w)

    # Normalize so mean weight = 1 (keeps loss scale stable)
    weights = np.array(weights)
    weights = weights / weights.mean()

    print("  Class weights:")
    for c, w in zip(CLASSES, weights):
        marker = " ⭐" if weak_classes and c in weak_classes else ""
        print(f"    {c:10s} {w:.3f}{marker}")
    return weights.tolist()


# ============================================================
# 7) Training loop
# ============================================================
def train_one_epoch(model, loader_l, loader_u, optimizer, criterion,
                    device, lambda_u, max_grad_norm=5.0):
    model.train()
    total_loss, total_sup, total_unsup = 0.0, 0.0, 0.0
    correct, total = 0, 0
    mask_sum, n_batches = 0.0, 0

    # Cycle unlabeled loader to match labeled iterations
    unl_iter = iter(loader_u)
    for x_l, y_l in loader_l:
        # Get unlabeled batch
        try:
            x_uw, x_us = next(unl_iter)
        except StopIteration:
            unl_iter = iter(loader_u)
            x_uw, x_us = next(unl_iter)

        x_l, y_l = x_l.to(device), y_l.to(device)
        x_uw, x_us = x_uw.to(device), x_us.to(device)

        optimizer.zero_grad()
        logits_l = model(x_l)
        logits_uw = model(x_uw)
        logits_us = model(x_us)

        loss_sup, loss_unsup, mask_ratio = criterion(
            logits_l, y_l, logits_uw, logits_us
        )
        loss = loss_sup + lambda_u * loss_unsup
        loss.backward()

        # Gradient clipping for stability with pseudo-labels
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            preds = logits_l.argmax(1)
            correct += (preds == y_l).sum().item()
            total += y_l.size(0)

        total_loss += loss.item()
        total_sup += loss_sup.item()
        total_unsup += loss_unsup.item()
        mask_sum += mask_ratio
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "loss_sup": total_sup / n_batches,
        "loss_unsup": total_unsup / n_batches,
        "train_acc": correct / total,
        "mask_ratio": mask_sum / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


# ============================================================
# 8) Parallel evaluation on all TESTING_DATA_ROOTS
# ============================================================
@torch.no_grad()
def evaluate_on_root(model, root, class_to_idx, device):
    """Evaluate on all folders inside a testing root."""
    from torch.utils.data import DataLoader
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support)

    folders = TEST_FOLDERS_BY_ROOT.get(root, [])
    if not folders:
        return None

    per_folder = []
    for folder in folders:
        items = list_split(folder)
        known_items = [(p, c) for p, c in items if c in class_to_idx]
        if not known_items:
            continue

        ds = IndexedImageDataset(known_items, class_to_idx, eval_transform())
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

        all_preds, all_labels = [], []
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(y)

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        acc = accuracy_score(y_true, y_pred)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0
        )
        _, _, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        per_folder.append({
            "folder": os.path.basename(folder),
            "n_images": len(known_items),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "per_class_recall": {CLASSES[i]: float(r[i])
                                  for i in range(NUM_CLASSES)},
        })

    if not per_folder:
        return None

    return {
        "root": root,
        "root_tag": os.path.basename(os.path.normpath(root)),
        "average_accuracy": float(np.mean([f["accuracy"] for f in per_folder])),
        "average_macro_f1": float(np.mean([f["macro_f1"] for f in per_folder])),
        "per_folder": per_folder,
    }


def save_root_report(result, root):
    out_dir = build_output_dir(root)
    path = os.path.join(out_dir, "fixmatch_v2_summary.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


# ============================================================
# 9) MAIN
# ============================================================
def main():
    print("=" * 70)
    print("FIXMATCH v2 — Semi-Supervised + Class Weighting + V2 Dataset")
    print("=" * 70)

    seed_everything()

    # --- 1) Loaders ---
    print("\n[1/5] Building loaders...")
    loader_l, loader_u, loader_v, class_to_idx, tr_items, va_items = \
        build_loaders(batch_size_l=16, batch_size_u=32, num_workers=2)

    # --- 2) Class weights ---
    print("\n[2/5] Computing class weights...")
    class_weights = compute_class_weights(
        tr_items, class_to_idx,
        weak_classes=["kamyun", "kamyunet"],
        weak_multiplier=2.0,
    )

    # --- 3) Model (warm start from resnet_ft_v2) ---
    print("\n[3/5] Building model with warm start...")
    ckpt_path = os.path.join(ARTIFACTS, "resnet_ft_v2.pt")
    if not os.path.exists(ckpt_path):
        print(f"  ✗ ERROR: {ckpt_path} not found.")
        print("  Please run retrain_resnet_ft.py first.")
        return

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["state_dict"]
    # Handle legacy key names
    new_state = {}
    for k, v in state.items():
        if k in ("fc.weight", "fc.bias"):
            new_state[f"fc.1.{k.split('.')[1]}"] = v
        else:
            new_state[k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"  [warn] missing: {missing}")
    print(f"  ✓ Loaded warm start from resnet_ft_v2.pt")

    # Freeze early layers — train layer3 + layer4 + fc
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(("layer3", "layer4", "fc"))

    model = model.to(DEVICE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_train:,}")

    # --- 4) Optimizer / Loss / Scheduler ---
    print("\n[4/5] Setting up training...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4, weight_decay=1e-4,
    )
    EPOCHS = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    criterion = FixMatchLoss(
        class_weights=class_weights, confidence_threshold=0.95
    )

    # Lambda_u warm-up: start at 0, ramp to 1 over first 5 epochs
    def lambda_u_at(epoch):
        return min(1.0, epoch / 5.0)

    # --- 5) Training loop ---
    print("\n[5/5] Training...")
    best_acc = 0.0
    history = []
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        lam = lambda_u_at(epoch)
        metrics = train_one_epoch(
            model, loader_l, loader_u, optimizer, criterion,
            DEVICE, lambda_u=lam,
        )
        va_acc = evaluate(model, loader_v, DEVICE)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "lambda_u": lam,
            "loss": metrics["loss"],
            "loss_sup": metrics["loss_sup"],
            "loss_unsup": metrics["loss_unsup"],
            "train_acc": metrics["train_acc"],
            "val_acc": va_acc,
            "mask_ratio": metrics["mask_ratio"],
        })

        print(f"ep {epoch:02d} | λ_u {lam:.2f} | "
              f"loss {metrics['loss']:.4f} "
              f"(sup {metrics['loss_sup']:.3f} / unsup {metrics['loss_unsup']:.3f}) | "
              f"tr_acc {metrics['train_acc']:.4f} | "
              f"va_acc {va_acc:.4f} | "
              f"mask {metrics['mask_ratio']:.3f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "arch": "resnet18_fixmatch_v2",
                "config": {
                    "epoch": epoch,
                    "val_acc": va_acc,
                    "lambda_u": lam,
                    "class_weights": class_weights,
                },
            }, os.path.join(ARTIFACTS, "fixmatch_v2.pt"))
            print(f"  ↳ New best: va_acc = {va_acc:.4f}")

    elapsed = time.time() - t_start
    print(f"\n✓ Training finished in {elapsed/60:.1f} min")
    print(f"  Best val acc: {best_acc:.4f}")
    print(f"  (ResNet-ft-v2 baseline: 0.9091)")

    # --- Parallel evaluation on all testing roots ---
    print("\n" + "=" * 70)
    print("PARALLEL EVALUATION")
    print("=" * 70)

    # Load best checkpoint
    ckpt = torch.load(os.path.join(ARTIFACTS, "fixmatch_v2.pt"),
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_results = {"model": "fixmatch_v2", "arch": "resnet", "roots": {}}
    for root in TESTING_DATA_ROOTS:
        tag = os.path.basename(os.path.normpath(root))
        print(f"\n>>> Evaluating on {tag}...")
        result = evaluate_on_root(model, root, class_to_idx, DEVICE)
        if result is None:
            print(f"  ✗ No valid folders in {root}")
            continue

        path = save_root_report(result, root)
        all_results["roots"][root] = {
            "tag": tag,
            "average_accuracy": result["average_accuracy"],
            "average_macro_f1": result["average_macro_f1"],
            "output": path,
        }
        print(f"  Accuracy: {result['average_accuracy']:.4f}")
        print(f"  Macro F1: {result['average_macro_f1']:.4f}")

        # Per-class recall comparison
        print(f"  Per-class recall (averaged over folders):")
        recalls = {}
        for cls in CLASSES:
            vals = [f["per_class_recall"][cls] for f in result["per_folder"]]
            recalls[cls] = float(np.mean(vals))
        for cls, r in sorted(recalls.items(), key=lambda x: x[1]):
            marker = " ⚠️" if r < 0.6 else ""
            print(f"    {cls:10s} {r:.3f}{marker}")

    # --- Save training history ---
    with open(os.path.join(ARTIFACTS, "fixmatch_v2_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(ARTIFACTS, "fixmatch_v2_all_roots.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("✓ FIXMATCH v2 COMPLETE")
    print("=" * 70)
    print(f"  Best val acc: {best_acc:.4f}")
    print(f"  Baseline:     0.9091")
    print(f"  Delta:        {best_acc - 0.9091:+.4f}")
    print(f"\nArtifacts: artifacts/fixmatch_v2*.json")


if __name__ == "__main__":
    main()