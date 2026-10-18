"""train_antishortcut.py — Train AntiShortcutNet (Phase 11).

Loads warm start from resnet_ft_v2, replaces the head with:
    - MultiRegionAttention (4 regions)
    - FuzzyClassifier (8 Gaussian memberships)

Key metrics after training:
    - Val accuracy (should stay high, ~0.92+)
    - Attention distribution (should NOT be bottom-heavy)
    - Test3 blue-van predictions (should be mostly 'vanet', not 'ambulance')
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, TRAIN_DIR,
    VAL_FRACTION, SEED, BATCH_SIZE,
)
from data import (
    IndexedImageDataset, train_transform, eval_transform,
    stratified_val_split,
)
from audit import list_split
from models_attention import AntiShortcutNet, FuzzyLoss


OUT_DIR = os.path.join(ARTIFACTS, "antishortcut")
os.makedirs(OUT_DIR, exist_ok=True)


def seed_everything(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, return_attention=True)
        loss = criterion(out["memberships"], out["attention"], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        preds = out["memberships"].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        total_loss += loss.item() * y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    attn_sum = torch.zeros(4).to(device)
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, return_attention=True)
        preds = out["memberships"].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        attn_sum += out["attention"].sum(dim=0)
        n += out["attention"].size(0)
    return correct / total, (attn_sum / n).cpu().tolist()


def main():
    print("=" * 70)
    print("PHASE 11: Anti-Shortcut Training")
    print("=" * 70)

    seed_everything()

    # --- Data ---
    items = list_split(TRAIN_DIR)
    c2i = {c: i for i, c in enumerate(CLASSES)}
    items = [(p, c) for p, c in items if c in c2i]
    tr_items, va_items, _, _ = stratified_val_split(items, c2i,
                                                     VAL_FRACTION, SEED)
    print(f"  Train: {len(tr_items)}, Val: {len(va_items)}")

    ds_tr = IndexedImageDataset(tr_items, c2i, train_transform(True))
    ds_va = IndexedImageDataset(va_items, c2i, eval_transform())
    loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=64, shuffle=False,
                            num_workers=0, pin_memory=True)

    # --- Model (warm start backbone) ---
    print("\n  Building AntiShortcutNet ...")
    model = AntiShortcutNet(num_classes=NUM_CLASSES,
                             freeze_backbone=False).to(DEVICE)

    # Warm-start backbone from resnet_ft_v2 (optional, but helps)
    ckpt_path = os.path.join(ARTIFACTS, "resnet_ft_v2.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt["state_dict"]
        # Filter backbone keys
        backbone_state = {k: v for k, v in state.items()
                          if not k.startswith(("fc",))}
        missing, unexpected = model.load_state_dict(backbone_state,
                                                    strict=False)
        print(f"  Loaded backbone (missing: {len(missing)}, unexpected: {len(unexpected)})")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_train:,}")

    # --- Optimizer + Loss ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    EPOCHS = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5,
    )
    criterion = FuzzyLoss(alpha=0.0, beta=0.1)

    # --- Training ---
    print("\n  Training ...")
    best_acc = 0.0
    history = []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, loader_tr, optimizer,
                                       criterion, DEVICE)
        va_acc, va_attn = evaluate(model, loader_va, DEVICE)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "tr_loss": tr_loss,
            "tr_acc": tr_acc,
            "va_acc": va_acc,
            "attention": va_attn,
        })

        attn_str = " ".join([f"{a:.2f}" for a in va_attn])
        print(f"ep {epoch:02d} | loss {tr_loss:.4f} | "
              f"tr_acc {tr_acc:.4f} | va_acc {va_acc:.4f} | "
              f"attn [{attn_str}]")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": c2i,
                "arch": "antishortcut_resnet18",
                "config": {"epoch": epoch, "val_acc": va_acc,
                            "attention": va_attn},
            }, os.path.join(ARTIFACTS, "antishortcut.pt"))

    elapsed = time.time() - t0
    print(f"\n✓ Best val acc: {best_acc:.4f} ({elapsed/60:.1f} min)")

    # --- Report attention distribution ---
    final_attn = history[-1]["attention"]
    regions = ["TOP", "MIDDLE", "BOTTOM", "FULL"]
    print("\n  Final attention distribution (per region):")
    for name, w in zip(regions, final_attn):
        bar = "█" * int(w * 50)
        print(f"    {name:8s} {w:.3f}  {bar}")

    with open(os.path.join(OUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()