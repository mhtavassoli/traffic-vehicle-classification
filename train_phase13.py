"""train_phase13.py — Phase13Net (Topology + MIMO + Entropy Routing)."""
import os
import json
import time

import torch
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
from models_phase13 import Phase13Net, Phase13Loss


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    cls_sum, sh_sum, ent_sum, bal_sum = 0.0, 0.0, 0.0, 0.0
    attn_w_sum, topo_w_sum = 0.0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x, return_all=True)

        # Get 5 new outputs
        loss, cls_l, sh_l, ent, bal = criterion(
            out["memberships"], 
            out["attention"], 
            out["expert_weights"], 
            y
        )


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = y.size(0)
        preds = out["memberships"].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += bs

        total_loss += loss.item() * bs
        cls_sum    += cls_l       * bs
        sh_sum     += sh_l        * bs
        ent_sum    += ent         * bs
        bal_sum    += bal         * bs
        
        attn_w_sum += out["expert_weights"][:, 0].sum().item()
        topo_w_sum += out["expert_weights"][:, 1].sum().item()

    return {
        "loss":     total_loss / total,
        "acc":      correct / total,
        "attn_w":   attn_w_sum / total,
        "topo_w":   topo_w_sum / total,
        "cls_loss": cls_sum / total,
        "sh_loss":  sh_sum / total,
        "entropy":  ent_sum / total,
        "balance":  bal_sum / total
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, return_all=True)
        preds = out["memberships"].argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    print("=" * 70)
    print("PHASE 13: MIMO + Topology + Entropy Routing (Reject Option ready)")
    print("=" * 70)

    items = list_split(TRAIN_DIR)
    c2i = {c: i for i, c in enumerate(CLASSES)}
    items = [(p, c) for p, c in items if c in c2i]
    tr_items, va_items, _, _ = stratified_val_split(items, c2i, VAL_FRACTION, SEED)

    ds_tr = IndexedImageDataset(tr_items, c2i, train_transform(True))
    ds_va = IndexedImageDataset(va_items, c2i, eval_transform())
    loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=64, shuffle=False,
                           num_workers=2, pin_memory=True)

    # The correct model →
    model = Phase13Net(num_classes=NUM_CLASSES).to(DEVICE)

    # Warm-start: first phase12.pt، then antishortcut.pt
    for ckpt_name in ("phase12.pt", "antishortcut.pt"):
        ckpt_path = os.path.join(ARTIFACTS, ckpt_name)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            state = model.state_dict()
            loaded = 0
            for k, v in ckpt["state_dict"].items():
                if k in state and state[k].shape == v.shape:
                    state[k] = v
                    loaded += 1
            model.load_state_dict(state)
            print(f"  ✓ Warm-started from {ckpt_name} ({loaded} tensors).")
            break

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    EPOCHS = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5,
    )
    criterion = Phase13Loss(beta_shortcut=0.1, gamma_entropy=0.1, delta_balance=0.5)

    best_acc = 0.0
    history = []
    t0 = time.time()

    print("\n  Training ...")
    for epoch in range(1, EPOCHS + 1):
        tr = train_epoch(model, loader_tr, optimizer, criterion, DEVICE)
        va_acc = evaluate(model, loader_va, DEVICE)
        scheduler.step()

        print(
            f"ep {epoch:02d} | loss {tr['loss']:.4f} (cls {tr['cls_loss']:.3f}, sh {tr['sh_loss']:.3f}, ent {tr['entropy']:.3f}) | "
            f"tr {tr['acc']:.4f} | va {va_acc:.4f} | "
            f"W: Attn={tr['attn_w']:.2f} Topo={tr['topo_w']:.2f}"
        )

        history.append({"epoch": epoch, "va_acc": va_acc, **tr})

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": c2i,
                "arch": "phase13_mimo_entropy",
                "config": {"epoch": epoch, "val_acc": va_acc},
            }, os.path.join(ARTIFACTS, "phase13.pt"))

    print(f"\n✓ Best val acc: {best_acc:.4f} ({(time.time()-t0)/60:.1f} min)")
    with open(os.path.join(ARTIFACTS, "phase13_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()