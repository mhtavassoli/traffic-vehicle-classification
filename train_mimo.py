"""train_mimo.py — Train the MIMO-Inspired classifier."""
import os, json, torch, numpy as np
from torch.utils.data import DataLoader
from config import (CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS, BATCH_SIZE,
                    TRAIN_DIR, VAL_FRACTION, SEED, EPOCHS, LR)
from data import IndexedImageDataset, train_transform, eval_transform, stratified_val_split
from audit import list_split
from models_mimo import build_mimo_classifier, MIMOLoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x, return_all=True)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        preds = outputs["logits"].argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, return_all=False)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    print("=" * 70)
    print("MIMO-Inspired Multi-Expert Training")
    print("=" * 70)

    items = list_split(TRAIN_DIR)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    tr_items, va_items, _, _ = stratified_val_split(items, class_to_idx,
                                                    VAL_FRACTION, SEED)

    ds_tr = IndexedImageDataset(tr_items, class_to_idx, train_transform(True))
    ds_va = IndexedImageDataset(va_items, class_to_idx, eval_transform())
    tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    model = build_mimo_classifier(NUM_CLASSES).to(DEVICE)

    # Freeze backbone, train only experts first
    for name, p in model.backbone.named_parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    criterion = MIMOLoss(alpha=1.0, beta=0.3, gamma=0.3, delta=0.3)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, optimizer, criterion, DEVICE)
        va_acc = evaluate(model, va_loader, DEVICE)
        print(f"ep {epoch:02d} | tr_loss {tr_loss:.4f} | tr_acc {tr_acc:.4f} | va_acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "arch": "mimo_resnet18",
                "config": {"epoch": epoch, "val_acc": va_acc}
            }, os.path.join(ARTIFACTS, "mimo_resnet18.pt"))

    print(f"\n✓ Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()