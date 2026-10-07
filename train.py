"""Single training loop shared by every experiment (CE / BCE / schedulers)."""
import time, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from config import NUM_CLASSES, DEVICE  # set as "cuda"/"cpu" in config

def make_loss(loss_type):
    """Return (criterion, target_adapter)."""
    if loss_type == "ce":
        return nn.CrossEntropyLoss(), lambda t: t              # int labels
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss(), lambda t: F.one_hot(t, NUM_CLASSES).float()
    raise ValueError(loss_type)

def train_one_epoch(model, loader, optimizer, criterion, target_adapter, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target_adapter(y))
        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, target_adapter, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_logits, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, target_adapter(y))
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        all_logits.append(logits.cpu()); all_labels.append(y.cpu())
    return (loss_sum / total, correct / total,
            torch.cat(all_logits), torch.cat(all_labels))

def fit(model, train_loader, val_loader, *, loss_type="ce", epochs=20,
        lr=1e-3, wd=0.0, scheduler=None, device=DEVICE,
        param_groups=None, log_every=1, optimizer=None):
    """
    Returns history dict with per-epoch train/val loss, accuracy, and LR.
    Saves the best checkpoint by validation accuracy (never test).
    optimizer: optional — pass pre-built optimizer (needed when using a scheduler).
    """
    torch.cuda.empty_cache() # Clearing the cache
    model=model.to(device) # For not "RuntimeError: Expected all tensors to be on the same device"
    criterion, adapter = make_loss(loss_type)
    
    # FIX: use the pre-built optimizer if given, else build one
    # opt = torch.optim.AdamW(param_groups or model.parameters(), lr=lr, weight_decay=wd)
    if optimizer is None:
        opt = torch.optim.AdamW(param_groups or model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = optimizer
        
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, criterion, adapter, device)
        va_loss, va_acc, *_ = evaluate(model, val_loader, criterion, adapter, device)

        # Scheduler: StepLR uses step(); ReduceLROnPlateau uses step(val_loss)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_loss)
            else:
                scheduler.step()

        current_lr = opt.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(current_lr)

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % log_every == 0:
            print(f"  ep {epoch:02d} | tr_loss {tr_loss:.4f} | "
                  f"va_loss {va_loss:.4f} | va_acc {va_acc:.4f} | lr {current_lr:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_acc"] = best_acc
    return history