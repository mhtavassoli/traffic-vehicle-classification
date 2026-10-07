"""Orchestrator: runs the 8 required controlled ablations and logs a table."""
import json, os, numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader
from config import *  # noqa
from audit import run_audit, build_cleaned_index
from data import (IndexedImageDataset, train_transform, eval_transform,
                  stratified_val_split, simulate_imbalance,
                  BalancedBatchSampler)
from models import TrafficCNN, build_resnet18, resnet_param_groups
from train import fit, evaluate, make_loss
from evaluate import full_report, plot_confusion, rank_confused_pairs

def seed_everything(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_loaders(use_aug=True, balanced=False, simulate_imbalance_cfg=None):
    """Returns train_loader, val_loader for one experiment."""
    # 1) Audit + clean
    report, dups, conflicts = run_audit()
    per_split = {s: [] for s in ("train", "test", "unclean")}
    # NOTE: reuse functions to reconstruct items; simplified here.
    from audit import list_split
    per_split = {"train": list_split(TRAIN_DIR),
                 "test":  list_split(TEST_DIR),
                 "unclean": list_split(UNCLEAN_DIR)}
    cleaned, _ = build_cleaned_index(dups, per_split)

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    tr_items = cleaned["train"]
    if simulate_imbalance_cfg:
        tr_items = simulate_imbalance(tr_items, class_to_idx, simulate_imbalance_cfg)
    tr_items, va_items, _, _ = stratified_val_split(tr_items, class_to_idx, VAL_FRACTION)

    ds_tr = IndexedImageDataset(tr_items, class_to_idx, train_transform(use_aug))
    ds_va = IndexedImageDataset(va_items, class_to_idx, eval_transform())

    if balanced:
        targets = [class_to_idx[c] for _, c in tr_items]
        bs = BalancedBatchSampler(targets, BATCH_SIZE, NUM_CLASSES)
        tr_loader = DataLoader(ds_tr, batch_sampler=bs, num_workers=NUM_WORKERS)
    else:
        tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS)
    return tr_loader, va_loader, class_to_idx, cleaned

def run_experiment(name, *, use_aug=True, dropout=0.3, pool_type="max",
                   wd=0.0, scheduler="none", loss_type="ce", balanced=False,
                   arch="cnn", resnet_mode="feature_extract", epochs=EPOCHS):
    seed_everything()
    tr_loader, va_loader, c2i, cleaned = build_loaders(
        use_aug=use_aug, balanced=balanced)

    if arch == "cnn":
        model = TrafficCNN(NUM_CLASSES, dropout=dropout, pool_type=pool_type).to(DEVICE)
        groups = None
    else:
        model = build_resnet18(resnet_mode, NUM_CLASSES).to(DEVICE)
        groups = resnet_param_groups(model, lr_head=LR, lr_backbone=LR / 10, wd=wd)

    # Build optimizer ONCE, share between scheduler and fit
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)

    sched = None
    if scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    hist = fit(model, tr_loader, va_loader, loss_type=loss_type, epochs=epochs,
               lr=LR, wd=wd, scheduler=sched, param_groups=groups, optimizer=opt)

    # Evaluate on val only (test frozen until final choice)
    _, _, logits, labels = evaluate(
        model, va_loader, *make_loss(loss_type), DEVICE)
    y_pred = logits.argmax(1).numpy(); y_true = labels.numpy()
    rep = full_report(y_true, y_pred)
    rep["history"] = hist
    rep["name"] = name
    with open(os.path.join(ARTIFACTS, f"{name}.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)
    torch.save({"state_dict": model.state_dict(),
                "class_to_idx": c2i,
                "config": {"name": name, "loss": loss_type, "arch": arch}},
               os.path.join(ARTIFACTS, f"{name}.pt"))
    return rep

if __name__ == "__main__":
    experiments = [
        ("cnn_baseline",        dict()),
        ("cnn_no_aug",          dict(use_aug=False)),
        ("cnn_dropout_0",       dict(dropout=0.0)),
        ("cnn_dropout_05",      dict(dropout=0.5)),
        ("cnn_avgpool",         dict(pool_type="avg")),
        ("cnn_wd_1e4",          dict(wd=1e-4)),
        ("cnn_steplr",          dict(scheduler="step")),
        ("cnn_plateau",         dict(scheduler="plateau")),
        ("cnn_balanced",        dict(balanced=True)),
        ("cnn_bce",             dict(loss_type="bce")),
        ("resnet_feat",         dict(arch="resnet", resnet_mode="feature_extract")),
        ("resnet_ft",           dict(arch="resnet", resnet_mode="fine_tune")),
    ]
    rows = []
    for name, kw in experiments:
        print(f"\n>>> {name}")
        r = run_experiment(name, **kw)
        rows.append({
            "experiment": name,
            "macro_precision": r["macro_precision"],
            "macro_recall": r["macro_recall"],
            "macro_f1": r["macro_f1"],
            "lowest_recall_class": r["lowest_recall_class"],
            "lowest_precision_class": r["lowest_precision_class"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(ARTIFACTS, "summary.csv"), index=False)