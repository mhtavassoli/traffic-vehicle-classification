"""evaluate_test_folder.py — Final evaluation on mentor's held-out test data.

CRITICAL RULE: This is the ONLY legitimate use of test data.
It should be run ONCE per model, after all training decisions are frozen.

Multi-root support
------------------
The config exposes:
    TEST_FOLDERS_BY_ROOT : {root_path -> [folder_path, ...]}
    EVAL_OUTPUT_DIRS     : {root_path -> artifacts/eval/<tag>/}
so every testing root (TestingData1, TestingData2, single image, class-folder,
...) is evaluated independently and written to its own output directory.
"""
import os, sys, json, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)

from config import (CLASSES, NUM_CLASSES, DEVICE, ARTIFACTS,
                    TESTING_DATA_ROOT, TEST_FOLDERS, BATCH_SIZE,
                    TEST_FOLDERS_BY_ROOT, EVAL_OUTPUT_DIRS)
from data import IndexedImageDataset, eval_transform
from audit import list_split
from analysis import load_checkpoint


# ============================================================
# Evaluate one folder
# ============================================================
@torch.no_grad()
def evaluate_folder(model, folder_path, class_to_idx):
    """Run inference on all images in folder_path/class/*.jpg."""
    items = list_split(folder_path)

    # Filter: only known classes
    known_items = [(p, c) for p, c in items if c in class_to_idx]
    unknown_items = [(p, c) for p, c in items if c not in class_to_idx]

    if not known_items:
        return None

    ds = IndexedImageDataset(known_items, class_to_idx, eval_transform())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(1)

    y_true = labels.numpy()
    y_pred = preds.numpy()

    acc = accuracy_score(y_true, y_pred)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASSES)), zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Low-confidence analysis
    conf, _ = probs.max(1)
    conf = conf.numpy()
    review_rate = float((conf < 0.70).mean())

    return {
        "folder": os.path.basename(folder_path),
        "folder_path": folder_path,
        "n_images": len(known_items),
        "n_unknown_classes": len(unknown_items),
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f),
        "per_class": {
            CLASSES[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(sup[i]),
            } for i in range(len(CLASSES))
        },
        "lowest_recall_class": CLASSES[int(np.argmin(r))],
        "lowest_precision_class": CLASSES[int(np.argmin(p))],
        "review_rate_at_0.70": review_rate,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "confidences": conf.tolist(),
        "probs": probs.numpy().tolist(),
    }


# ============================================================
# Confusion matrix plot
# ============================================================
def plot_confusion_from_results(y_true, y_pred, save_path, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    if normalize:
        cm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black",
                    fontsize=9)

    fig.colorbar(im, ax=ax)
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return cm


# ============================================================
# Error gallery
# ============================================================
def plot_test_errors(model, folder_path, class_to_idx, save_path, n=12):
    """Show misclassified images."""
    items = list_split(folder_path)
    known_items = [(p, c) for p, c in items if c in class_to_idx]
    ds = IndexedImageDataset(known_items, class_to_idx, eval_transform())
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    model.eval()
    errors = []
    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(DEVICE)
            probs = torch.softmax(model(x_dev), 1)
            conf, pred = probs.max(1)
            for i in range(x.size(0)):
                if pred[i] != y[i] and len(errors) < n:
                    errors.append((x[i].cpu(), y[i].item(),
                                   pred[i].item(), conf[i].item()))

    if not errors:
        print(f"  No errors in {os.path.basename(folder_path)} 🎉")
        return errors

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (img, true, pred, cf) in enumerate(errors):
        ax = axes[idx]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        ax.imshow(np.clip(img_np, 0, 1))
        ax.set_title(f"True: {CLASSES[true]}\nPred: {CLASSES[pred]} ({cf:.2f})",
                     color="red", fontsize=9)
        ax.axis("off")
    for i in range(len(errors), len(axes)):
        axes[i].axis("off")
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return errors


# ============================================================
# Per-root evaluation
# ============================================================
def evaluate_root(model, class_to_idx, root, folders, output_dir):
    """
    Evaluate every test folder that belongs to a single testing root.

    Returns
    -------
    list[dict] : light-weight result dicts (without raw arrays), one per folder.
    """
    print(f"\n{'#' * 70}")
    print(f"# ROOT: {root}")
    print(f"# Output dir: {output_dir}")
    print(f"# Folders: {len(folders)}")
    print(f"{'#' * 70}")

    os.makedirs(output_dir, exist_ok=True)
    root_results = []

    for folder in folders:
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {os.path.basename(folder)}")
        print("=" * 70)

        result = evaluate_folder(model, folder, class_to_idx)
        if result is None:
            print(f"  ⚠ Skipped (no known-class images)")
            continue

        print(f"  Images: {result['n_images']}")
        print(f"  Accuracy:        {result['accuracy']:.4f}")
        print(f"  Macro Precision: {result['macro_precision']:.4f}")
        print(f"  Macro Recall:    {result['macro_recall']:.4f}")
        print(f"  Macro F1:        {result['macro_f1']:.4f}")
        print(f"  Lowest recall:   {result['lowest_recall_class']}")
        print(f"  Lowest precision: {result['lowest_precision_class']}")
        print(f"  Review rate (conf<0.70): {result['review_rate_at_0.70']:.3f}")

        # Confusion matrix
        cm_path = os.path.join(output_dir,
                               f"{result['folder']}_confusion.png")
        plot_confusion_from_results(
            result["y_true"], result["y_pred"], cm_path
        )
        print(f"  ✓ Saved confusion matrix: {cm_path}")

        # Error gallery
        err_path = os.path.join(output_dir,
                                f"{result['folder']}_errors.png")
        errs = plot_test_errors(model, folder, class_to_idx, err_path, n=12)
        if errs:
            print(f"  ✓ Saved {len(errs)} errors: {err_path}")

        # Remove heavy fields before saving JSON
        result_light = {k: v for k, v in result.items()
                        if k not in ("y_true", "y_pred", "confidences", "probs")}
        root_results.append(result_light)

        # Save per-folder JSON inside the root's own output dir
        with open(os.path.join(output_dir,
                               f"{result['folder']}_report.json"), "w") as f:
            json.dump(result_light, f, indent=2)

    # Per-root summary
    if root_results:
        print(f"\n{'-' * 70}")
        print(f"AGGREGATE SUMMARY — {os.path.basename(root)}")
        print(f"{'-' * 70}")

        header = f"{'Folder':<12} {'N':>5} {'Acc':>7} {'MacroP':>8} {'MacroR':>8} {'MacroF1':>9}"
        print(header)
        print("-" * len(header))
        for r in root_results:
            print(f"{r['folder']:<12} {r['n_images']:>5} "
                  f"{r['accuracy']:>7.4f} {r['macro_precision']:>8.4f} "
                  f"{r['macro_recall']:>8.4f} {r['macro_f1']:>9.4f}")

        avg_acc = float(np.mean([r["accuracy"] for r in root_results]))
        avg_f1 = float(np.mean([r["macro_f1"] for r in root_results]))
        print("-" * len(header))
        print(f"{'AVERAGE':<12} {'':>5} {avg_acc:>7.4f} {'':>8} {'':>8} {avg_f1:>9.4f}")

        summary = {
            "root": root,
            "root_tag": os.path.basename(os.path.normpath(root)),
            "folders_evaluated": len(root_results),
            "average_accuracy": avg_acc,
            "average_macro_f1": avg_f1,
            "per_folder": root_results,
        }
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Root summary saved to {output_dir}/summary.json")

    return root_results


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet_ft_v2",
                        help="Checkpoint name in artifacts/")
    parser.add_argument("--arch", default="resnet",
                        choices=["cnn", "resnet"])
    parser.add_argument("--folder", default=None,
                        help="Specific test folder. If set, evaluate only it.")
    parser.add_argument("--root", default=None,
                        help="Restrict evaluation to one testing root.")
    args = parser.parse_args()

    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model} ({args.arch})")

    # Load model once — shared across every testing root
    model, ckpt = load_checkpoint(args.model, args.arch)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    print(f"✓ Loaded model from {args.model}.pt")

    # -------- Single-folder mode (debug / targeted evaluation) --------
    if args.folder:
        # Determine which root this folder belongs to (for output dir)
        folder_abs = os.path.abspath(args.folder)
        owner_root = None
        for root in TEST_FOLDERS_BY_ROOT:
            if folder_abs.startswith(os.path.abspath(root)):
                owner_root = root
                break
        if owner_root is None:
            owner_root = TESTING_DATA_ROOT  # fall back to default root tag

        # Prefer pre-computed dir, else build one from config helper
        output_dir = EVAL_OUTPUT_DIRS.get(owner_root)
        if output_dir is None:
            try:
                from config import build_output_dir
                output_dir = build_output_dir(owner_root)
            except Exception:
                output_dir = os.path.join(ARTIFACTS, "eval",
                                          os.path.basename(os.path.normpath(owner_root)))
                os.makedirs(output_dir, exist_ok=True)

        evaluate_root(model, class_to_idx, owner_root,
                      [folder_abs], output_dir)
        return

    # -------- Multi-root mode (default) --------
    # Build the {root -> folders} mapping. If the new config is empty
    # (e.g. an older config.py), fall back to the legacy single-root form.
    if TEST_FOLDERS_BY_ROOT:
        roots_to_eval = TEST_FOLDERS_BY_ROOT
    else:
        # Legacy fallback: only TESTING_DATA_ROOT / TEST_FOLDERS available
        roots_to_eval = {TESTING_DATA_ROOT: TEST_FOLDERS}

    # Optional restriction via --root
    if args.root:
        wanted = os.path.abspath(args.root)
        roots_to_eval = {
            r: f for r, f in roots_to_eval.items()
            if os.path.abspath(r) == wanted
        }
        if not roots_to_eval:
            print(f"✗ ERROR: --root {args.root} not found in TEST_FOLDERS_BY_ROOT")
            sys.exit(1)

    # Show the plan
    print(f"\nTesting roots to evaluate: {len(roots_to_eval)}")
    for root, folders in roots_to_eval.items():
        tag = os.path.basename(os.path.normpath(root))
        print(f"  • {tag:<20} ({len(folders)} folder(s))  <- {root}")

    # Evaluate each root into its own output directory
    all_roots_summary = {}
    for root, folders in roots_to_eval.items():
        # Resolve output dir: prefer pre-computed, else build on the fly
        output_dir = EVAL_OUTPUT_DIRS.get(root)
        if output_dir is None:
            try:
                from config import build_output_dir
                output_dir = build_output_dir(root)
            except Exception:
                output_dir = os.path.join(ARTIFACTS, "eval",
                                          os.path.basename(os.path.normpath(root)))
                os.makedirs(output_dir, exist_ok=True)

        if not folders:
            # Fallback: look for subfolders containing known classes
            if os.path.isdir(root):
                folders = [
                    os.path.join(root, n)
                    for n in sorted(os.listdir(root))
                    if os.path.isdir(os.path.join(root, n))
                ]
            if not folders:
                print(f"\n⚠ Skipping root {root} (no folders found)")
                continue

        root_results = evaluate_root(
            model, class_to_idx, root, folders, output_dir
        )
        if root_results:
            all_roots_summary[root] = {
                "tag": os.path.basename(os.path.normpath(root)),
                "output_dir": output_dir,
                "folders_evaluated": len(root_results),
                "average_accuracy": float(np.mean(
                    [r["accuracy"] for r in root_results])),
                "average_macro_f1": float(np.mean(
                    [r["macro_f1"] for r in root_results])),
            }

    # -------- Cross-root comparison summary --------
    if len(all_roots_summary) > 1:
        compare_dir = os.path.join(ARTIFACTS, "eval")
        os.makedirs(compare_dir, exist_ok=True)
        compare_path = os.path.join(compare_dir, "summary_all_roots.json")

        print(f"\n{'=' * 70}")
        print("CROSS-ROOT COMPARISON")
        print("=" * 70)
        header = f"{'Root':<22} {'#Folders':>9} {'AvgAcc':>9} {'AvgF1':>9}"
        print(header)
        print("-" * len(header))
        for root, s in all_roots_summary.items():
            print(f"{s['tag']:<22} {s['folders_evaluated']:>9} "
                  f"{s['average_accuracy']:>9.4f} {s['average_macro_f1']:>9.4f}")

        with open(compare_path, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "arch": args.arch,
                    "roots": all_roots_summary,
                },
                f, indent=2
            )
        print(f"\n✓ Cross-root summary saved to {compare_path}")

    print(f"\n{'=' * 70}")
    print("✓ FINAL TEST EVALUATION COMPLETE")
    print("=" * 70)
    print("\n⚠️  REMEMBER: This was the ONE legitimate use of test data.")
    print("   Do NOT tune the model based on these results.")
    print("   Any further changes require a NEW held-out test set.")


if __name__ == "__main__":
    main()