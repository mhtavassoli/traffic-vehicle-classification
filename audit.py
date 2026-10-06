"""Data-quality audit and duplicate detection via image-content hashes.

Order of operations (critical for preventing leakage):
    1. scan every split, verify readability & format
    2. report class counts, size distributions
    3. hash every image, detect duplicates ACROSS splits
    4. produce a cleaned index (list of allowed paths + labels)
"""
import os, hashlib, json
from collections import defaultdict, Counter
from PIL import Image, UnidentifiedImageError
import numpy as np

from config import TRAIN_DIR, TEST_DIR, UNCLEAN_DIR, CLASSES

def list_split(split_dir):
    """Return [(path, class_name)] for every image file under split_dir/class/."""
    items = []
    if not os.path.isdir(split_dir):
        return items
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            items.append((os.path.join(cls_dir, fname), cls))
    return items

def load_all_train_items():
    """Merge V1 and V2 train splits when USE_DATASET_V2 is True."""
    from config import TRAIN_DIR, TRAIN_DIR_V2, USE_DATASET_V2
    items = list_split(TRAIN_DIR)
    if USE_DATASET_V2:
        items += list_split(TRAIN_DIR_V2)
    return items

def audit_readability(items):
    """Return (ok_items, unreadable, tiny, size_hist)."""
    ok, bad, tiny = [], [], []
    sizes = Counter()
    for path, cls in items:
        try:
            with Image.open(path) as im:
                im.verify()          # cheap integrity check
            with Image.open(path) as im:
                w, h = im.size
            if w < 32 or h < 32:
                tiny.append((path, cls, (w, h)))
            else:
                ok.append((path, cls))
                sizes[(w, h)] += 1
        except (UnidentifiedImageError, OSError):
            bad.append((path, cls))
    return ok, bad, tiny, sizes

def image_hash(path, size=(64, 64)):
    """Content hash of the resized RGB pixels (robust to metadata/compression)."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize(size)
        return hashlib.md5(im.tobytes()).hexdigest()

def find_duplicates(per_split_items):
    """per_split_items: {split_name: [(path, cls), ...]}"""
    hash_to_locs = defaultdict(list)   # hash -> [(split, path, cls)]
    for split, items in per_split_items.items():
        for path, cls in items:
            hash_to_locs[image_hash(path)].append((split, path, cls))
    # Duplicates inside a split AND across splits.
    dups = {h: locs for h, locs in hash_to_locs.items() if len(locs) > 1}
    conflicts = []  # same image, different class labels
    for h, locs in dups.items():
        if len({c for _, _, c in locs}) > 1:
            conflicts.append((h, locs))
    return dups, conflicts

def run_audit(report_path=os.path.join("artifacts", "audit.json")):
    per_split = {
        "train":   list_split(TRAIN_DIR),
        "test":    list_split(TEST_DIR),
        "unclean": list_split(UNCLEAN_DIR),
    }
    report = {"counts": {}, "unreadable": {}, "tiny": {}, "size_top": {},
              "duplicates": {}, "label_conflicts": []}

    clean = {}
    for split, items in per_split.items():
        ok, bad, tiny, sizes = audit_readability(items)
        report["counts"][split] = dict(Counter(c for _, c in ok))
        report["unreadable"][split] = bad
        report["tiny"][split] = tiny
        report["size_top"][split] = sizes.most_common(5)
        clean[split] = ok

    dups, conflicts = find_duplicates(clean)
    report["duplicates"]["total_groups"] = len(dups)
    report["duplicates"]["cross_split_groups"] = sum(
        1 for locs in dups.values() if len({s for s, _, _ in locs}) > 1
    )
    report["label_conflicts"] = [(h, locs) for h, locs in conflicts]

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report, dups, conflicts

# ---------- Cleaning policy ----------
def build_cleaned_index(dups, per_split):
    """
    Cleaning rule (documented decision):
      - If a hash appears in BOTH train and test -> drop from train (test stays frozen).
      - If a hash appears in unclean AND train/test -> drop from unclean
        (unclean is only for OOD/novelty analysis).
      - Within a single split, keep the first occurrence, drop the rest.
    """
    drop = set()
    for h, locs in dups.items():
        splits = {s for s, _, _ in locs}
        for s, p, _ in locs:
            if "test" in splits and s == "train":
                drop.add(p)
            elif "unclean" in splits and s in ("train", "test"):
                drop.add(p)
            elif len(locs) > 1 and s == "unclean":
                # keep at most one copy inside unclean
                drop.add(p)
    cleaned = {s: [(p, c) for (p, c) in items if p not in drop]
               for s, items in per_split.items()}
    return cleaned, drop