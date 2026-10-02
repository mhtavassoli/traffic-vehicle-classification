"""Datasets, transforms, stratified split, and BalancedBatchSampler."""
import os, numpy as np, torch
from torch.utils.data import Dataset, Sampler, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from config import (IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, SEED,
                    NUM_CLASSES, CLASSES)

# ---------- Transforms ----------
def train_transform(use_aug=True):
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if use_aug:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        ]
    ops += [transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)

def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# ---------- Dataset ----------
class IndexedImageDataset(Dataset):
    """A tiny Dataset driven by (path, class) pairs — avoids ImageFolder pitfalls."""
    def __init__(self, items, class_to_idx, transform=None):
        self.items = items
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, cls = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.class_to_idx[cls]

# ---------- Stratified split ----------
def stratified_val_split(items, class_to_idx, val_fraction=0.2, seed=SEED):
    labels = np.array([class_to_idx[c] for _, c in items])
    idx = np.arange(len(items))
    tr_idx, va_idx = train_test_split(idx, test_size=val_fraction,
                                      stratify=labels, random_state=seed)
    tr = [items[i] for i in tr_idx]
    va = [items[i] for i in va_idx]
    return tr, va, tr_idx.tolist(), va_idx.tolist()

# ---------- Simulated imbalance ----------
def simulate_imbalance(items, class_to_idx, keep_fraction, seed=SEED):
    """keep_fraction: dict {class_name: fraction_to_keep}. Returns filtered items."""
    rng = np.random.default_rng(seed)
    kept = []
    for cls in CLASSES:
        cls_items = [it for it in items if it[1] == cls]
        f = keep_fraction.get(cls, 1.0)
        n_keep = max(1, int(len(cls_items) * f))
        idx = rng.permutation(len(cls_items))[:n_keep]
        kept.extend([cls_items[i] for i in idx])
    return kept

# ---------- BalancedBatchSampler ----------
class BalancedBatchSampler(Sampler):
    """
    Each batch contains exactly batch_size // num_classes samples per class.
    Smaller classes are oversampled with replacement.
    """
    def __init__(self, targets, batch_size, num_classes=NUM_CLASSES, seed=SEED):
        assert batch_size % num_classes == 0, "batch_size must be divisible by num_classes"
        self.targets = np.asarray(targets)
        self.per_class = batch_size // num_classes
        self.num_classes = num_classes
        self.class_indices = [np.where(self.targets == c)[0] for c in range(num_classes)]
        self.num_batches = max(len(idx) for idx in self.class_indices) // self.per_class
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch): self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_batches):
            batch = []
            for c in range(self.num_classes):
                idx = self.class_indices[c]
                replace = len(idx) < self.per_class
                chosen = rng.choice(idx, size=self.per_class, replace=replace)
                batch.extend(chosen.tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self): return self.num_batches