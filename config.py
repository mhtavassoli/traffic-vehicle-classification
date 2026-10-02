"""Central configuration: single source of truth for reproducibility."""
import os
from dataclasses import dataclass, field
from typing import List, Tuple

SEED = 42
DATA_ROOT = "dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR  = os.path.join(DATA_ROOT, "test")
UNCLEAN_DIR = os.path.join(DATA_ROOT, "unclean")

CLASSES = ["ambulance", "autobus", "kamyun", "kamyunet",
           "minibus", "savari", "taxi", "vanet"]
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 128
VAL_FRACTION = 0.20
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 0.0
DROPOUT = 0.30
NUM_WORKERS = 4
CONFIDENCE_THRESHOLD = 0.70

# ImageNet statistics — required whenever we use pretrained ResNet18
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

ARTIFACTS = "artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)