"""Central configuration: single source of truth for reproducibility."""
import os
from dataclasses import dataclass, field
from typing import List, Tuple
import torch
from dotenv import load_dotenv

load_dotenv()  # Load .env file if exists

# ============ Data Paths (confidential, from environment) ============
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_V1 = os.getenv("DATA_ROOT_V1", os.path.join(_HERE, "..", "data"))
_DATA_V2 = os.getenv("DATA_ROOT_V2", os.path.join(_HERE, "..", "datasetv2_TrainUclean"))

DATA_ROOT_V1 = os.path.abspath(os.path.join(_HERE, _DATA_V1))
DATA_ROOT_V2 = os.path.abspath(os.path.join(_HERE, _DATA_V2))

TRAIN_DIR   = os.path.join(DATA_ROOT_V1, "train")
TEST_DIR    = os.path.join(DATA_ROOT_V1, "test")
UNCLEAN_DIR = os.path.join(DATA_ROOT_V1, "unclean")

# Supplementary dataset (V2): to be MERGED into V1, not used independently
TRAIN_DIR_V2   = os.path.join(DATA_ROOT_V2, "train")
UNCLEAN_DIR_V2 = os.path.join(DATA_ROOT_V2, "unclean")

# ============ Flag to enable V2 ============
USE_DATASET_V2 = os.getenv("USE_DATASET_V2", "false").lower() == "true"

# =========================================================================
SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # The config.py folder itself
PROJECT_ROOT = os.path.dirname(BASE_DIR) # We move up one level to reach HW12-0-Project-Vehicle.
# DATA_ROOT = "dataset"
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR  = os.path.join(DATA_ROOT, "test")
UNCLEAN_DIR = os.path.join(DATA_ROOT, "unclean")

CLASSES = ["ambulance", "autobus", "kamyun", "kamyunet",
           "minibus", "savari", "taxi", "vanet"]
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 128
VAL_FRACTION = 0.20
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 0.0
DROPOUT = 0.30
NUM_WORKERS = 4
CONFIDENCE_THRESHOLD = 0.70

# ImageNet statistics — required whenever we use pretrained ResNet18
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

ARTIFACTS = "artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)