"""
Central configuration: single source of truth for reproducibility.

All input paths are resolved from environment variables (see .env) so the
same code base can be executed against different dataset / testing-data
combinations without editing any source file.

Path resolution
---------------
Relative paths in .env (e.g. "../dataset", "../TestingData1") are resolved
against BASE_DIR — the folder that contains this config.py, i.e. the
`traffic-vehicle-classification` code folder. This makes
    ../dataset  ->  <...>/HW12-0-Project-Vehicle/dataset
    ../TestingData1 ->  <...>/HW12-0-Project-Vehicle/TestingData1
which matches the on-disk layout used in this repository.

Supported scenarios
-------------------
1. Baseline run            : USE_DATASET_V2=false, TESTING_DATA_ROOTS=../TestingData1
2. Augmented run           : USE_DATASET_V2=true,  TESTING_DATA_ROOTS=../TestingData1,../TestingData2
3. Single-image test       : TEST_IMAGE_PATH=../samples/my_car.jpg
4. Single class-folder test: TEST_CLASS_FOLDER=../TestingData3
5. Cross-test comparison   : keep the trained model fixed and only flip
                             TESTING_DATA_ROOTS between TestingData1 / TestingData2.
"""
import os
import shutil
from typing import Dict, List, Tuple

import torch
from dotenv import load_dotenv

load_dotenv()  # Load .env file if present

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = _HERE                              # folder that holds config.py
PROJECT_ROOT = os.path.dirname(BASE_DIR)      # HW12-0-Project-Vehicle folder


def _resolve(path: str, default_rel: str) -> str:
    """
    Resolve `path` (absolute, or relative to BASE_DIR — the code folder
    that contains this config.py).

    Examples (with BASE_DIR = .../HW12-0-Project-Vehicle/traffic-vehicle-classification)
    ---------------------------------------------------------------------------------
    "../dataset"            -> .../HW12-0-Project-Vehicle/dataset
    "../TestingData1"       -> .../HW12-0-Project-Vehicle/TestingData1
    "E:/abs/path/data"      -> E:/abs/path/data   (unchanged)
    """
    if not path:
        path = default_rel
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(BASE_DIR, path))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ===========================================================================
# Dataset roots
# ===========================================================================
DATA_ROOT_V1 = _resolve(os.getenv("DATA_ROOT_V1"), "dataset")
DATA_ROOT_V2 = _resolve(os.getenv("DATA_ROOT_V2"), "datasetSupplementary")

USE_DATASET_V2 = _env_bool("USE_DATASET_V2", False)


# ===========================================================================
# Optional V1 + V2 merging
# ===========================================================================
def _link_or_copy(src: str, dst: str) -> None:
    """Create dst -> src using symlink > hardlink > copy (cross-platform)."""
    if os.path.exists(dst) or os.path.islink(dst):
        return
    try:
        os.symlink(src, dst)                 # Linux / macOS
    except (OSError, NotImplementedError):
        try:
            os.link(src, dst)                # Windows NTFS same-volume
        except OSError:
            shutil.copy2(src, dst)           # last resort


def _merge_split(sources: List[Tuple[str, str]], out_split: str) -> None:
    """
    Merge multiple class-organised split folders into `out_split`.

    sources : list of (src_root, filename_prefix)
              e.g. [(DATA_ROOT_V1/train, ""), (DATA_ROOT_V2/train, "v2_")]
    """
    os.makedirs(out_split, exist_ok=True)
    for src_root, tag in sources:
        if not os.path.isdir(src_root):
            continue
        for cls in sorted(os.listdir(src_root)):
            src_cls = os.path.join(src_root, cls)
            if not os.path.isdir(src_cls):
                continue
            dst_cls = os.path.join(out_split, cls)
            os.makedirs(dst_cls, exist_ok=True)
            for fname in os.listdir(src_cls):
                src_file = os.path.join(src_cls, fname)
                if not os.path.isfile(src_file):
                    continue
                dst_file = os.path.join(dst_cls, f"{tag}{fname}")
                _link_or_copy(src_file, dst_file)


def _build_merged_root() -> str:
    """
    Physically merge V1 + V2 (train & unclean) into `dataset_merged/`
    next to the code folder. The test split is taken from V1 only because
    V2 does not ship one. The folder is rebuilt on every interpreter start
    so additions / deletions in the source folders are always reflected.
    """
    merged = os.path.join(PROJECT_ROOT, "dataset_merged")
    if os.path.isdir(merged):
        shutil.rmtree(merged)
    os.makedirs(merged, exist_ok=True)

    _merge_split(
        [(os.path.join(DATA_ROOT_V1, "train"),   ""),
         (os.path.join(DATA_ROOT_V2, "train"),   "v2_")],
        os.path.join(merged, "train"),
    )
    _merge_split(
        [(os.path.join(DATA_ROOT_V1, "unclean"), ""),
         (os.path.join(DATA_ROOT_V2, "unclean"), "v2_")],
        os.path.join(merged, "unclean"),
    )
    _merge_split(
        [(os.path.join(DATA_ROOT_V1, "test"), "")],
        os.path.join(merged, "test"),
    )
    return merged


# ===========================================================================
# Active dataset — automatically switches to merged root when V2 is enabled
# ===========================================================================
DATA_ROOT = _build_merged_root() if USE_DATASET_V2 else DATA_ROOT_V1

TRAIN_DIR   = os.path.join(DATA_ROOT, "train")
TEST_DIR    = os.path.join(DATA_ROOT, "test")
UNCLEAN_DIR = os.path.join(DATA_ROOT, "unclean")

# Raw V2 paths kept for convenience / backward compatibility
TRAIN_DIR_V2   = os.path.join(DATA_ROOT_V2, "train")
UNCLEAN_DIR_V2 = os.path.join(DATA_ROOT_V2, "unclean")


# ===========================================================================
# Training hyper-parameters
# ===========================================================================
SEED = 42
CLASSES = ["ambulance", "autobus", "kamyun", "kamyunet",
           "minibus", "savari", "taxi", "vanet"]
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 128
VAL_FRACTION = 0.20
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 0.0
DROPOUT = 0.30
NUM_WORKERS = 2
CONFIDENCE_THRESHOLD = 0.70
PIN_MEMORY = True

# ImageNet statistics — required whenever we use pretrained ResNet18
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARTIFACTS = "artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)


# ===========================================================================
# Testing targets — three independent kinds, all can coexist
# ===========================================================================
def _split_env_paths(raw: str) -> List[str]:
    """Split a comma-separated env var into a clean list of strings."""
    return [p.strip() for p in raw.split(",") if p.strip()]


# (A) Folder-style roots (test0/, test1/, ... inside each).
_raw_roots = os.getenv("TESTING_DATA_ROOTS", "").strip()
if _raw_roots:
    TESTING_DATA_ROOTS: List[str] = [
        _resolve(p, p) for p in _split_env_paths(_raw_roots)
    ]
else:
    # Legacy fallback: single TESTING_DATA_ROOT
    _legacy = os.getenv("TESTING_DATA_ROOT", "").strip()
    TESTING_DATA_ROOTS = [_resolve(_legacy, "TestingData1")]

# Convenience alias (first root) — keeps old scripts working unchanged
TESTING_DATA_ROOT = TESTING_DATA_ROOTS[0]
TESTING_DATA_TAG  = os.path.basename(os.path.normpath(TESTING_DATA_ROOT))


# (B) Single-image test (one-sample test set)
_raw_img = os.getenv("TEST_IMAGE_PATH", "").strip()
TEST_IMAGE_PATH = _resolve(_raw_img, _raw_img) if _raw_img else None


# (C) Single class-organised folder (ambulance/, taxi/, ...)
_raw_cls = os.getenv("TEST_CLASS_FOLDER", "").strip()
TEST_CLASS_FOLDER = _resolve(_raw_cls, _raw_cls) if _raw_cls else None


def discover_test_folders(root: str) -> List[str]:
    """Find all subdirectories that contain class sub-folders (test0, test1, ...)."""
    if not os.path.isdir(root):
        return []
    folders: List[str] = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            subs = os.listdir(full)
            if any(s in CLASSES for s in subs):
                folders.append(full)
    return folders


# Mapping root -> folders, in case you want to iterate over every testing set
TEST_FOLDERS_BY_ROOT: Dict[str, List[str]] = {
    r: discover_test_folders(r) for r in TESTING_DATA_ROOTS
}

# Flattened list — the evaluation script can iterate over it directly
TEST_FOLDERS: List[str] = [
    f for folders in TEST_FOLDERS_BY_ROOT.values() for f in folders
]


# ===========================================================================
# Independent output layout — one sub-folder per testing target
# ===========================================================================
def target_tag(path: str | None) -> str:
    """
    Stable, filesystem-safe tag for a testing target.

    Examples
    --------
    ../TestingData1           -> "TestingData1"
    ../TestingData2           -> "TestingData2"
    ../samples/my_car.jpg     -> "my_car"
    ../TestingData3           -> "TestingData3"
    """
    if not path:
        return "unknown"
    return os.path.splitext(os.path.basename(os.path.normpath(path)))[0]


def build_output_dir(target_path: str) -> str:
    """
    Each testing target gets its own output directory:
        artifacts/eval/<tag>/ ...

    This guarantees TestingData1 / TestingData2 / single image / class-folder
    results never overwrite each other and can be compared side by side.
    """
    out_dir = os.path.join(ARTIFACTS, "eval", target_tag(target_path))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# Pre-computed output directories, keyed by the original target path.
EVAL_OUTPUT_DIRS: Dict[str, str] = {}
for _r in TESTING_DATA_ROOTS:
    EVAL_OUTPUT_DIRS[_r] = build_output_dir(_r)
if TEST_IMAGE_PATH:
    EVAL_OUTPUT_DIRS[TEST_IMAGE_PATH] = build_output_dir(TEST_IMAGE_PATH)
if TEST_CLASS_FOLDER:
    EVAL_OUTPUT_DIRS[TEST_CLASS_FOLDER] = build_output_dir(TEST_CLASS_FOLDER)