"""
Project Configuration
Dataset paths, hyperparameters, and global settings.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_ROOT = PROJECT_ROOT / "data"
MVTEC_AD_PATH = DATA_ROOT / "mvtec_ad"
PREPROCESSED_PATH = DATA_ROOT / "preprocessed"
MODELS_PATH = PROJECT_ROOT / "models"
XAI_OUTPUTS_PATH = PROJECT_ROOT / "xai_outputs"
RESULTS_PATH = PROJECT_ROOT / "results"

# Dataset Configuration
MVTEC_CATEGORIES = ["hazelnut", "screw"]  # Narrowed scope for 21-day deadline
TRAIN_SPLIT_RATIO = 0.7
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

# Model Configuration
RESNET18_INPUT_SIZE = 224
VIT_INPUT_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 3e-4
NUM_EPOCHS = 15
DEVICE = "cpu"  

# XAI Configuration
XAI_METHODS = ["gradcam", "integrated_gradients", "occlusion_sensitivity", "gradientsshap"]

# Binarization: top-% intensity keeps a constant "attention budget" across all methods,
# avoiding confounds from adaptive (Otsu) thresholds behaving differently on small vs. large defects
HEATMAP_BINARIZATION_METHOD = "top_percent"
TOP_PERCENT_THRESHOLD = 0.20  # keep top 20% highest-activation pixels

# Metrics Configuration
DISAGREEMENT_METRIC = "iou"  # "iou" or "spearman"

# Two aggregation strategies, used for different purposes:
# - MEAN: primary metric for severity-correlation analysis (RQ1, RQ2) — standard in literature (Krishna et al.)
# - MAX: reserved for Conflict Detector (Phase 4) — flags worst-case pairwise disagreement
DISAGREEMENT_AGGREGATION_PRIMARY = "mean"
DISAGREEMENT_AGGREGATION_CONFLICT = "max"

# Spearman's rho is computed at the PIXEL level (flattened heatmap correlation), not region-level
SPEARMAN_LEVEL = "pixel"

# Conflict Detector Configuration (Phase 4)
CONFLICT_DETECTOR_PERCENTILE = 75  # Flag top 25% disagreement as conflicted

# Paths to ensure exist
PATHS_TO_CREATE = [
    MVTEC_AD_PATH,
    PREPROCESSED_PATH,
    MODELS_PATH / "resnet18",
    MODELS_PATH / "vit",
    XAI_OUTPUTS_PATH / "gradcam",
    XAI_OUTPUTS_PATH / "integrated_gradients",
    XAI_OUTPUTS_PATH / "occlusion_sensitivity",
    XAI_OUTPUTS_PATH / "gradientsshap",
    RESULTS_PATH / "metrics",
    RESULTS_PATH / "plots",
]

# Create directories at import time
for path in PATHS_TO_CREATE:
    path.mkdir(parents=True, exist_ok=True)
