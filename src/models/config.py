"""
Centralized configuration for the TCN fault detection project.

All hyperparameters and paths are defined here so they can be
easily modified without changing model or training code.
"""


# ------------------------------------------------
# Data paths
# ------------------------------------------------

DATA_DIR = "data/processed"
RAW_DATA_DIR = "data/raw"
MODEL_DIR = "models"
RESULTS_DIR = "results"


# ------------------------------------------------
# Dataset
# ------------------------------------------------

WINDOW_SIZE = 256
INPUT_CHANNELS = 48
NUM_FAULT_CLASSES = 4

FAULT_TYPE_NAMES = {
    0: "Single Line-to-Ground (SLG)",
    1: "Line-to-Line (LL)",
    2: "Double Line-to-Ground (DLG)",
    3: "Three-Phase (LLL/LLLG)"
}


# ------------------------------------------------
# TCN architecture — configurable hyperparameters
# ------------------------------------------------

TCN_CHANNELS = [64, 64, 128, 128]
TCN_KERNEL_SIZE = 7
TCN_DROPOUT = 0.2


# ------------------------------------------------
# Training
# ------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# Learning rate scheduler
LR_SCHEDULER = "cosine"
LR_MIN = 1e-6
