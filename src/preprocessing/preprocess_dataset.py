import pandas as pd
import numpy as np
import json
import os


# ------------------------------------------------
# Paths
# ------------------------------------------------

LABEL_FILE = "data/raw/hv_double_line_90kv_labels.csv"
DATA_DIR = "data/raw/hv_double_line_90kv_preprocessed_data"
SPLIT_FILE = "data/processed/episode_split.json"
OUTPUT_DIR = "data/processed"


# ------------------------------------------------
# Configuration
# ------------------------------------------------

WINDOW_SIZE = 256


# ------------------------------------------------
# Load metadata and splits
# ------------------------------------------------

labels = pd.read_csv(LABEL_FILE)

with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)


# ------------------------------------------------
# Function to process one episode
# ------------------------------------------------

def create_windows(sample_id):

    metadata = labels[
        labels["sample_id"] == sample_id
    ].iloc[0]

    waveform_file = os.path.join(
        DATA_DIR,
        f"{sample_id}_sample_hv_double_line_90kv.pkl"
    )

    waveform = pd.read_pickle(waveform_file)

    time = waveform["time_s"].values

    signal_columns = [
        column for column in waveform.columns
        if column != "time_s"
    ]

    signals = waveform[signal_columns].values

    fault_start_time = metadata["t_evnt_start"]

    fault_start_index = np.abs(
        time - fault_start_time
    ).argmin()

    if fault_start_index < WINDOW_SIZE:
        return None

    if fault_start_index + WINDOW_SIZE > len(signals):
        return None

    # Pre-fault window
    prefault = signals[
        fault_start_index - WINDOW_SIZE:
        fault_start_index
    ]

    # Fault window
    fault = signals[
        fault_start_index:
        fault_start_index + WINDOW_SIZE
    ]

    # Convert:
    # (time, channels) -> (channels, time)

    prefault = prefault.T.astype(np.float32)
    fault = fault.T.astype(np.float32)

    # Labels
    prefault_detection = 0
    fault_detection = 1

    fault_type = int(metadata["sc_type"])

    return (
        prefault,
        fault,
        prefault_detection,
        fault_detection,
        fault_type
    )


# ------------------------------------------------
# Process a dataset split
# ------------------------------------------------

def process_split(split_name, sample_ids):

    X = []
    detection_labels = []
    fault_types = []
    episode_ids = []

    total = len(sample_ids)

    for i, sample_id in enumerate(sample_ids):

        if (i + 1) % 100 == 0 or i == 0:
            print(
                f"{split_name}: "
                f"{i + 1}/{total}"
            )

        result = create_windows(sample_id)

        if result is None:
            print(
                f"Skipping sample {sample_id}"
            )
            continue

        prefault, fault, prefault_label, fault_label, fault_type = result

        # Pre-fault
        X.append(prefault)
        detection_labels.append(prefault_label)
        fault_types.append(-1)
        episode_ids.append(sample_id)

        # Fault
        X.append(fault)
        detection_labels.append(fault_label)
        fault_types.append(fault_type)
        episode_ids.append(sample_id)

    X = np.stack(X)
    detection_labels = np.array(
        detection_labels,
        dtype=np.int64
    )
    fault_types = np.array(
        fault_types,
        dtype=np.int64
    )
    episode_ids = np.array(
        episode_ids,
        dtype=np.int64
    )

    return X, detection_labels, fault_types, episode_ids


# ------------------------------------------------
# Create output directory
# ------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------
# Process all splits
# ------------------------------------------------

print("\nProcessing training data...")
X_train, y_train_detection, y_train_fault, train_ids = process_split(
    "Train",
    splits["train"]
)

print("\nProcessing validation data...")
X_val, y_val_detection, y_val_fault, val_ids = process_split(
    "Validation",
    splits["validation"]
)

print("\nProcessing test data...")
X_test, y_test_detection, y_test_fault, test_ids = process_split(
    "Test",
    splits["test"]
)


# ------------------------------------------------
# Calculate normalization statistics
# USING TRAINING DATA ONLY
# ------------------------------------------------

print("\nCalculating normalization statistics...")

train_mean = X_train.mean(axis=(0, 2))
train_std = X_train.std(axis=(0, 2))

# Prevent division by zero
train_std[train_std < 1e-8] = 1.0


# ------------------------------------------------
# Normalize
# ------------------------------------------------

print("Normalizing datasets...")

X_train = (
    X_train - train_mean[:, None]
) / train_std[:, None]

X_val = (
    X_val - train_mean[:, None]
) / train_std[:, None]

X_test = (
    X_test - train_mean[:, None]
) / train_std[:, None]


# ------------------------------------------------
# Save datasets
# ------------------------------------------------

print("\nSaving processed datasets...")

np.savez(
    os.path.join(OUTPUT_DIR, "train.npz"),
    X=X_train.astype(np.float32),
    detection_labels=y_train_detection,
    fault_types=y_train_fault,
    episode_ids=train_ids
)

np.savez(
    os.path.join(OUTPUT_DIR, "validation.npz"),
    X=X_val.astype(np.float32),
    detection_labels=y_val_detection,
    fault_types=y_val_fault,
    episode_ids=val_ids
)

np.savez(
    os.path.join(OUTPUT_DIR, "test.npz"),
    X=X_test.astype(np.float32),
    detection_labels=y_test_detection,
    fault_types=y_test_fault,
    episode_ids=test_ids
)


# ------------------------------------------------
# Save normalization statistics
# ------------------------------------------------

np.savez(
    os.path.join(OUTPUT_DIR, "normalization.npz"),
    mean=train_mean,
    std=train_std
)


# ------------------------------------------------
# Final summary
# ------------------------------------------------

print("\n================================")
print("FULL PREPROCESSING COMPLETE")
print("================================")

print("\nTrain:")
print("X:", X_train.shape)

print("\nValidation:")
print("X:", X_val.shape)

print("\nTest:")
print("X:", X_test.shape)

print("\nDetection labels:")
print("Train:", y_train_detection.shape)
print("Validation:", y_val_detection.shape)
print("Test:", y_test_detection.shape)

print("\nFiles saved in:")
print(OUTPUT_DIR)