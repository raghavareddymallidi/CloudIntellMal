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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_windows.npz")


# ------------------------------------------------
# Configuration
# ------------------------------------------------

WINDOW_SIZE = 256
NUM_EPISODES = 10


# ------------------------------------------------
# Load labels and split
# ------------------------------------------------

labels = pd.read_csv(LABEL_FILE)

with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)

train_ids = splits["train"]

# Test only the first 10 training episodes
sample_ids = train_ids[:NUM_EPISODES]

print("Testing with", len(sample_ids), "episodes")


# ------------------------------------------------
# Storage
# ------------------------------------------------

X = []
detection_labels = []
fault_types = []
sample_ids_used = []


# ------------------------------------------------
# Process episodes
# ------------------------------------------------

for i, sample_id in enumerate(sample_ids):

    print(
        f"Processing {i + 1}/{len(sample_ids)} - "
        f"sample {sample_id}"
    )

    # Get metadata
    metadata = labels[
        labels["sample_id"] == sample_id
    ].iloc[0]

    # Waveform path
    waveform_file = os.path.join(
        DATA_DIR,
        f"{sample_id}_sample_hv_double_line_90kv.pkl"
    )

    # Load waveform
    waveform = pd.read_pickle(waveform_file)

    # Time
    time = waveform["time_s"].values

    # Select all signal channels
    signal_columns = [
        column for column in waveform.columns
        if column != "time_s"
    ]

    signals = waveform[signal_columns].values

    # Find fault start
    fault_start_time = metadata["t_evnt_start"]

    fault_start_index = np.abs(
        time - fault_start_time
    ).argmin()

    # Make sure enough samples exist before fault
    if fault_start_index < WINDOW_SIZE:
        print("Skipping: not enough pre-fault samples")
        continue

    # Make sure enough samples exist after fault start
    if fault_start_index + WINDOW_SIZE > len(signals):
        print("Skipping: not enough fault samples")
        continue

    # ------------------------------------------------
    # Pre-fault window
    # ------------------------------------------------

    prefault = signals[
        fault_start_index - WINDOW_SIZE:
        fault_start_index
    ]

    # ------------------------------------------------
    # Fault window
    # ------------------------------------------------

    fault = signals[
        fault_start_index:
        fault_start_index + WINDOW_SIZE
    ]

    # ------------------------------------------------
    # Convert to TCN format
    # (time, channels) -> (channels, time)
    # ------------------------------------------------

    prefault = prefault.T
    fault = fault.T

    # Add pre-fault sample
    X.append(prefault)
    detection_labels.append(0)
    fault_types.append(-1)
    sample_ids_used.append(sample_id)

    # Add fault sample
    X.append(fault)
    detection_labels.append(1)
    fault_types.append(int(metadata["sc_type"]))
    sample_ids_used.append(sample_id)


# ------------------------------------------------
# Convert to NumPy arrays
# ------------------------------------------------

X = np.array(X, dtype=np.float32)
detection_labels = np.array(
    detection_labels,
    dtype=np.int64
)
fault_types = np.array(
    fault_types,
    dtype=np.int64
)
sample_ids_used = np.array(
    sample_ids_used,
    dtype=np.int64
)


# ------------------------------------------------
# Save
# ------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.savez(
    OUTPUT_FILE,
    X=X,
    detection_labels=detection_labels,
    fault_types=fault_types,
    sample_ids=sample_ids_used
)


# ------------------------------------------------
# Results
# ------------------------------------------------

print("\n--------------------------------")
print("Window generation test complete")
print("--------------------------------")

print("X shape:", X.shape)
print("Detection labels:", detection_labels.shape)
print("Fault types:", fault_types.shape)
print("Sample IDs:", sample_ids_used.shape)

print("\nDetection distribution:")
print(
    "No-fault:",
    np.sum(detection_labels == 0)
)
print(
    "Fault:",
    np.sum(detection_labels == 1)
)

print("\nSaved to:")
print(OUTPUT_FILE)