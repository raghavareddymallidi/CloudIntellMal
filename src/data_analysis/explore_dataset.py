"""
PROTECT-90 Dataset Analysis

Comprehensive analysis of the PROTECT-90 (HV Double Line 90kV)
dataset including signal characteristics, label distributions,
and data quality checks.
"""

import os
import json
import numpy as np
import pandas as pd


# ------------------------------------------------
# Paths
# ------------------------------------------------

LABEL_FILE = "data/raw/hv_double_line_90kv_labels.csv"
DATA_DIR = "data/raw/hv_double_line_90kv_preprocessed_data"
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------
# Load metadata
# ------------------------------------------------

print("=" * 60)
print("PROTECT-90 DATASET ANALYSIS")
print("=" * 60)

labels = pd.read_csv(LABEL_FILE)

print(f"\nTotal episodes: {len(labels)}")
print(f"Label columns: {list(labels.columns)}")


# ------------------------------------------------
# Fault type distribution
# ------------------------------------------------

print("\n" + "-" * 40)
print("FAULT TYPE DISTRIBUTION")
print("-" * 40)

sc_type_counts = labels["sc_type"].value_counts().sort_index()

# Map fault types to names
fault_type_names = {
    0: "Single Line-to-Ground (SLG)",
    1: "Line-to-Line (LL)",
    2: "Double Line-to-Ground (DLG)",
    3: "Three-Phase (LLL/LLLG)"
}

for sc_type, count in sc_type_counts.items():
    name = fault_type_names.get(sc_type, f"Unknown ({sc_type})")
    pct = count / len(labels) * 100
    print(f"  sc_type {sc_type}: {count:>5} ({pct:.1f}%) — {name}")


# ------------------------------------------------
# Fault location analysis
# ------------------------------------------------

print("\n" + "-" * 40)
print("FAULT LOCATION STATISTICS")
print("-" * 40)

print(f"  Min location: {labels['sc_location'].min():.2f}")
print(f"  Max location: {labels['sc_location'].max():.2f}")
print(f"  Mean location: {labels['sc_location'].mean():.2f}")

print(f"\n  Fault target distribution:")
for target, count in labels["fault_target"].value_counts().items():
    print(f"    {target}: {count}")


# ------------------------------------------------
# Fault resistance
# ------------------------------------------------

print("\n" + "-" * 40)
print("FAULT RESISTANCE")
print("-" * 40)

print(f"  Min: {labels['fault_resistance'].min():.3f} Ω")
print(f"  Max: {labels['fault_resistance'].max():.3f} Ω")
print(f"  Mean: {labels['fault_resistance'].mean():.3f} Ω")
print(f"  Std: {labels['fault_resistance'].std():.3f} Ω")


# ------------------------------------------------
# Fault timing
# ------------------------------------------------

print("\n" + "-" * 40)
print("FAULT TIMING")
print("-" * 40)

fault_duration = labels["t_evnt_end"] - labels["t_evnt_start"]

print(f"  Fault start time — Min: {labels['t_evnt_start'].min():.4f}s, "
      f"Max: {labels['t_evnt_start'].max():.4f}s")
print(f"  Fault end time   — Min: {labels['t_evnt_end'].min():.4f}s, "
      f"Max: {labels['t_evnt_end'].max():.4f}s")
print(f"  Fault duration   — Min: {fault_duration.min():.4f}s, "
      f"Max: {fault_duration.max():.4f}s, "
      f"Mean: {fault_duration.mean():.4f}s")


# ------------------------------------------------
# Operating conditions
# ------------------------------------------------

print("\n" + "-" * 40)
print("OPERATING CONDITIONS")
print("-" * 40)

print(f"  line_2_3_b_on (parallel line): "
      f"{labels['line_2_3_b_on'].value_counts().to_dict()}")
print(f"  line_1_2_b_on (parallel line): "
      f"{labels['line_1_2_b_on'].value_counts().to_dict()}")
print(f"  ext_grid_3_on (external grid): "
      f"{labels['ext_grid_3_on'].value_counts().to_dict()}")


# ------------------------------------------------
# Waveform analysis (sample one episode)
# ------------------------------------------------

print("\n" + "-" * 40)
print("WAVEFORM CHARACTERISTICS")
print("-" * 40)

sample_id = labels["sample_id"].iloc[0]
waveform_file = os.path.join(
    DATA_DIR,
    f"{sample_id}_sample_hv_double_line_90kv.pkl"
)

waveform = pd.read_pickle(waveform_file)

time = waveform["time_s"].values
signal_columns = [
    col for col in waveform.columns
    if col != "time_s"
]

signals = waveform[signal_columns].values

dt = time[1] - time[0]
sampling_rate = 1.0 / dt

print(f"  Waveform shape: {waveform.shape}")
print(f"  Time range: {time[0]:.6f}s — {time[-1]:.6f}s")
print(f"  Duration: {time[-1] - time[0]:.4f}s")
print(f"  Sampling interval: {dt:.6f}s")
print(f"  Sampling rate: {sampling_rate:.2f} Hz")
print(f"  Total samples: {len(time)}")
print(f"  Number of channels: {len(signal_columns)}")

print(f"\n  Channel names:")
for i, col in enumerate(signal_columns):
    print(f"    [{i:2d}] {col}")

# Channel grouping
print(f"\n  Channel grouping:")
voltage_channels = [c for c in signal_columns if "vol" in c]
current_channels = [c for c in signal_columns if "cur" in c]
print(f"    Voltage channels: {len(voltage_channels)}")
print(f"    Current channels: {len(current_channels)}")


# ------------------------------------------------
# Per-channel statistics (from processed data)
# ------------------------------------------------

print("\n" + "-" * 40)
print("PROCESSED DATA STATISTICS")
print("-" * 40)

train_data = np.load(
    os.path.join(PROCESSED_DIR, "train.npz")
)

X_train = train_data["X"]
y_det = train_data["detection_labels"]
y_fault = train_data["fault_types"]

print(f"  Train X shape: {X_train.shape}")
print(f"    (samples, channels, timesteps)")
print(f"  Detection labels: {np.unique(y_det, return_counts=True)}")
print(f"  Fault types: {np.unique(y_fault, return_counts=True)}")

# Per-channel statistics
print(f"\n  Per-channel statistics (across all training windows):")
print(f"  {'Channel':>4} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10}")
print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

for ch in range(min(X_train.shape[1], 48)):
    ch_data = X_train[:, ch, :]
    print(f"  {ch:>4} | {ch_data.mean():>10.4f} | {ch_data.std():>10.4f} | "
          f"{ch_data.min():>10.4f} | {ch_data.max():>10.4f}")


# ------------------------------------------------
# Split information
# ------------------------------------------------

print("\n" + "-" * 40)
print("DATASET SPLIT")
print("-" * 40)

split_file = os.path.join(PROCESSED_DIR, "episode_split.json")

with open(split_file, "r") as f:
    splits = json.load(f)

print(f"  Train episodes: {len(splits['train'])}")
print(f"  Validation episodes: {len(splits['validation'])}")
print(f"  Test episodes: {len(splits['test'])}")

val_data = np.load(os.path.join(PROCESSED_DIR, "validation.npz"))
test_data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))

print(f"\n  Train windows: {X_train.shape[0]}")
print(f"  Validation windows: {val_data['X'].shape[0]}")
print(f"  Test windows: {test_data['X'].shape[0]}")


# ------------------------------------------------
# Save report
# ------------------------------------------------

report = {
    "dataset": "PROTECT-90 (HV Double Line 90kV)",
    "total_episodes": int(len(labels)),
    "sampling_rate_hz": float(sampling_rate),
    "duration_per_episode_s": float(time[-1] - time[0]),
    "samples_per_episode": int(len(time)),
    "num_channels": int(len(signal_columns)),
    "voltage_channels": len(voltage_channels),
    "current_channels": len(current_channels),
    "window_size": int(X_train.shape[2]),
    "fault_types": {
        str(k): {
            "name": fault_type_names.get(k, "Unknown"),
            "count": int(v)
        }
        for k, v in sc_type_counts.items()
    },
    "fault_resistance_range": {
        "min": float(labels["fault_resistance"].min()),
        "max": float(labels["fault_resistance"].max()),
        "mean": float(labels["fault_resistance"].mean())
    },
    "split": {
        "train_episodes": len(splits["train"]),
        "validation_episodes": len(splits["validation"]),
        "test_episodes": len(splits["test"]),
        "train_windows": int(X_train.shape[0]),
        "validation_windows": int(val_data["X"].shape[0]),
        "test_windows": int(test_data["X"].shape[0])
    }
}

report_file = os.path.join(RESULTS_DIR, "protect90_analysis.json")

with open(report_file, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n\nAnalysis report saved to: {report_file}")
print("\n" + "=" * 60)
print("PROTECT-90 ANALYSIS COMPLETE")
print("=" * 60)
