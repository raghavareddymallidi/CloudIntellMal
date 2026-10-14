"""
IEEE-13-Node Renewable Energy Dataset Analysis

Analyzes the IEEE-13-node distribution feeder fault dataset
structure, signal characteristics, and compatibility with
the PROTECT-90 dataset.
"""

import os
import json
import numpy as np
import pandas as pd


# ------------------------------------------------
# Paths
# ------------------------------------------------

FAULTY_DIR = "data/raw/secondary/FaultySignal_withRE"
HEALTHY_DIR = "data/raw/secondary/HealthySignal_withRE"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


print("=" * 60)
print("IEEE-13-NODE DATASET ANALYSIS")
print("=" * 60)


# ------------------------------------------------
# Load faulty signals
# ------------------------------------------------

print("\n" + "-" * 40)
print("FAULTY SIGNALS")
print("-" * 40)

faulty_files = {
    "Va": os.path.join(FAULTY_DIR, "voltage_phaseA.csv"),
    "Vb": os.path.join(FAULTY_DIR, "voltage_phaseB.csv"),
    "Vc": os.path.join(FAULTY_DIR, "voltage_phaseC.csv"),
}

faulty_data = {}

for phase, filepath in faulty_files.items():

    print(f"\n  Loading {phase}...")

    df = pd.read_csv(filepath, sep=";", encoding="utf-8-sig")

    # Ensure all values are numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")

    faulty_data[phase] = df

    print(f"    Shape: {df.shape}")
    print(f"    Columns (first 5): {list(df.columns[:5])}")
    print(f"    Columns (last 5): {list(df.columns[-5:])}")
    print(f"    Data type: {df.dtypes.iloc[0]}")

    values = df.values[~np.isnan(df.values)]
    print(f"    Min value: {values.min():.4f}")
    print(f"    Max value: {values.max():.4f}")
    print(f"    Mean value: {values.mean():.4f}")
    print(f"    Has NaN: {df.isna().any().any()}")


# ------------------------------------------------
# Load healthy signals
# ------------------------------------------------

print("\n" + "-" * 40)
print("HEALTHY SIGNALS")
print("-" * 40)

healthy_files = {
    "Va": os.path.join(HEALTHY_DIR, "voltage_phaseA_healthy.csv"),
    "Vb": os.path.join(HEALTHY_DIR, "voltage_phaseB_healthy.csv"),
    "Vc": os.path.join(HEALTHY_DIR, "voltage_phaseC_healthy.csv"),
}

healthy_data = {}

for phase, filepath in healthy_files.items():

    print(f"\n  Loading {phase}...")

    df = pd.read_csv(filepath, encoding="utf-8-sig")

    # Ensure all values are numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")

    healthy_data[phase] = df

    print(f"    Shape: {df.shape}")
    print(f"    Columns (first 5): {list(df.columns[:5])}")
    print(f"    Data type: {df.dtypes.iloc[0]}")

    values = df.values[~np.isnan(df.values)]
    print(f"    Min value: {values.min():.4f}")
    print(f"    Max value: {values.max():.4f}")
    print(f"    Mean value: {values.mean():.4f}")
    print(f"    Has NaN: {df.isna().any().any()}")


# ------------------------------------------------
# Data structure analysis
# ------------------------------------------------

print("\n" + "-" * 40)
print("DATA STRUCTURE")
print("-" * 40)

fa = faulty_data["Va"]
ha = healthy_data["Va"]

# Each row is one fault event
# Each column is one time sample
# So shape (rows, cols) = (num_events, num_time_samples)

num_faulty_events = fa.shape[0]
num_healthy_events = ha.shape[0]
samples_per_event = fa.shape[1]

print(f"\n  Faulty events: {num_faulty_events}")
print(f"  Healthy events: {num_healthy_events}")
print(f"  Time samples per event: {samples_per_event}")
print(f"  Total events: {num_faulty_events + num_healthy_events}")

# Column names suggest measurement point indices
print(f"\n  Column naming convention:")
print(f"    First column: {fa.columns[0]}")
print(f"    Last column: {fa.columns[-1]}")

# Extract measurement point numbers
col_nums = []
for col in fa.columns:
    try:
        # Try to extract number from column name like 'Vout_pA1'
        num = int("".join(filter(str.isdigit, col)))
        col_nums.append(num)
    except ValueError:
        pass

if col_nums:
    print(f"    Measurement points: {min(col_nums)} to {max(col_nums)}")


# ------------------------------------------------
# Signal characteristics
# ------------------------------------------------

print("\n" + "-" * 40)
print("SIGNAL CHARACTERISTICS")
print("-" * 40)

# Analyze a single faulty event
event_0_va = fa.iloc[0].values.astype(np.float64)
event_0_vb = faulty_data["Vb"].iloc[0].values.astype(np.float64)
event_0_vc = faulty_data["Vc"].iloc[0].values.astype(np.float64)

print(f"\n  Sample faulty event (index 0):")
print(f"    Va — Min: {event_0_va.min():.2f}, "
      f"Max: {event_0_va.max():.2f}, "
      f"Mean: {event_0_va.mean():.2f}")
print(f"    Vb — Min: {event_0_vb.min():.2f}, "
      f"Max: {event_0_vb.max():.2f}, "
      f"Mean: {event_0_vb.mean():.2f}")
print(f"    Vc — Min: {event_0_vc.min():.2f}, "
      f"Max: {event_0_vc.max():.2f}, "
      f"Mean: {event_0_vc.mean():.2f}")

# Analyze a single healthy event
h_event_0_va = ha.iloc[0].values.astype(np.float64)

print(f"\n  Sample healthy event (index 0):")
print(f"    Va — Min: {h_event_0_va.min():.2f}, "
      f"Max: {h_event_0_va.max():.2f}, "
      f"Mean: {h_event_0_va.mean():.2f}")

# Estimate voltage level
va_peak = np.abs(fa.values).max()
print(f"\n  Estimated peak voltage: {va_peak:.2f} V")
print(f"  Estimated RMS voltage: {va_peak / np.sqrt(2):.2f} V")

# No explicit sampling rate — try to estimate from signal
# Check for periodicity (50 or 60 Hz)
from scipy.signal import find_peaks

peaks, _ = find_peaks(event_0_va, distance=10)

if len(peaks) >= 2:
    avg_period_samples = np.mean(np.diff(peaks))
    print(f"\n  Estimated period (samples): {avg_period_samples:.1f}")
    print(f"  If 50Hz: sampling rate ≈ {avg_period_samples * 50:.0f} Hz")
    print(f"  If 60Hz: sampling rate ≈ {avg_period_samples * 60:.0f} Hz")


# ------------------------------------------------
# Class distribution (label inference)
# ------------------------------------------------

print("\n" + "-" * 40)
print("LABEL ANALYSIS")
print("-" * 40)

print(f"\n  No explicit fault-type labels found in CSV files.")
print(f"  Labels must be inferred from:")
print(f"    - File organization (faulty vs healthy)")
print(f"    - Row index may encode fault type")
print(f"    - External metadata if available")

print(f"\n  Available labels (from file structure):")
print(f"    Faulty: {num_faulty_events} events")
print(f"    Healthy: {num_healthy_events} events")
print(f"    → Binary detection labels available")
print(f"    → Fault-type classification NOT available "
      f"without additional metadata")


# ------------------------------------------------
# Compatibility with PROTECT-90
# ------------------------------------------------

print("\n" + "-" * 40)
print("COMPATIBILITY WITH PROTECT-90")
print("-" * 40)

compatibility = {
    "channels": {
        "protect90": "48 (V+I at 8 locations × 3 phases)",
        "ieee13": "3 (V only × 3 phases)",
        "compatible": False,
        "reason": "IEEE-13 lacks current channels entirely"
    },
    "signals": {
        "protect90": "Voltage + Current (Va,Vb,Vc,Ia,Ib,Ic)",
        "ieee13": "Voltage only (Va,Vb,Vc)",
        "compatible": False,
        "reason": "Cannot create 6-channel input from voltage-only data"
    },
    "samples_per_event": {
        "protect90": 6400,
        "ieee13": samples_per_event,
        "compatible": False,
        "reason": "Different event lengths"
    },
    "fault_labels": {
        "protect90": "4 types (sc_type 0-3)",
        "ieee13": "Binary only (faulty/healthy)",
        "compatible": False,
        "reason": "IEEE-13 has no fault-type classification labels"
    },
    "grid_topology": {
        "protect90": "90kV HV transmission (double line)",
        "ieee13": "13-node distribution feeder with renewables",
        "compatible": False,
        "reason": "Different voltage levels and grid types"
    }
}

for aspect, info in compatibility.items():
    status = "✓" if info["compatible"] else "✗"
    print(f"\n  [{status}] {aspect}:")
    print(f"      PROTECT-90: {info['protect90']}")
    print(f"      IEEE-13:    {info['ieee13']}")
    if not info["compatible"]:
        print(f"      Issue:      {info['reason']}")

print("\n" + "-" * 40)
print("RECOMMENDATION")
print("-" * 40)

print("""
  The IEEE-13-node and PROTECT-90 datasets CANNOT be merged.

  Recommended strategies:

  A. SEPARATE TRAINING/EVALUATION
     - Train separate models on each dataset
     - Compare performance independently
     - IEEE-13: voltage-only binary detection model
     - PROTECT-90: full 48-channel detection + classification

  B. CROSS-DATASET GENERALIZATION TEST
     - Train on PROTECT-90 voltage channels only
     - Test zero-shot on IEEE-13 after resampling
     - Evaluates domain generalization

  C. TRANSFER LEARNING (future work)
     - Pretrain shared feature extractor on PROTECT-90
     - Fine-tune on IEEE-13 with reduced input

  For the current project phase:
  → Focus on PROTECT-90 as the primary dataset
  → Use IEEE-13 for supplementary binary detection evaluation
""")


# ------------------------------------------------
# Save report
# ------------------------------------------------

report = {
    "dataset": "IEEE-13-Node Renewable Energy Fault Dataset",
    "faulty_events": int(num_faulty_events),
    "healthy_events": int(num_healthy_events),
    "total_events": int(num_faulty_events + num_healthy_events),
    "samples_per_event": int(samples_per_event),
    "channels": 3,
    "channel_names": ["Va (voltage_phaseA)", "Vb (voltage_phaseB)", "Vc (voltage_phaseC)"],
    "signal_type": "Voltage only (no current)",
    "has_fault_type_labels": False,
    "has_detection_labels": True,
    "compatible_with_protect90": False,
    "incompatibility_reasons": [
        "IEEE-13 has voltage only (no current channels)",
        "Different number of channels (3 vs 48)",
        "Different event lengths (1024 vs 6400 samples)",
        "No fault-type labels in IEEE-13",
        "Different grid topologies and voltage levels"
    ],
    "recommended_strategy": "Separate training/evaluation; "
                           "IEEE-13 for supplementary binary detection only"
}

report_file = os.path.join(RESULTS_DIR, "ieee13_analysis.json")

with open(report_file, "w") as f:
    json.dump(report, f, indent=2)

print(f"Analysis report saved to: {report_file}")
print("\n" + "=" * 60)
print("IEEE-13-NODE ANALYSIS COMPLETE")
print("=" * 60)
