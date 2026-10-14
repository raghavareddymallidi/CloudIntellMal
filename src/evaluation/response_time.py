"""
Response Time Analysis for TCN Inference.

Measures single-sample and batch inference times for the trained
TCN model to evaluate its suitability for protection applications.

Reports: average, minimum, maximum, standard deviation, and
throughput across multiple measurement runs.
"""

import os
import time
import json
import numpy as np
import torch
import platform

from src.models.tcn import TCN
from src.models.config import (
    DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    INPUT_CHANNELS,
    NUM_FAULT_CLASSES,
    TCN_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_DROPOUT
)


# ------------------------------------------------
# Device
# ------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ------------------------------------------------
# Load model
# ------------------------------------------------

print("\nLoading TCN model...")

model = TCN(
    input_channels=INPUT_CHANNELS,
    num_classes=NUM_FAULT_CLASSES,
    tcn_channels=TCN_CHANNELS,
    kernel_size=TCN_KERNEL_SIZE,
    dropout=TCN_DROPOUT
)

checkpoint = torch.load(
    os.path.join(MODEL_DIR, "tcn_best.pt"),
    map_location="cpu",
    weights_only=True
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())

print(f"Model parameters: {total_params:,}")
print(f"Receptive field: {model.get_receptive_field()} samples")


# ------------------------------------------------
# Load test data
# ------------------------------------------------

print("\nLoading test data...")

test_data = np.load(
    os.path.join(DATA_DIR, "test.npz")
)

X_test = torch.tensor(
    test_data["X"],
    dtype=torch.float32
)

print(f"Test samples: {X_test.shape[0]}")


# ------------------------------------------------
# Warmup (ensure model is loaded on device)
# ------------------------------------------------

print("\nWarming up model...")

with torch.no_grad():
    dummy = torch.randn(1, INPUT_CHANNELS, 256).to(device)
    for _ in range(10):
        _ = model(dummy)


# ------------------------------------------------
# Single-sample inference timing
# ------------------------------------------------

print("\nMeasuring single-sample inference time...")

NUM_SINGLE_SAMPLES = min(1000, len(X_test))

single_times = []

with torch.no_grad():
    for i in range(NUM_SINGLE_SAMPLES):

        x = X_test[i:i + 1].to(device)

        start = time.perf_counter()
        _, _ = model(x)
        end = time.perf_counter()

        single_times.append((end - start) * 1000)  # ms

single_times = np.array(single_times)


# ------------------------------------------------
# Batch inference timing
# ------------------------------------------------

print("Measuring batch inference time...")

BATCH_SIZE = 64
batch_times = []

with torch.no_grad():
    for start_idx in range(0, len(X_test), BATCH_SIZE):

        end_idx = min(start_idx + BATCH_SIZE, len(X_test))
        X_batch = X_test[start_idx:end_idx].to(device)

        start = time.perf_counter()
        _, _ = model(X_batch)
        end = time.perf_counter()

        batch_size_actual = end_idx - start_idx
        per_sample = (end - start) / batch_size_actual * 1000  # ms
        batch_times.append(per_sample)

batch_times = np.array(batch_times)


# ------------------------------------------------
# System information
# ------------------------------------------------

system_info = {
    "platform": platform.platform(),
    "processor": platform.processor(),
    "python_version": platform.python_version(),
    "pytorch_version": torch.__version__,
    "device": str(device)
}


# ------------------------------------------------
# Print results
# ------------------------------------------------

print("\n" + "=" * 60)
print("RESPONSE TIME ANALYSIS")
print("=" * 60)

print(f"\nSystem: {system_info['platform']}")
print(f"Processor: {system_info['processor']}")
print(f"PyTorch: {system_info['pytorch_version']}")
print(f"Device: {device}")

print(f"\n--- Single-Sample Inference ({NUM_SINGLE_SAMPLES} samples) ---")
print(f"Average: {single_times.mean():.3f} ms")
print(f"Std Dev: {single_times.std():.3f} ms")
print(f"Minimum: {single_times.min():.3f} ms")
print(f"Maximum: {single_times.max():.3f} ms")
print(f"Median : {np.median(single_times):.3f} ms")

print(f"\n--- Batch Inference (batch size {BATCH_SIZE}) ---")
print(f"Average per sample: {batch_times.mean():.3f} ms")
print(f"Std Dev per sample: {batch_times.std():.3f} ms")
print(f"Minimum per sample: {batch_times.min():.3f} ms")
print(f"Maximum per sample: {batch_times.max():.3f} ms")

throughput = 1000 / batch_times.mean()
print(f"\nThroughput: {throughput:.0f} samples/s")


# ------------------------------------------------
# Save results
# ------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

results = {
    "model": "TCN",
    "parameters": int(total_params),
    "receptive_field": model.get_receptive_field(),
    "system": system_info,
    "single_sample": {
        "num_samples": int(NUM_SINGLE_SAMPLES),
        "average_ms": float(single_times.mean()),
        "std_ms": float(single_times.std()),
        "min_ms": float(single_times.min()),
        "max_ms": float(single_times.max()),
        "median_ms": float(np.median(single_times))
    },
    "batch": {
        "batch_size": BATCH_SIZE,
        "average_per_sample_ms": float(batch_times.mean()),
        "std_per_sample_ms": float(batch_times.std()),
        "min_per_sample_ms": float(batch_times.min()),
        "max_per_sample_ms": float(batch_times.max()),
        "throughput_samples_per_s": float(throughput)
    }
}

# Save JSON
json_path = os.path.join(RESULTS_DIR, "response_time_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

# Save text report
txt_path = os.path.join(RESULTS_DIR, "response_time_results.txt")
with open(txt_path, "w") as f:

    f.write("TCN RESPONSE TIME ANALYSIS\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"System: {system_info['platform']}\n")
    f.write(f"Processor: {system_info['processor']}\n")
    f.write(f"PyTorch: {system_info['pytorch_version']}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Model parameters: {total_params:,}\n")
    f.write(f"Receptive field: {model.get_receptive_field()} samples\n\n")

    f.write(f"Single-Sample Inference ({NUM_SINGLE_SAMPLES} samples)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Average: {single_times.mean():.3f} ms\n")
    f.write(f"Std Dev: {single_times.std():.3f} ms\n")
    f.write(f"Minimum: {single_times.min():.3f} ms\n")
    f.write(f"Maximum: {single_times.max():.3f} ms\n")
    f.write(f"Median : {np.median(single_times):.3f} ms\n\n")

    f.write(f"Batch Inference (batch size {BATCH_SIZE})\n")
    f.write("-" * 40 + "\n")
    f.write(f"Average per sample: {batch_times.mean():.3f} ms\n")
    f.write(f"Std Dev per sample: {batch_times.std():.3f} ms\n")
    f.write(f"Minimum per sample: {batch_times.min():.3f} ms\n")
    f.write(f"Maximum per sample: {batch_times.max():.3f} ms\n")
    f.write(f"Throughput: {throughput:.0f} samples/s\n")


print("\n" + "=" * 60)
print("RESPONSE TIME ANALYSIS COMPLETE")
print("=" * 60)
print(f"Results saved to: {txt_path}")
print(f"Results saved to: {json_path}")
