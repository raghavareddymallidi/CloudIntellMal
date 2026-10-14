"""
Noise Robustness Analysis for TCN Fault Detection.

Evaluates the trained TCN model under additive Gaussian noise
at various SNR levels to measure degradation in detection and
classification performance.

SNR levels tested: Clean, 40 dB, 30 dB, 20 dB, 10 dB
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score

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

print("Model loaded successfully.")


# ------------------------------------------------
# Load test data
# ------------------------------------------------

print("Loading test data...")

test_data = np.load(
    os.path.join(DATA_DIR, "test.npz")
)

X_test = test_data["X"]
y_det_test = test_data["detection_labels"]
y_cls_test = test_data["fault_types"]

print(f"Test samples: {X_test.shape[0]}")


# ------------------------------------------------
# Noise addition function
# ------------------------------------------------

def add_noise(signal, snr_db):
    """
    Add Gaussian noise to a signal at a specified SNR level.

    Args:
        signal: numpy array of shape (samples, channels, timesteps)
        snr_db: signal-to-noise ratio in decibels

    Returns:
        Noisy signal with same shape
    """

    # Calculate signal power per sample
    signal_power = np.mean(signal ** 2, axis=(1, 2), keepdims=True)

    # Calculate noise power from SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate noise
    noise = np.random.randn(*signal.shape).astype(np.float32)
    noise = noise * np.sqrt(noise_power)

    return signal + noise


# ------------------------------------------------
# Evaluation function
# ------------------------------------------------

def evaluate_noisy(X_noisy, batch_size=64):
    """Run TCN inference on noisy data and return predictions."""

    X_tensor = torch.tensor(X_noisy, dtype=torch.float32)

    all_det_preds = []
    all_cls_preds = []

    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            end = min(start + batch_size, len(X_tensor))
            X_batch = X_tensor[start:end].to(device)

            det_out, cls_out = model(X_batch)

            all_det_preds.extend(
                torch.argmax(det_out, dim=1).cpu().numpy()
            )
            all_cls_preds.extend(
                torch.argmax(cls_out, dim=1).cpu().numpy()
            )

    return np.array(all_det_preds), np.array(all_cls_preds)


# ------------------------------------------------
# Run noise experiments
# ------------------------------------------------

SNR_LEVELS = [None, 40, 30, 20, 10]
SNR_LABELS = ["Clean", "40 dB", "30 dB", "20 dB", "10 dB"]

# Set random seed for reproducibility
np.random.seed(42)

results = []

fault_mask = y_cls_test != -1

print("\n" + "=" * 60)
print("NOISE ROBUSTNESS ANALYSIS")
print("=" * 60)

for snr, label in zip(SNR_LEVELS, SNR_LABELS):

    if snr is None:
        X_noisy = X_test
    else:
        X_noisy = add_noise(X_test, snr)

    det_preds, cls_preds = evaluate_noisy(X_noisy)

    # Detection metrics
    det_acc = accuracy_score(y_det_test, det_preds)
    det_f1_val = f1_score(
        y_det_test, det_preds, average="binary", zero_division=0
    )

    # Classification metrics (fault windows only)
    cls_acc = accuracy_score(
        y_cls_test[fault_mask],
        cls_preds[fault_mask]
    )
    cls_f1_val = f1_score(
        y_cls_test[fault_mask],
        cls_preds[fault_mask],
        average="macro",
        zero_division=0
    )

    results.append({
        "snr": label,
        "detection_accuracy": float(det_acc),
        "detection_f1": float(det_f1_val),
        "classification_accuracy": float(cls_acc),
        "classification_f1": float(cls_f1_val)
    })

    print(
        f"\n{label:>6s} | "
        f"Det Acc: {det_acc * 100:.2f}% | "
        f"Det F1: {det_f1_val * 100:.2f}% | "
        f"Cls Acc: {cls_acc * 100:.2f}% | "
        f"Cls F1: {cls_f1_val * 100:.2f}%"
    )


# ------------------------------------------------
# Save results
# ------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

# Save CSV
csv_path = os.path.join(RESULTS_DIR, "noise_results.csv")
with open(csv_path, "w") as f:
    f.write(
        "SNR,Detection_Accuracy,Detection_F1,"
        "Classification_Accuracy,Classification_F1\n"
    )
    for r in results:
        f.write(
            f"{r['snr']},"
            f"{r['detection_accuracy']:.6f},"
            f"{r['detection_f1']:.6f},"
            f"{r['classification_accuracy']:.6f},"
            f"{r['classification_f1']:.6f}\n"
        )

# Save JSON
json_path = os.path.join(RESULTS_DIR, "noise_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)


# ------------------------------------------------
# Visualization
# ------------------------------------------------

snr_labels = [r["snr"] for r in results]
det_accs = [r["detection_accuracy"] * 100 for r in results]
cls_accs = [r["classification_accuracy"] * 100 for r in results]
det_f1s = [r["detection_f1"] * 100 for r in results]
cls_f1s = [r["classification_f1"] * 100 for r in results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(
    snr_labels, det_accs,
    "o-", color="#2196F3", linewidth=2, markersize=8,
    label="Detection"
)
axes[0].plot(
    snr_labels, cls_accs,
    "s-", color="#FF5722", linewidth=2, markersize=8,
    label="Classification"
)
axes[0].set_xlabel("Noise Level (SNR)", fontsize=12)
axes[0].set_ylabel("Accuracy (%)", fontsize=12)
axes[0].set_title("TCN Accuracy vs Noise Level", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 105])

# F1 plot
axes[1].plot(
    snr_labels, det_f1s,
    "o-", color="#2196F3", linewidth=2, markersize=8,
    label="Detection"
)
axes[1].plot(
    snr_labels, cls_f1s,
    "s-", color="#FF5722", linewidth=2, markersize=8,
    label="Classification"
)
axes[1].set_xlabel("Noise Level (SNR)", fontsize=12)
axes[1].set_ylabel("F1 Score (%)", fontsize=12)
axes[1].set_title("TCN F1 Score vs Noise Level", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 105])

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "noise_robustness.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()


print("\n" + "=" * 60)
print("NOISE ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 60)
print(f"Results saved to: {csv_path}")
print(f"Results saved to: {json_path}")
print(f"Plot saved to: {RESULTS_DIR}/noise_robustness.png")
