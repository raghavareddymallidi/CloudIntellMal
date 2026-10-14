"""
Comprehensive TCN Model Evaluation

Computes all metrics from the project specification:
    - Detection: accuracy, precision, recall, F1, FPR, FNR
    - Classification: accuracy, precision, recall, F1 (per-class + macro)
    - Inference time (per-sample and batch)
    - Confusion matrices
    - Classification report
"""

import os
import time
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.models.tcn import TCN
from src.models.config import (
    DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    INPUT_CHANNELS,
    NUM_FAULT_CLASSES,
    TCN_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_DROPOUT,
    FAULT_TYPE_NAMES
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

y_detection = test_data["detection_labels"]
y_fault = test_data["fault_types"]

print("Test X:", X_test.shape)
print("Detection labels:", y_detection.shape)
print("Fault labels:", y_fault.shape)


# ------------------------------------------------
# Load model
# ------------------------------------------------

print("\nLoading model...")

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

# Support both old format (state_dict only) and new (checkpoint dict)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Val loss: {checkpoint.get('val_loss', '?')}")
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,}")
print(f"  Receptive field: {model.get_receptive_field()} samples")


# ------------------------------------------------
# Generate predictions with timing
# ------------------------------------------------

print("\nGenerating predictions...")

batch_size = 64

all_det_preds = []
all_cls_preds = []
all_det_probs = []
all_cls_probs = []

# Inference timing
inference_times = []

with torch.no_grad():

    for start in range(0, len(X_test), batch_size):

        end = min(start + batch_size, len(X_test))
        X_batch = X_test[start:end].to(device)

        t_start = time.time()
        det_out, cls_out = model(X_batch)
        t_end = time.time()

        inference_times.append(t_end - t_start)

        # Predictions
        det_preds = torch.argmax(det_out, dim=1).cpu().numpy()
        cls_preds = torch.argmax(cls_out, dim=1).cpu().numpy()

        # Probabilities
        det_probs = torch.softmax(det_out, dim=1).cpu().numpy()
        cls_probs = torch.softmax(cls_out, dim=1).cpu().numpy()

        all_det_preds.extend(det_preds)
        all_cls_preds.extend(cls_preds)
        all_det_probs.extend(det_probs)
        all_cls_probs.extend(cls_probs)


all_det_preds = np.array(all_det_preds)
all_cls_preds = np.array(all_cls_preds)
all_det_probs = np.array(all_det_probs)
all_cls_probs = np.array(all_cls_probs)


# ------------------------------------------------
# Inference time analysis
# ------------------------------------------------

total_inference_time = sum(inference_times)
per_sample_time = total_inference_time / len(X_test)

print(f"\nTotal inference time: {total_inference_time:.4f}s")
print(f"Per-sample inference: {per_sample_time * 1000:.3f}ms")
print(f"Throughput: {len(X_test) / total_inference_time:.0f} samples/s")


# ------------------------------------------------
# Detection metrics
# ------------------------------------------------

print("\n" + "=" * 60)
print("FAULT DETECTION RESULTS")
print("=" * 60)

det_accuracy = accuracy_score(y_detection, all_det_preds)
det_precision = precision_score(
    y_detection, all_det_preds, average="binary", zero_division=0
)
det_recall = recall_score(
    y_detection, all_det_preds, average="binary", zero_division=0
)
det_f1 = f1_score(
    y_detection, all_det_preds, average="binary", zero_division=0
)

det_cm = confusion_matrix(y_detection, all_det_preds)

# FPR and FNR
tn, fp, fn, tp = det_cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"\nAccuracy : {det_accuracy:.4f}")
print(f"Precision: {det_precision:.4f}")
print(f"Recall   : {det_recall:.4f}")
print(f"F1 Score : {det_f1:.4f}")
print(f"FPR      : {fpr:.4f}")
print(f"FNR      : {fnr:.4f}")
print(f"\nConfusion Matrix:")
print(f"  TN={tn:>5}  FP={fp:>5}")
print(f"  FN={fn:>5}  TP={tp:>5}")


# ------------------------------------------------
# Classification metrics
# ------------------------------------------------

print("\n" + "=" * 60)
print("FAULT CLASSIFICATION RESULTS")
print("=" * 60)

fault_mask = y_fault != -1
y_fault_true = y_fault[fault_mask]
y_fault_pred = all_cls_preds[fault_mask]

cls_accuracy = accuracy_score(y_fault_true, y_fault_pred)
cls_precision = precision_score(
    y_fault_true, y_fault_pred, average="macro", zero_division=0
)
cls_recall = recall_score(
    y_fault_true, y_fault_pred, average="macro", zero_division=0
)
cls_f1 = f1_score(
    y_fault_true, y_fault_pred, average="macro", zero_division=0
)

print(f"\nAccuracy : {cls_accuracy:.4f}")
print(f"Precision: {cls_precision:.4f} (macro)")
print(f"Recall   : {cls_recall:.4f} (macro)")
print(f"F1 Score : {cls_f1:.4f} (macro)")

# Per-class report
class_labels = list(range(NUM_FAULT_CLASSES))
class_names = [
    FAULT_TYPE_NAMES.get(i, f"Class {i}")
    for i in class_labels
]

print("\nClassification Report:")
print(
    classification_report(
        y_fault_true,
        y_fault_pred,
        labels=class_labels,
        target_names=class_names,
        zero_division=0
    )
)

fault_cm = confusion_matrix(
    y_fault_true,
    y_fault_pred,
    labels=class_labels
)

print("Confusion Matrix:")
print(fault_cm)


# ------------------------------------------------
# Save results
# ------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

# Save confusion matrices
np.save(
    os.path.join(RESULTS_DIR, "detection_confusion_matrix.npy"),
    det_cm
)

np.save(
    os.path.join(RESULTS_DIR, "fault_classification_confusion_matrix.npy"),
    fault_cm
)

# Save comprehensive results to JSON
results = {
    "model": "TCN",
    "parameters": int(total_params),
    "receptive_field": model.get_receptive_field(),
    "detection": {
        "accuracy": float(det_accuracy),
        "precision": float(det_precision),
        "recall": float(det_recall),
        "f1_score": float(det_f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "confusion_matrix": det_cm.tolist()
    },
    "classification": {
        "accuracy": float(cls_accuracy),
        "precision_macro": float(cls_precision),
        "recall_macro": float(cls_recall),
        "f1_macro": float(cls_f1),
        "confusion_matrix": fault_cm.tolist(),
        "per_class": {}
    },
    "inference": {
        "total_time_s": float(total_inference_time),
        "per_sample_ms": float(per_sample_time * 1000),
        "throughput_samples_per_s": float(
            len(X_test) / total_inference_time
        )
    }
}

# Per-class metrics
for i, name in enumerate(class_names):
    mask = y_fault_true == i
    if mask.sum() > 0:
        cls_p = precision_score(
            y_fault_true == i,
            y_fault_pred == i,
            zero_division=0
        )
        cls_r = recall_score(
            y_fault_true == i,
            y_fault_pred == i,
            zero_division=0
        )
        cls_f = f1_score(
            y_fault_true == i,
            y_fault_pred == i,
            zero_division=0
        )
        results["classification"]["per_class"][name] = {
            "precision": float(cls_p),
            "recall": float(cls_r),
            "f1_score": float(cls_f),
            "support": int(mask.sum())
        }

results_file = os.path.join(
    RESULTS_DIR,
    "tcn_evaluation_results.json"
)

with open(results_file, "w") as f:
    json.dump(results, f, indent=2)


# ------------------------------------------------
# Save text report
# ------------------------------------------------

report_file = os.path.join(
    RESULTS_DIR,
    "tcn_evaluation_report.txt"
)

with open(report_file, "w") as f:

    f.write("TCN EVALUATION REPORT\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Model parameters: {total_params:,}\n")
    f.write(f"Receptive field: {model.get_receptive_field()} samples\n\n")

    f.write("FAULT DETECTION\n")
    f.write("-" * 30 + "\n")
    f.write(f"Accuracy : {det_accuracy:.4f}\n")
    f.write(f"Precision: {det_precision:.4f}\n")
    f.write(f"Recall   : {det_recall:.4f}\n")
    f.write(f"F1 Score : {det_f1:.4f}\n")
    f.write(f"FPR      : {fpr:.4f}\n")
    f.write(f"FNR      : {fnr:.4f}\n\n")

    f.write("FAULT CLASSIFICATION\n")
    f.write("-" * 30 + "\n")
    f.write(f"Accuracy : {cls_accuracy:.4f}\n")
    f.write(f"Precision: {cls_precision:.4f} (macro)\n")
    f.write(f"Recall   : {cls_recall:.4f} (macro)\n")
    f.write(f"F1 Score : {cls_f1:.4f} (macro)\n\n")

    f.write("INFERENCE PERFORMANCE\n")
    f.write("-" * 30 + "\n")
    f.write(f"Per-sample: {per_sample_time * 1000:.3f} ms\n")
    f.write(f"Throughput: {len(X_test) / total_inference_time:.0f} samples/s\n")


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(f"\nResults saved to:")
print(f"  {results_file}")
print(f"  {report_file}")