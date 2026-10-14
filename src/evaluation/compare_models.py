"""
Model Comparison Module.

Loads results from all models (TCN, Decision Tree, SVM, CNN, LSTM)
and generates:
    - Unified comparison table
    - Accuracy comparison bar chart
    - F1 score comparison bar chart
    - Confusion matrix heatmaps for TCN
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.models.config import RESULTS_DIR, FAULT_TYPE_NAMES


# ------------------------------------------------
# Load results
# ------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)


def load_json(filename):
    """Load a JSON results file or return None."""
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def parse_txt_results(filename):
    """
    Parse a text results file for Detection/Classification
    accuracy values. Returns dict with accuracy values.
    """

    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None

    result = {
        "detection": {},
        "classification": {},
        "timing": {}
    }

    with open(path, "r") as f:
        content = f.read()

    lines = content.split("\n")
    section = None

    def parse_value(raw):
        """Parse a metric value, auto-detecting decimal vs percentage."""
        is_pct = "%" in raw
        val = raw.replace("%", "").strip()
        val = float(val)
        if is_pct or val > 1.0:
            return val / 100
        return val

    for line in lines:
        line_stripped = line.strip()

        if "Fault Detection" in line_stripped or "FAULT DETECTION" in line_stripped:
            section = "detection"
        elif "Fault Classification" in line_stripped or "FAULT CLASSIFICATION" in line_stripped:
            section = "classification"

        if section and "Accuracy" in line_stripped:
            try:
                raw = line_stripped.split(":")[-1].strip()
                result[section]["accuracy"] = parse_value(raw)
            except (ValueError, IndexError):
                pass

        if section and "F1" in line_stripped:
            try:
                raw = line_stripped.split(":")[-1].strip()
                raw = raw.replace("(macro)", "").strip()
                key = "f1_score" if section == "detection" else "f1_macro"
                result[section][key] = parse_value(raw)
            except (ValueError, IndexError):
                pass

        if "Training time" in line_stripped:
            try:
                val = line_stripped.split(":")[-1].strip()
                val = val.replace("s", "").strip()
                result["timing"]["training_time_s"] = float(val)
            except (ValueError, IndexError):
                pass

        if "Inference" in line_stripped and "sample" in line_stripped:
            try:
                val = line_stripped.split(":")[-1].strip()
                val = val.replace("ms", "").strip()
                result["timing"]["inference_per_sample_ms"] = float(val)
            except (ValueError, IndexError):
                pass

    return result


# Load all results
models = {}

# TCN (JSON)
tcn = load_json("tcn_evaluation_results.json")
if tcn:
    models["TCN"] = tcn

# Decision Tree (text)
dt = parse_txt_results("decision_tree_results.txt")
if dt:
    models["Decision Tree"] = dt

# SVM (text)
svm = parse_txt_results("svm_results.txt")
if svm:
    models["SVM"] = svm

# CNN (JSON)
cnn = load_json("cnn_results.json")
if cnn:
    models["CNN"] = cnn

# LSTM (JSON)
lstm = load_json("lstm_results.json")
if lstm:
    models["LSTM"] = lstm


print(f"\nModels loaded: {list(models.keys())}")


# ------------------------------------------------
# Build comparison table
# ------------------------------------------------

def get_metric(model_data, task, metric, default=None):
    """Safely extract a metric from model results."""

    if model_data is None:
        return default

    task_data = model_data.get(task, {})
    if task_data is None:
        return default

    return task_data.get(metric, default)


print("\n" + "-" * 80)
print(
    f"{'Model':<15} | "
    f"{'Det Acc':>8} | "
    f"{'Det F1':>8} | "
    f"{'Cls Acc':>8} | "
    f"{'Cls F1':>8} | "
    f"{'Train(s)':>9} | "
    f"{'Inf(ms)':>8}"
)
print("-" * 80)

comparison_rows = []

for name, data in models.items():

    det_acc = get_metric(data, "detection", "accuracy")
    det_f1 = get_metric(data, "detection", "f1_score")
    cls_acc = get_metric(data, "classification", "accuracy")
    cls_f1 = get_metric(
        data, "classification", "f1_macro",
        get_metric(data, "classification", "f1_score")
    )

    train_time = get_metric(data, "timing", "training_time_s")
    inf_time = get_metric(
        data, "timing", "inference_per_sample_ms",
        get_metric(data, "inference", "per_sample_ms")
    )

    row = {
        "model": name,
        "det_accuracy": det_acc,
        "det_f1": det_f1,
        "cls_accuracy": cls_acc,
        "cls_f1": cls_f1,
        "training_time_s": train_time,
        "inference_ms": inf_time
    }
    comparison_rows.append(row)

    print(
        f"{name:<15} | "
        f"{det_acc * 100 if det_acc else 0:>7.2f}% | "
        f"{det_f1 * 100 if det_f1 else 0:>7.2f}% | "
        f"{cls_acc * 100 if cls_acc else 0:>7.2f}% | "
        f"{cls_f1 * 100 if cls_f1 else 0:>7.2f}% | "
        f"{train_time if train_time else 0:>9.2f} | "
        f"{inf_time if inf_time else 0:>7.3f}"
    )

print("-" * 80)


# ------------------------------------------------
# Save comparison text
# ------------------------------------------------

comp_file = os.path.join(RESULTS_DIR, "model_comparison.txt")

with open(comp_file, "w") as f:

    f.write("MODEL COMPARISON RESULTS\n")
    f.write("=" * 80 + "\n\n")

    f.write(
        f"{'Model':<15} | "
        f"{'Det Acc':>8} | "
        f"{'Det F1':>8} | "
        f"{'Cls Acc':>8} | "
        f"{'Cls F1':>8} | "
        f"{'Train(s)':>9} | "
        f"{'Inf(ms)':>8}\n"
    )
    f.write("-" * 80 + "\n")

    for row in comparison_rows:
        f.write(
            f"{row['model']:<15} | "
            f"{row['det_accuracy'] * 100 if row['det_accuracy'] else 0:>7.2f}% | "
            f"{row['det_f1'] * 100 if row['det_f1'] else 0:>7.2f}% | "
            f"{row['cls_accuracy'] * 100 if row['cls_accuracy'] else 0:>7.2f}% | "
            f"{row['cls_f1'] * 100 if row['cls_f1'] else 0:>7.2f}% | "
            f"{row['training_time_s'] if row['training_time_s'] else 0:>9.2f} | "
            f"{row['inference_ms'] if row['inference_ms'] else 0:>7.3f}\n"
        )

    f.write("-" * 80 + "\n")

# Save comparison JSON
comp_json = os.path.join(RESULTS_DIR, "model_comparison.json")
with open(comp_json, "w") as f:
    json.dump(comparison_rows, f, indent=2)


# ------------------------------------------------
# Visualization: Accuracy Comparison
# ------------------------------------------------

model_names = [r["model"] for r in comparison_rows]
det_accs = [
    r["det_accuracy"] * 100 if r["det_accuracy"] else 0
    for r in comparison_rows
]
cls_accs = [
    r["cls_accuracy"] * 100 if r["cls_accuracy"] else 0
    for r in comparison_rows
]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(
    x - width / 2, det_accs, width,
    label="Detection Accuracy",
    color="#2196F3", edgecolor="white", linewidth=0.5
)
bars2 = ax.bar(
    x + width / 2, cls_accs, width,
    label="Classification Accuracy",
    color="#FF5722", edgecolor="white", linewidth=0.5
)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center", va="bottom", fontsize=9
    )

for bar in bars2:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center", va="bottom", fontsize=9
    )

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title(
    "Model Comparison — Detection & Classification Accuracy",
    fontsize=13, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0, 110])
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "accuracy_comparison.png"),
    dpi=150, bbox_inches="tight"
)
plt.close()


# ------------------------------------------------
# Visualization: F1 Score Comparison
# ------------------------------------------------

det_f1s = [
    r["det_f1"] * 100 if r["det_f1"] else 0
    for r in comparison_rows
]
cls_f1s = [
    r["cls_f1"] * 100 if r["cls_f1"] else 0
    for r in comparison_rows
]

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(
    x - width / 2, det_f1s, width,
    label="Detection F1",
    color="#4CAF50", edgecolor="white", linewidth=0.5
)
bars2 = ax.bar(
    x + width / 2, cls_f1s, width,
    label="Classification F1",
    color="#9C27B0", edgecolor="white", linewidth=0.5
)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center", va="bottom", fontsize=9
    )

for bar in bars2:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center", va="bottom", fontsize=9
    )

ax.set_ylabel("F1 Score (%)", fontsize=12)
ax.set_title(
    "Model Comparison — Detection & Classification F1 Score",
    fontsize=13, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0, 110])
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "f1_comparison.png"),
    dpi=150, bbox_inches="tight"
)
plt.close()


# ------------------------------------------------
# Visualization: TCN Confusion Matrices
# ------------------------------------------------

if tcn:

    fault_names = [
        FAULT_TYPE_NAMES.get(i, f"Class {i}").split(" ")[0]
        for i in range(4)
    ]

    # Detection confusion matrix
    det_cm = np.array(tcn["detection"]["confusion_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im1 = axes[0].imshow(det_cm, cmap="Blues", aspect="auto")
    axes[0].set_title(
        "TCN — Detection Confusion Matrix",
        fontsize=12, fontweight="bold"
    )
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("Actual", fontsize=11)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["No Fault", "Fault"])
    axes[0].set_yticklabels(["No Fault", "Fault"])

    for i in range(2):
        for j in range(2):
            axes[0].text(
                j, i, str(det_cm[i, j]),
                ha="center", va="center",
                fontsize=16, fontweight="bold",
                color="white" if det_cm[i, j] > det_cm.max() / 2 else "black"
            )

    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Classification confusion matrix
    cls_cm = np.array(tcn["classification"]["confusion_matrix"])

    im2 = axes[1].imshow(cls_cm, cmap="Oranges", aspect="auto")
    axes[1].set_title(
        "TCN — Classification Confusion Matrix",
        fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("Actual", fontsize=11)
    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(fault_names)
    axes[1].set_yticklabels(fault_names)

    for i in range(4):
        for j in range(4):
            axes[1].text(
                j, i, str(cls_cm[i, j]),
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cls_cm[i, j] > cls_cm.max() / 2 else "black"
            )

    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "tcn_confusion_matrices.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()

    print("\nTCN confusion matrices saved.")


print("\n" + "=" * 60)
print("MODEL COMPARISON COMPLETE")
print("=" * 60)
print(f"Results saved to: {comp_file}")
print(f"Results saved to: {comp_json}")
print(f"Plots saved to: {RESULTS_DIR}/accuracy_comparison.png")
print(f"Plots saved to: {RESULTS_DIR}/f1_comparison.png")
print(f"Plots saved to: {RESULTS_DIR}/tcn_confusion_matrices.png")
