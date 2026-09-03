import os
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


# ------------------------------------------------
# Paths
# ------------------------------------------------

TEST_FILE = "data/processed/test.npz"
MODEL_FILE = "models/tcn_best.pt"
RESULTS_DIR = "results"


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

test_data = np.load(TEST_FILE)

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

model = TCN(
    input_channels=48,
    num_classes=4
)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location="cpu"
    )
)

model = model.to(device)
model.eval()

print("\nModel loaded successfully.")


# ------------------------------------------------
# Generate predictions
# ------------------------------------------------

print("\nGenerating predictions...")

all_detection_predictions = []
all_fault_predictions = []


batch_size = 64

with torch.no_grad():

    for start in range(0, len(X_test), batch_size):

        end = min(
            start + batch_size,
            len(X_test)
        )

        X_batch = X_test[start:end].to(device)

        detection_output, classification_output = model(
            X_batch
        )

        detection_predictions = torch.argmax(
            detection_output,
            dim=1
        ).cpu().numpy()

        fault_predictions = torch.argmax(
            classification_output,
            dim=1
        ).cpu().numpy()

        all_detection_predictions.extend(
            detection_predictions
        )

        all_fault_predictions.extend(
            fault_predictions
        )


all_detection_predictions = np.array(
    all_detection_predictions
)

all_fault_predictions = np.array(
    all_fault_predictions
)


# ------------------------------------------------
# Detection metrics
# ------------------------------------------------

detection_accuracy = accuracy_score(
    y_detection,
    all_detection_predictions
)

detection_precision = precision_score(
    y_detection,
    all_detection_predictions,
    average="binary",
    zero_division=0
)

detection_recall = recall_score(
    y_detection,
    all_detection_predictions,
    average="binary",
    zero_division=0
)

detection_f1 = f1_score(
    y_detection,
    all_detection_predictions,
    average="binary",
    zero_division=0
)


print("\n================================")
print("FAULT DETECTION RESULTS")
print("================================")

print(
    f"Accuracy : {detection_accuracy:.4f}"
)

print(
    f"Precision: {detection_precision:.4f}"
)

print(
    f"Recall   : {detection_recall:.4f}"
)

print(
    f"F1 Score : {detection_f1:.4f}"
)


print("\nDetection confusion matrix:")

detection_cm = confusion_matrix(
    y_detection,
    all_detection_predictions
)

print(detection_cm)


# ------------------------------------------------
# Fault classification
# ------------------------------------------------

# Only fault windows have valid fault-type labels.
fault_mask = y_fault != -1

y_fault_true = y_fault[fault_mask]

y_fault_pred = all_fault_predictions[fault_mask]


fault_accuracy = accuracy_score(
    y_fault_true,
    y_fault_pred
)

fault_precision = precision_score(
    y_fault_true,
    y_fault_pred,
    average="macro",
    zero_division=0
)

fault_recall = recall_score(
    y_fault_true,
    y_fault_pred,
    average="macro",
    zero_division=0
)

fault_f1 = f1_score(
    y_fault_true,
    y_fault_pred,
    average="macro",
    zero_division=0
)


print("\n================================")
print("FAULT CLASSIFICATION RESULTS")
print("================================")

print(
    f"Accuracy : {fault_accuracy:.4f}"
)

print(
    f"Precision: {fault_precision:.4f}"
)

print(
    f"Recall   : {fault_recall:.4f}"
)

print(
    f"F1 Score : {fault_f1:.4f}"
)


# ------------------------------------------------
# Classification report
# ------------------------------------------------

print("\nClassification report:")

print(
    classification_report(
        y_fault_true,
        y_fault_pred,
        labels=[0, 1, 2, 3],
        target_names=[
            "Class 0",
            "Class 1",
            "Class 2",
            "Class 3"
        ],
        zero_division=0
    )
)


# ------------------------------------------------
# Classification confusion matrix
# ------------------------------------------------

print("Fault classification confusion matrix:")

fault_cm = confusion_matrix(
    y_fault_true,
    y_fault_pred,
    labels=[0, 1, 2, 3]
)

print(fault_cm)


# ------------------------------------------------
# Save results
# ------------------------------------------------

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "detection_confusion_matrix.npy"
    ),
    detection_cm
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "fault_classification_confusion_matrix.npy"
    ),
    fault_cm
)


# ------------------------------------------------
# Final message
# ------------------------------------------------

print("\n================================")
print("EVALUATION COMPLETE")
print("================================")

print(
    "Results saved in:",
    RESULTS_DIR
)