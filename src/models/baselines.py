import os
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ------------------------------------------------
# Paths
# ------------------------------------------------

DATA_DIR = "data/processed"
RESULTS_DIR = "results"


# ------------------------------------------------
# Load data
# ------------------------------------------------

print("Loading processed data...")

train_data = np.load(
    os.path.join(DATA_DIR, "train.npz")
)

test_data = np.load(
    os.path.join(DATA_DIR, "test.npz")
)

X_train = train_data["X"]
X_test = test_data["X"]

y_train_detection = train_data["detection_labels"]
y_test_detection = test_data["detection_labels"]

y_train_fault = train_data["fault_types"]
y_test_fault = test_data["fault_types"]


print("Training data:", X_train.shape)
print("Test data:", X_test.shape)


# ------------------------------------------------
# Flatten waveform data
# ------------------------------------------------

# TCN input:
# (samples, channels, time)
#
# Decision Tree input:
# (samples, features)

X_train_flat = X_train.reshape(
    X_train.shape[0],
    -1
)

X_test_flat = X_test.reshape(
    X_test.shape[0],
    -1
)


print("Flattened training data:", X_train_flat.shape)
print("Flattened test data:", X_test_flat.shape)


# ------------------------------------------------
# Decision Tree - Fault Detection
# ------------------------------------------------

print("\nTraining Decision Tree for fault detection...")

detection_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=20
)

detection_model.fit(
    X_train_flat,
    y_train_detection
)

detection_predictions = detection_model.predict(
    X_test_flat
)


# ------------------------------------------------
# Detection metrics
# ------------------------------------------------

detection_accuracy = accuracy_score(
    y_test_detection,
    detection_predictions
)

detection_precision = precision_score(
    y_test_detection,
    detection_predictions,
    zero_division=0
)

detection_recall = recall_score(
    y_test_detection,
    detection_predictions,
    zero_division=0
)

detection_f1 = f1_score(
    y_test_detection,
    detection_predictions,
    zero_division=0
)


print("\n================================")
print("DECISION TREE - FAULT DETECTION")
print("================================")

print(f"Accuracy : {detection_accuracy:.4f}")
print(f"Precision: {detection_precision:.4f}")
print(f"Recall   : {detection_recall:.4f}")
print(f"F1 Score : {detection_f1:.4f}")

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test_detection,
        detection_predictions
    )
)


# ------------------------------------------------
# Decision Tree - Fault Classification
# ------------------------------------------------

print("\nTraining Decision Tree for fault classification...")

# Only fault windows have valid fault labels
train_fault_mask = y_train_fault != -1
test_fault_mask = y_test_fault != -1

X_train_fault = X_train_flat[train_fault_mask]
X_test_fault = X_test_flat[test_fault_mask]

y_train_fault_valid = y_train_fault[train_fault_mask]
y_test_fault_valid = y_test_fault[test_fault_mask]


classification_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=20
)

classification_model.fit(
    X_train_fault,
    y_train_fault_valid
)

classification_predictions = classification_model.predict(
    X_test_fault
)


# ------------------------------------------------
# Classification metrics
# ------------------------------------------------

classification_accuracy = accuracy_score(
    y_test_fault_valid,
    classification_predictions
)

classification_precision = precision_score(
    y_test_fault_valid,
    classification_predictions,
    average="macro",
    zero_division=0
)

classification_recall = recall_score(
    y_test_fault_valid,
    classification_predictions,
    average="macro",
    zero_division=0
)

classification_f1 = f1_score(
    y_test_fault_valid,
    classification_predictions,
    average="macro",
    zero_division=0
)


print("\n================================")
print("DECISION TREE - FAULT CLASSIFICATION")
print("================================")

print(f"Accuracy : {classification_accuracy:.4f}")
print(f"Precision: {classification_precision:.4f}")
print(f"Recall   : {classification_recall:.4f}")
print(f"F1 Score : {classification_f1:.4f}")

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test_fault_valid,
        classification_predictions,
        labels=[0, 1, 2, 3]
    )
)


# ------------------------------------------------
# Save results
# ------------------------------------------------

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

with open(
    os.path.join(
        RESULTS_DIR,
        "decision_tree_results.txt"
    ),
    "w"
) as f:

    f.write("Decision Tree Results\n")
    f.write("====================\n\n")

    f.write("Fault Detection\n")
    f.write(
        f"Accuracy: {detection_accuracy:.4f}\n"
    )
    f.write(
        f"Precision: {detection_precision:.4f}\n"
    )
    f.write(
        f"Recall: {detection_recall:.4f}\n"
    )
    f.write(
        f"F1 Score: {detection_f1:.4f}\n"
    )

    f.write("\nFault Classification\n")
    f.write(
        f"Accuracy: {classification_accuracy:.4f}\n"
    )
    f.write(
        f"Precision: {classification_precision:.4f}\n"
    )
    f.write(
        f"Recall: {classification_recall:.4f}\n"
    )
    f.write(
        f"F1 Score: {classification_f1:.4f}\n"
    )


print("\nDecision Tree evaluation complete.")
print(
    "Results saved to:",
    "results/decision_tree_results.txt"
)