import os
import time
import numpy as np
from scipy.stats import skew, kurtosis

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------
# Paths
# ------------------------------------------------

from src.models.config import DATA_DIR, RESULTS_DIR


# ------------------------------------------------
# Feature extraction
# ------------------------------------------------

def extract_features(X):
    """
    Extract statistical features from windowed waveforms.

    Input shape:  (samples, channels, time)
    Output shape: (samples, channels * 8)

    For each channel, compute:
        1. Mean
        2. Standard deviation
        3. Min
        4. Max
        5. RMS (root mean square)
        6. Peak-to-peak amplitude
        7. Skewness
        8. Kurtosis
    """

    num_samples = X.shape[0]
    num_channels = X.shape[1]

    features = np.zeros(
        (num_samples, num_channels * 8),
        dtype=np.float32
    )

    for i in range(num_channels):

        channel = X[:, i, :]

        offset = i * 8

        # Mean
        features[:, offset + 0] = np.mean(
            channel, axis=1
        )

        # Standard deviation
        features[:, offset + 1] = np.std(
            channel, axis=1
        )

        # Min
        features[:, offset + 2] = np.min(
            channel, axis=1
        )

        # Max
        features[:, offset + 3] = np.max(
            channel, axis=1
        )

        # RMS
        features[:, offset + 4] = np.sqrt(
            np.mean(channel ** 2, axis=1)
        )

        # Peak-to-peak
        features[:, offset + 5] = (
            np.max(channel, axis=1) -
            np.min(channel, axis=1)
        )

        # Skewness
        features[:, offset + 6] = skew(
            channel, axis=1
        )

        # Kurtosis
        features[:, offset + 7] = kurtosis(
            channel, axis=1
        )

    # Replace NaN values from constant channels
    # (skewness/kurtosis undefined for zero variance)
    np.nan_to_num(features, copy=False, nan=0.0)

    return features


# ------------------------------------------------
# Metrics helper
# ------------------------------------------------

def print_metrics(name, y_true, y_pred):

    print(f"\n{name}")
    print("-" * 40)

    print(
        f"Accuracy : "
        f"{accuracy_score(y_true, y_pred) * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%"
    )

    print(
        f"F1 Score : "
        f"{f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100:.2f}%"
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# ------------------------------------------------
# Load processed data
# ------------------------------------------------

print("Loading processed data...")

train = np.load(
    os.path.join(DATA_DIR, "train.npz")
)

test = np.load(
    os.path.join(DATA_DIR, "test.npz")
)

X_train_raw = train["X"]
y_det_train = train["detection_labels"]
y_cls_train = train["fault_types"]

X_test_raw = test["X"]
y_det_test = test["detection_labels"]
y_cls_test = test["fault_types"]


print("Raw training data:", X_train_raw.shape)
print("Raw test data:", X_test_raw.shape)


# ------------------------------------------------
# Extract statistical features
# ------------------------------------------------

print("\nExtracting statistical features...")

feature_start = time.time()

X_train = extract_features(X_train_raw)
X_test = extract_features(X_test_raw)

feature_time = time.time() - feature_start

print(f"Feature extraction completed in {feature_time:.2f}s")
print("Training features:", X_train.shape)
print("Test features:", X_test.shape)


# ------------------------------------------------
# Normalize features
# ------------------------------------------------

print("\nNormalizing features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------------
# 1. Fault Detection (SVM)
# --------------------------------------------------

print("\nTraining SVM for fault detection...")

det_start = time.time()

svm_detection = LinearSVC(
    C=1.0,
    max_iter=10000,
    dual=False,
    random_state=42
)

svm_detection.fit(X_train, y_det_train)

det_train_time = time.time() - det_start

print(f"Detection SVM trained in {det_train_time:.2f}s")

det_pred = svm_detection.predict(X_test)

print_metrics(
    "SVM Fault Detection",
    y_det_test,
    det_pred
)


# --------------------------------------------------
# 2. Fault Classification (SVM)
# --------------------------------------------------

print("\nTraining SVM for fault classification...")

# Only fault windows have valid fault-type labels
train_fault_mask = y_cls_train != -1
test_fault_mask = y_cls_test != -1

X_train_fault = X_train[train_fault_mask]
y_train_fault = y_cls_train[train_fault_mask]

X_test_fault = X_test[test_fault_mask]
y_test_fault = y_cls_test[test_fault_mask]

print(
    f"Classification samples: "
    f"{len(X_train_fault)} train, "
    f"{len(X_test_fault)} test"
)

cls_start = time.time()

svm_classification = LinearSVC(
    C=1.0,
    max_iter=10000,
    dual=False,
    random_state=42
)

svm_classification.fit(
    X_train_fault,
    y_train_fault
)

cls_train_time = time.time() - cls_start

print(f"Classification SVM trained in {cls_train_time:.2f}s")

cls_pred = svm_classification.predict(X_test_fault)

print_metrics(
    "SVM Fault Classification",
    y_test_fault,
    cls_pred
)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\nClassification report:")

print(
    classification_report(
        y_test_fault,
        cls_pred,
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


# --------------------------------------------------
# Save results
# --------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

with open(
    os.path.join(RESULTS_DIR, "svm_results.txt"),
    "w"
) as f:

    f.write("SVM BASELINE RESULTS\n")
    f.write("====================\n\n")

    f.write("Feature Extraction\n")
    f.write("------------------\n")
    f.write(
        "Method: Statistical features "
        "(mean, std, min, max, RMS, "
        "peak-to-peak, skewness, kurtosis)\n"
    )
    f.write(
        f"Raw input shape: "
        f"{X_train_raw.shape}\n"
    )
    f.write(
        f"Feature shape: "
        f"{X_train.shape}\n"
    )
    f.write(
        f"Feature extraction time: "
        f"{feature_time:.2f}s\n"
    )

    f.write("\n\nFault Detection\n")
    f.write("----------------\n")
    f.write(
        f"Accuracy : "
        f"{accuracy_score(y_det_test, det_pred) * 100:.2f}%\n"
    )
    f.write(
        f"Precision: "
        f"{precision_score(y_det_test, det_pred, average='weighted', zero_division=0) * 100:.2f}%\n"
    )
    f.write(
        f"Recall   : "
        f"{recall_score(y_det_test, det_pred, average='weighted', zero_division=0) * 100:.2f}%\n"
    )
    f.write(
        f"F1 Score : "
        f"{f1_score(y_det_test, det_pred, average='weighted', zero_division=0) * 100:.2f}%\n"
    )
    f.write(
        f"Training time: {det_train_time:.2f}s\n"
    )
    f.write("\nConfusion Matrix:\n")
    f.write(
        str(confusion_matrix(y_det_test, det_pred))
    )

    f.write("\n\n\nFault Classification\n")
    f.write("--------------------\n")
    f.write(
        f"Accuracy : "
        f"{accuracy_score(y_test_fault, cls_pred) * 100:.2f}%\n"
    )
    f.write(
        f"Precision: "
        f"{precision_score(y_test_fault, cls_pred, average='weighted', zero_division=0) * 100:.2f}%\n"
    )
    f.write(
        f"Recall   : "
        f"{recall_score(y_test_fault, cls_pred, average='weighted', zero_division=0) * 100:.2f}%\n"
    )
    f.write(
        f"F1 Score : "
        f"{f1_score(y_test_fault, cls_pred, average='weighted', zero_division=0) * 100:.2f}%\n"
    )
    f.write(
        f"Training time: {cls_train_time:.2f}s\n"
    )
    f.write("\nConfusion Matrix:\n")
    f.write(
        str(confusion_matrix(y_test_fault, cls_pred))
    )

    f.write("\n\n\nClassification Report:\n")
    f.write(
        classification_report(
            y_test_fault,
            cls_pred,
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


print("\n================================")
print("SVM BASELINE COMPLETE")
print("================================")

print(
    "\nResults saved to:",
    os.path.join(RESULTS_DIR, "svm_results.txt")
)