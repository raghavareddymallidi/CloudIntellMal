"""
1D CNN Baseline for Microgrid Fault Detection and Classification.

Architecture:
    Input (48 channels, 256 timesteps)
        → Conv1d Block 1 (64 filters, kernel=7) → ReLU → MaxPool
        → Conv1d Block 2 (128 filters, kernel=5) → ReLU → MaxPool
        → Conv1d Block 3 (128 filters, kernel=3) → ReLU
        → Global Average Pooling
        → Detection Head (2 classes)
        → Classification Head (4 classes)

This provides a comparison between ordinary convolutional processing
and the dilated temporal processing of the TCN.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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
MODEL_DIR = "models"
RESULTS_DIR = "results"


# ------------------------------------------------
# CNN Model
# ------------------------------------------------

class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network with dual output heads
    for fault detection and classification.
    """

    def __init__(
        self,
        input_channels=48,
        num_classes=4,
        dropout=0.2
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Detection head (binary: fault / no fault)
        self.detection_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        # Classification head (multi-class)
        self.classification_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channels, timesteps)

        Returns:
            detection_output: (batch, 2)
            classification_output: (batch, num_classes)
        """

        x = self.features(x)

        # Global average pooling over temporal dimension
        features = x.mean(dim=2)

        detection_output = self.detection_head(features)
        classification_output = self.classification_head(features)

        return detection_output, classification_output




def run_training():
    # ------------------------------------------------
    # Device & Training
    # ------------------------------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)


    # ------------------------------------------------
    # Load data
    # ------------------------------------------------

    print("\nLoading processed data...")

    train_data = np.load(os.path.join(DATA_DIR, "train.npz"))
    val_data = np.load(os.path.join(DATA_DIR, "validation.npz"))
    test_data = np.load(os.path.join(DATA_DIR, "test.npz"))

    X_train = torch.tensor(train_data["X"], dtype=torch.float32)
    y_train_det = torch.tensor(train_data["detection_labels"], dtype=torch.long)
    y_train_fault = torch.tensor(train_data["fault_types"], dtype=torch.long)

    X_val = torch.tensor(val_data["X"], dtype=torch.float32)
    y_val_det = torch.tensor(val_data["detection_labels"], dtype=torch.long)
    y_val_fault = torch.tensor(val_data["fault_types"], dtype=torch.long)

    X_test = torch.tensor(test_data["X"], dtype=torch.float32)
    y_test_det = test_data["detection_labels"]
    y_test_fault = test_data["fault_types"]

    print("Training X:", X_train.shape)
    print("Validation X:", X_val.shape)
    print("Test X:", X_test.shape)


    # ------------------------------------------------
    # DataLoaders
    # ------------------------------------------------

    BATCH_SIZE = 64

    train_loader = DataLoader(
        TensorDataset(X_train, y_train_det, y_train_fault),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val_det, y_val_fault),
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # ------------------------------------------------
    # Model
    # ------------------------------------------------

    model = CNN1D(
        input_channels=48,
        num_classes=4,
        dropout=0.2
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nCNN parameters: {total_params:,}")


    # ------------------------------------------------
    # Training
    # ------------------------------------------------

    EPOCHS = 20
    LEARNING_RATE = 0.001

    detection_loss_fn = nn.CrossEntropyLoss()
    classification_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 10
    epochs_without_improvement = 0

    print(f"\nStarting CNN training for {EPOCHS} epochs...\n")

    training_start = time.time()

    for epoch in range(EPOCHS):

        epoch_start = time.time()

        # Train
        model.train()
        total_train_loss = 0.0
        correct_det = 0
        total_det = 0

        for X, y_det, y_fault in train_loader:

            X = X.to(device)
            y_det = y_det.to(device)
            y_fault = y_fault.to(device)

            optimizer.zero_grad()

            det_out, cls_out = model(X)

            det_loss = detection_loss_fn(det_out, y_det)
            cls_loss = classification_loss_fn(cls_out, y_fault)

            loss = det_loss + cls_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()

            preds = torch.argmax(det_out, dim=1)
            correct_det += (preds == y_det).sum().item()
            total_det += len(y_det)

        avg_train_loss = total_train_loss / len(train_loader)
        train_det_acc = correct_det / total_det

        # Validate
        model.eval()
        total_val_loss = 0.0
        correct_val_det = 0
        total_val_det = 0

        with torch.no_grad():
            for X, y_det, y_fault in val_loader:
                X = X.to(device)
                y_det = y_det.to(device)
                y_fault = y_fault.to(device)

                det_out, cls_out = model(X)
                det_loss = detection_loss_fn(det_out, y_det)
                cls_loss = classification_loss_fn(cls_out, y_fault)
                loss = det_loss + cls_loss

                total_val_loss += loss.item()

                preds = torch.argmax(det_out, dim=1)
                correct_val_det += (preds == y_det).sum().item()
                total_val_det += len(y_det)

        avg_val_loss = total_val_loss / len(val_loader)
        val_det_acc = correct_val_det / total_val_det

        scheduler.step()
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Train Det Acc: {train_det_acc:.4f} | "
            f"Val Det Acc: {val_det_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": avg_val_loss,
                    "val_det_acc": val_det_acc,
                    "hyperparameters": {
                        "input_channels": 48,
                        "num_classes": 4,
                        "dropout": 0.2,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                    }
                },
                os.path.join(MODEL_DIR, "cnn_best.pt")
            )
            print(f"  → Best model saved (epoch {epoch + 1})")
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs)"
                )
                break


    total_training_time = time.time() - training_start

    print(f"\nCNN training complete in {total_training_time:.1f}s")
    print(f"Best epoch: {best_epoch}")


    # ------------------------------------------------
    # Evaluation
    # ------------------------------------------------

    print("\nEvaluating CNN on test set...")

    # Load best model
    cnn_checkpoint = torch.load(
        os.path.join(MODEL_DIR, "cnn_best.pt"),
        map_location=device,
        weights_only=True
    )
    if "model_state_dict" in cnn_checkpoint:
        model.load_state_dict(cnn_checkpoint["model_state_dict"])
    else:
        model.load_state_dict(cnn_checkpoint)

    model.eval()

    all_det_preds = []
    all_cls_preds = []

    inference_start = time.time()

    with torch.no_grad():
        for start in range(0, len(X_test), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X_test))
            X_batch = X_test[start:end].to(device)

            det_out, cls_out = model(X_batch)

            all_det_preds.extend(
                torch.argmax(det_out, dim=1).cpu().numpy()
            )
            all_cls_preds.extend(
                torch.argmax(cls_out, dim=1).cpu().numpy()
            )

    total_inference_time = time.time() - inference_start

    all_det_preds = np.array(all_det_preds)
    all_cls_preds = np.array(all_cls_preds)


    # Detection metrics
    det_accuracy = accuracy_score(y_test_det, all_det_preds)
    det_precision = precision_score(
        y_test_det, all_det_preds, average="binary", zero_division=0
    )
    det_recall = recall_score(
        y_test_det, all_det_preds, average="binary", zero_division=0
    )
    det_f1 = f1_score(
        y_test_det, all_det_preds, average="binary", zero_division=0
    )
    det_cm = confusion_matrix(y_test_det, all_det_preds)


    # Classification metrics (fault windows only)
    fault_mask = y_test_fault != -1
    y_fault_true = y_test_fault[fault_mask]
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
    cls_cm = confusion_matrix(
        y_fault_true, y_fault_pred, labels=[0, 1, 2, 3]
    )

    per_sample_ms = total_inference_time / len(X_test) * 1000


    # ------------------------------------------------
    # Print results
    # ------------------------------------------------

    print("\n" + "=" * 50)
    print("CNN BASELINE — FAULT DETECTION")
    print("=" * 50)

    print(f"Accuracy : {det_accuracy * 100:.2f}%")
    print(f"Precision: {det_precision * 100:.2f}%")
    print(f"Recall   : {det_recall * 100:.2f}%")
    print(f"F1 Score : {det_f1 * 100:.2f}%")
    print(f"\nConfusion Matrix:\n{det_cm}")

    print("\n" + "=" * 50)
    print("CNN BASELINE — FAULT CLASSIFICATION")
    print("=" * 50)

    print(f"Accuracy : {cls_accuracy * 100:.2f}%")
    print(f"Precision: {cls_precision * 100:.2f}%")
    print(f"Recall   : {cls_recall * 100:.2f}%")
    print(f"F1 Score : {cls_f1 * 100:.2f}%")
    print(f"\nConfusion Matrix:\n{cls_cm}")

    print(f"\nTraining time: {total_training_time:.2f}s")
    print(f"Inference time per sample: {per_sample_ms:.3f}ms")


    # ------------------------------------------------
    # Save results
    # ------------------------------------------------

    results = {
        "model": "CNN",
        "parameters": int(total_params),
        "detection": {
            "accuracy": float(det_accuracy),
            "precision": float(det_precision),
            "recall": float(det_recall),
            "f1_score": float(det_f1),
            "confusion_matrix": det_cm.tolist()
        },
        "classification": {
            "accuracy": float(cls_accuracy),
            "precision_macro": float(cls_precision),
            "recall_macro": float(cls_recall),
            "f1_macro": float(cls_f1),
            "confusion_matrix": cls_cm.tolist()
        },
        "timing": {
            "training_time_s": float(total_training_time),
            "inference_per_sample_ms": float(per_sample_ms),
            "best_epoch": best_epoch
        }
    }

    # Save JSON
    with open(
        os.path.join(RESULTS_DIR, "cnn_results.json"), "w"
    ) as f:
        json.dump(results, f, indent=2)

    # Save text report
    with open(
        os.path.join(RESULTS_DIR, "cnn_results.txt"), "w"
    ) as f:

        f.write("CNN BASELINE RESULTS\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Best epoch: {best_epoch}\n\n")

        f.write("Fault Detection\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy : {det_accuracy * 100:.2f}%\n")
        f.write(f"Precision: {det_precision * 100:.2f}%\n")
        f.write(f"Recall   : {det_recall * 100:.2f}%\n")
        f.write(f"F1 Score : {det_f1 * 100:.2f}%\n")
        f.write(f"\nConfusion Matrix:\n{det_cm}\n")

        f.write("\nFault Classification\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy : {cls_accuracy * 100:.2f}%\n")
        f.write(f"Precision: {cls_precision * 100:.2f}%\n")
        f.write(f"Recall   : {cls_recall * 100:.2f}%\n")
        f.write(f"F1 Score : {cls_f1 * 100:.2f}%\n")
        f.write(f"\nConfusion Matrix:\n{cls_cm}\n")

        f.write(f"\nTraining time: {total_training_time:.2f}s\n")
        f.write(f"Inference per sample: {per_sample_ms:.3f}ms\n")


    print("\n" + "=" * 50)
    print("CNN BASELINE COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {RESULTS_DIR}/cnn_results.txt")
    print(f"Results saved to: {RESULTS_DIR}/cnn_results.json")


if __name__ == "__main__":
    run_training()
