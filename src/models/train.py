"""
Enhanced TCN Training Pipeline

Features:
    - Cosine annealing learning rate scheduler
    - Gradient clipping
    - Early stopping
    - Per-epoch training/validation metrics
    - Model checkpointing with metadata
    - Training history logging to JSON
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRADIENT_CLIP,
    EARLY_STOPPING_PATIENCE,
    LR_MIN
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
# Load processed data
# ------------------------------------------------

print("\nLoading training data...")

train_data = np.load(
    os.path.join(DATA_DIR, "train.npz")
)

val_data = np.load(
    os.path.join(DATA_DIR, "validation.npz")
)

X_train = torch.tensor(
    train_data["X"],
    dtype=torch.float32
)

y_train_detection = torch.tensor(
    train_data["detection_labels"],
    dtype=torch.long
)

y_train_fault = torch.tensor(
    train_data["fault_types"],
    dtype=torch.long
)

X_val = torch.tensor(
    val_data["X"],
    dtype=torch.float32
)

y_val_detection = torch.tensor(
    val_data["detection_labels"],
    dtype=torch.long
)

y_val_fault = torch.tensor(
    val_data["fault_types"],
    dtype=torch.long
)

print("Training X:", X_train.shape)
print("Validation X:", X_val.shape)


# ------------------------------------------------
# DataLoaders
# ------------------------------------------------

train_dataset = TensorDataset(
    X_train,
    y_train_detection,
    y_train_fault
)

val_dataset = TensorDataset(
    X_val,
    y_val_detection,
    y_val_fault
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ------------------------------------------------
# Model
# ------------------------------------------------

model = TCN(
    input_channels=INPUT_CHANNELS,
    num_classes=NUM_FAULT_CLASSES,
    tcn_channels=TCN_CHANNELS,
    kernel_size=TCN_KERNEL_SIZE,
    dropout=TCN_DROPOUT
)

model = model.to(device)

# Print model summary
total_params = sum(
    p.numel() for p in model.parameters()
)
trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(f"\nModel parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Receptive field: {model.get_receptive_field()} samples")


# ------------------------------------------------
# Loss functions
# ------------------------------------------------

detection_loss_fn = nn.CrossEntropyLoss()

classification_loss_fn = nn.CrossEntropyLoss(
    ignore_index=-1
)


# ------------------------------------------------
# Optimizer and scheduler
# ------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=LR_MIN
)


# ------------------------------------------------
# Training loop
# ------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

best_val_loss = float("inf")
epochs_without_improvement = 0
best_epoch = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "train_det_acc": [],
    "val_det_acc": [],
    "learning_rate": [],
    "epoch_time": []
}

print("\nStarting training...")
print(f"Epochs: {EPOCHS}")
print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Batch size: {BATCH_SIZE}")
print()

training_start = time.time()


for epoch in range(EPOCHS):

    epoch_start = time.time()

    # ------------------------------------------------
    # Train
    # ------------------------------------------------

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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        optimizer.step()

        total_train_loss += loss.item()

        # Detection accuracy
        preds = torch.argmax(det_out, dim=1)
        correct_det += (preds == y_det).sum().item()
        total_det += len(y_det)

    avg_train_loss = total_train_loss / len(train_loader)
    train_det_acc = correct_det / total_det

    # ------------------------------------------------
    # Validate
    # ------------------------------------------------

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

    # Update learning rate
    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]
    epoch_time = time.time() - epoch_start

    # ------------------------------------------------
    # Log
    # ------------------------------------------------

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_det_acc"].append(train_det_acc)
    history["val_det_acc"].append(val_det_acc)
    history["learning_rate"].append(current_lr)
    history["epoch_time"].append(epoch_time)

    print(
        f"Epoch {epoch + 1:3d}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Train Det Acc: {train_det_acc:.4f} | "
        f"Val Det Acc: {val_det_acc:.4f} | "
        f"LR: {current_lr:.6f} | "
        f"Time: {epoch_time:.1f}s"
    )

    # ------------------------------------------------
    # Save best model
    # ------------------------------------------------

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        model_path = os.path.join(
            MODEL_DIR, "tcn_best.pt"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_loss": avg_val_loss,
            "val_det_acc": val_det_acc,
            "hyperparameters": {
                "input_channels": INPUT_CHANNELS,
                "num_classes": NUM_FAULT_CLASSES,
                "tcn_channels": TCN_CHANNELS,
                "kernel_size": TCN_KERNEL_SIZE,
                "dropout": TCN_DROPOUT,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY
            }
        }

        torch.save(checkpoint, model_path)

        print(f"  → Best model saved (epoch {epoch + 1})")

    else:
        epochs_without_improvement += 1

    # ------------------------------------------------
    # Early stopping
    # ------------------------------------------------

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch + 1} "
            f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)"
        )
        break


# ------------------------------------------------
# Training summary
# ------------------------------------------------

total_training_time = time.time() - training_start

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"\nTotal training time: {total_training_time:.1f}s")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best val detection accuracy: {history['val_det_acc'][best_epoch - 1]:.4f}")

model_path = os.path.join(MODEL_DIR, "tcn_best.pt")
print(f"Model saved to: {model_path}")


# ------------------------------------------------
# Save training history
# ------------------------------------------------

history_file = os.path.join(
    RESULTS_DIR,
    "training_history.json"
)

with open(history_file, "w") as f:
    json.dump(history, f, indent=2)

print(f"Training history saved to: {history_file}")