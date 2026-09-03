import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.tcn import TCN


# ------------------------------------------------
# Configuration
# ------------------------------------------------

DATA_DIR = "data/processed"
MODEL_DIR = "models"

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001


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
    input_channels=48,
    num_classes=4
)

model = model.to(device)


# ------------------------------------------------
# Loss functions
# ------------------------------------------------

detection_loss_function = nn.CrossEntropyLoss()

classification_loss_function = nn.CrossEntropyLoss(
    ignore_index=-1
)


# ------------------------------------------------
# Optimizer
# ------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ------------------------------------------------
# Training
# ------------------------------------------------

best_val_loss = float("inf")

os.makedirs(MODEL_DIR, exist_ok=True)

print("\nStarting training...")


for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    for X, y_detection, y_fault in train_loader:

        X = X.to(device)
        y_detection = y_detection.to(device)
        y_fault = y_fault.to(device)

        optimizer.zero_grad()

        detection_output, classification_output = model(X)

        detection_loss = detection_loss_function(
            detection_output,
            y_detection
        )

        classification_loss = classification_loss_function(
            classification_output,
            y_fault
        )

        total_loss = (
            detection_loss +
            classification_loss
        )

        total_loss.backward()

        optimizer.step()

        total_train_loss += total_loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )


    # ------------------------------------------------
    # Validation
    # ------------------------------------------------

    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for X, y_detection, y_fault in val_loader:

            X = X.to(device)
            y_detection = y_detection.to(device)
            y_fault = y_fault.to(device)

            detection_output, classification_output = model(X)

            detection_loss = detection_loss_function(
                detection_output,
                y_detection
            )

            classification_loss = classification_loss_function(
                classification_output,
                y_fault
            )

            total_loss = (
                detection_loss +
                classification_loss
            )

            total_val_loss += total_loss.item()

    average_val_loss = (
        total_val_loss / len(val_loader)
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Val Loss: {average_val_loss:.4f}"
    )


    # ------------------------------------------------
    # Save best model
    # ------------------------------------------------

    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss

        model_path = os.path.join(
            MODEL_DIR,
            "tcn_best.pt"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            "  Best model saved."
        )


print("\nTraining completed.")
print("Best validation loss:", best_val_loss)
print("Model saved to:", model_path)