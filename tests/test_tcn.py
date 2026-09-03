import torch
from src.models.tcn import TCN


# Create the model
model = TCN(
    input_channels=48,
    num_classes=4
)


# Create dummy TCN input
# Batch size = 1
# Channels = 48
# Time steps = 256
x = torch.randn(1, 48, 256)


# Forward pass
detection_output, classification_output = model(x)


# Display shapes
print("Input shape:", x.shape)
print("Detection output shape:", detection_output.shape)
print("Classification output shape:", classification_output.shape)


# Verify expected shapes
assert detection_output.shape == (1, 2)
assert classification_output.shape == (1, 4)


print("\nTCN architecture test passed successfully.")