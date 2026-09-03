import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.relu = nn.ReLU()

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):

        residual = self.residual(x)

        x = self.conv1(x)

        # Remove future information introduced by padding
        x = x[:, :, :residual.size(2)]

        x = self.relu(x)

        x = self.conv2(x)

        x = x[:, :, :residual.size(2)]

        x = self.relu(x)

        return x + residual


class TCN(nn.Module):

    def __init__(
        self,
        input_channels=48,
        num_classes=4
    ):
        super().__init__()

        self.block1 = ResidualBlock(
            input_channels,
            64,
            kernel_size=3,
            dilation=1
        )

        self.block2 = ResidualBlock(
            64,
            64,
            kernel_size=3,
            dilation=2
        )

        self.block3 = ResidualBlock(
            64,
            128,
            kernel_size=3,
            dilation=4
        )

        self.block4 = ResidualBlock(
            128,
            128,
            kernel_size=3,
            dilation=8
        )

        self.detection_head = nn.Linear(
            128,
            2
        )

        self.classification_head = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global temporal pooling
        features = x.mean(dim=2)

        detection_output = self.detection_head(features)

        classification_output = self.classification_head(features)

        return detection_output, classification_output