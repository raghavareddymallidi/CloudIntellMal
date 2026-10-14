"""
Temporal Convolutional Network (TCN) for Microgrid Fault Detection
and Classification.

Architecture:
    Input (channels, timesteps)
        → Residual Block 1 (dilation=1)
        → Residual Block 2 (dilation=2)
        → Residual Block 3 (dilation=4)
        → Residual Block 4 (dilation=8)
        → Global Average Pooling
        → Detection Head (Fault / No Fault)
        → Classification Head (Fault Type)

Each Residual Block:
    CausalConv1d → WeightNorm → ReLU → Dropout
    → CausalConv1d → WeightNorm → ReLU → Dropout
    + 1×1 Residual Connection

Hyperparameter justification:
    - Kernel size 7: captures ~1ms per kernel at 6.4kHz
    - 4 blocks with dilations [1,2,4,8]: receptive field = 361 samples
      (~56ms, >1 full power cycle at 50Hz)
    - Channels [64,64,128,128]: progressive capacity increase
    - Dropout 0.2: moderate regularization
    - WeightNorm: stabilizes training (standard in TCN literature)
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Module):
    """
    Causal convolution that pads only on the left side
    to prevent information leakage from the future.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1
    ):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                dilation=dilation
            )
        )

    def forward(self, x):

        out = self.conv(x)

        # Remove the right-side padding to enforce causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]

        return out


class ResidualBlock(nn.Module):
    """
    TCN Residual Block with causal convolutions,
    weight normalization, and dropout.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout=0.2
    ):
        super().__init__()

        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation
        )

        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation
        )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 1x1 convolution for matching channel dimensions
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.relu_out = nn.ReLU()

    def forward(self, x):

        residual = self.residual(x)

        # First causal convolution
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second causal convolution
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        return self.relu_out(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network with dual output heads
    for fault detection and classification.

    Args:
        input_channels: Number of input signal channels (default: 48)
        num_classes: Number of fault type classes (default: 4)
        tcn_channels: List of channel sizes for each block
        kernel_size: Convolution kernel size (default: 7)
        dropout: Dropout rate (default: 0.2)
    """

    def __init__(
        self,
        input_channels=48,
        num_classes=4,
        tcn_channels=None,
        kernel_size=7,
        dropout=0.2
    ):
        super().__init__()

        if tcn_channels is None:
            tcn_channels = [64, 64, 128, 128]

        # Build TCN blocks with exponentially increasing dilation
        blocks = []
        prev_channels = input_channels

        for i, out_channels in enumerate(tcn_channels):

            dilation = 2 ** i

            blocks.append(
                ResidualBlock(
                    prev_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

            prev_channels = out_channels

        self.tcn = nn.Sequential(*blocks)

        # Feature dimension after global pooling
        feature_dim = tcn_channels[-1]

        # Fault detection head (binary: fault / no fault)
        self.detection_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        # Fault classification head (multi-class)
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channels, timesteps)

        Returns:
            detection_output: (batch, 2) — fault/no-fault logits
            classification_output: (batch, num_classes) — fault type logits
        """

        # TCN feature extraction
        x = self.tcn(x)

        # Global average pooling over temporal dimension
        features = x.mean(dim=2)

        # Dual prediction heads
        detection_output = self.detection_head(features)
        classification_output = self.classification_head(features)

        return detection_output, classification_output

    def get_receptive_field(self):
        """Calculate the receptive field of this TCN."""

        num_blocks = len(self.tcn)

        # For each block: rf contribution = 2 * (kernel_size - 1) * dilation
        # (2 conv layers per block)
        kernel_size = self.tcn[0].conv1.conv.kernel_size[0]

        total = 1
        for i in range(num_blocks):
            dilation = 2 ** i
            total += 2 * (kernel_size - 1) * dilation

        return total