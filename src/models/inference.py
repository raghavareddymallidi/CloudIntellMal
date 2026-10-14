"""
TCN Inference Pipeline

Provides a clean interface for running fault detection and
classification on new microgrid waveform data.

Usage:
    from src.models.inference import FaultDetector

    detector = FaultDetector()
    result = detector.predict(waveform_array)
"""

import os
import time
import numpy as np
import torch

from src.models.tcn import TCN
from src.models.config import (
    MODEL_DIR,
    DATA_DIR,
    INPUT_CHANNELS,
    NUM_FAULT_CLASSES,
    TCN_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_DROPOUT,
    FAULT_TYPE_NAMES
)


class FaultDetector:
    """
    Fault detection and classification inference engine.

    Loads the trained TCN model and normalization statistics,
    then provides a predict() method for new waveform data.
    """

    def __init__(
        self,
        model_path=None,
        normalization_path=None
    ):

        if model_path is None:
            model_path = os.path.join(
                MODEL_DIR, "tcn_best.pt"
            )

        if normalization_path is None:
            normalization_path = os.path.join(
                DATA_DIR, "normalization.npz"
            )

        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load model
        self.model = TCN(
            input_channels=INPUT_CHANNELS,
            num_classes=NUM_FAULT_CLASSES,
            tcn_channels=TCN_CHANNELS,
            kernel_size=TCN_KERNEL_SIZE,
            dropout=TCN_DROPOUT
        )

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )
            self.model_epoch = checkpoint.get("epoch", None)
            self.model_val_loss = checkpoint.get("val_loss", None)
        else:
            self.model.load_state_dict(checkpoint)
            self.model_epoch = None
            self.model_val_loss = None

        self.model = self.model.to(self.device)
        self.model.eval()

        # Load normalization statistics
        norm = np.load(normalization_path)
        self.norm_mean = norm["mean"]
        self.norm_std = norm["std"]

        self.fault_type_names = FAULT_TYPE_NAMES

    def preprocess(self, waveform):
        """
        Normalize raw waveform using training statistics.

        Args:
            waveform: numpy array of shape (channels, timesteps)

        Returns:
            Normalized waveform
        """

        normalized = (
            waveform - self.norm_mean[:, None]
        ) / self.norm_std[:, None]

        return normalized.astype(np.float32)

    def predict(self, waveform):
        """
        Run fault detection and classification on a waveform.

        Args:
            waveform: numpy array of shape (channels, timesteps)
                      or (batch, channels, timesteps)

        Returns:
            Dictionary with prediction results
        """

        start_time = time.time()

        # Handle single sample
        if waveform.ndim == 2:
            waveform = waveform[np.newaxis, :]

        # Preprocess each sample
        processed = np.stack([
            self.preprocess(w) for w in waveform
        ])

        # Convert to tensor
        x = torch.tensor(
            processed,
            dtype=torch.float32
        ).to(self.device)

        # Inference
        with torch.no_grad():
            det_out, cls_out = self.model(x)

        # Probabilities
        det_probs = torch.softmax(det_out, dim=1).cpu().numpy()
        cls_probs = torch.softmax(cls_out, dim=1).cpu().numpy()

        # Predictions
        det_preds = np.argmax(det_probs, axis=1)
        cls_preds = np.argmax(cls_probs, axis=1)

        inference_time = time.time() - start_time

        # Build results
        results = []

        for i in range(len(waveform)):

            is_fault = bool(det_preds[i] == 1)

            fault_type = int(cls_preds[i]) if is_fault else None
            fault_name = (
                self.fault_type_names.get(fault_type, "Unknown")
                if is_fault else "No Fault"
            )

            result = {
                "fault_detected": is_fault,
                "fault_type": fault_type,
                "fault_name": fault_name,
                "detection_confidence": float(
                    det_probs[i][det_preds[i]]
                ),
                "classification_confidence": float(
                    cls_probs[i][cls_preds[i]]
                ) if is_fault else None,
                "detection_probabilities": {
                    "no_fault": float(det_probs[i][0]),
                    "fault": float(det_probs[i][1])
                },
                "classification_probabilities": {
                    self.fault_type_names.get(j, f"Type {j}"):
                    float(cls_probs[i][j])
                    for j in range(NUM_FAULT_CLASSES)
                },
                "inference_time_ms": float(
                    inference_time / len(waveform) * 1000
                )
            }

            results.append(result)

        # Return single result for single input
        if len(results) == 1:
            return results[0]

        return results

    def get_model_info(self):
        """Return model metadata."""

        return {
            "model_type": "TCN",
            "input_channels": INPUT_CHANNELS,
            "num_classes": NUM_FAULT_CLASSES,
            "tcn_channels": TCN_CHANNELS,
            "kernel_size": TCN_KERNEL_SIZE,
            "dropout": TCN_DROPOUT,
            "parameters": sum(
                p.numel() for p in self.model.parameters()
            ),
            "receptive_field": self.model.get_receptive_field(),
            "training_epoch": self.model_epoch,
            "training_val_loss": self.model_val_loss,
            "device": str(self.device),
            "fault_types": self.fault_type_names
        }


if __name__ == "__main__":

    # Quick test with random data
    detector = FaultDetector()

    print("Model info:")
    info = detector.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Test with random waveform
    waveform = np.random.randn(48, 256).astype(np.float32)
    result = detector.predict(waveform)

    print("\nPrediction result:")
    for k, v in result.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv:.4f}")
        else:
            print(f"  {k}: {v}")
