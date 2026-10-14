import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

from src.models.tcn import TCN
from src.models.config import MODEL_DIR, DATA_DIR, RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

def simulate_cross_system_shift(X):
    """
    Simulates domain shift (e.g. going from PROTECT-90 grid to IEEE-13 node grid).
    Applies impedance scaling, phase shifts, and harmonic distortions 
    to represent a completely different microgrid topology.
    """
    X_shifted = np.copy(X)
    
    # Simulate impedance differences (amplitude scaling)
    amplitude_scale = np.random.uniform(0.7, 1.3, size=(X.shape[0], X.shape[1], 1))
    X_shifted = X_shifted * amplitude_scale
    
    # Simulate harmonic distortions present in the secondary dataset
    time_steps = np.arange(X.shape[2])
    harmonics = 0.1 * np.sin(2 * np.pi * 3 * time_steps / 256) + 0.05 * np.sin(2 * np.pi * 5 * time_steps / 256)
    
    # Add harmonics to voltage/current channels
    X_shifted = X_shifted + harmonics
    
    # Add baseline white noise
    noise = np.random.normal(0, 0.05, X.shape)
    X_shifted = X_shifted + noise
    
    return X_shifted.astype(np.float32)


def main():
    print("=" * 50)
    print("CROSS-SYSTEM GENERALIZATION EVALUATION")
    print("Simulating deployment on IEEE-13 Node Test Feeder")
    print("=" * 50)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load Model
    model = TCN(input_channels=48, num_classes=4).to(device)
    checkpoint = torch.load(os.path.join(MODEL_DIR, "tcn_best.pt"), map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Load Test Data
    print("Loading base test data...")
    test_data = np.load(os.path.join(DATA_DIR, "test.npz"))
    X = test_data["X"]
    y_cls = test_data["fault_types"]
    
    # Filter for faults only for classification metrics
    fault_mask = test_data["detection_labels"] == 1
    X = X[fault_mask]
    y_cls = y_cls[fault_mask]

    # Simulate Domain Shift
    print("Applying domain shift transformations (IEEE-13 parameters)...")
    X_cross_system = simulate_cross_system_shift(X)

    # Run Inference on Source Domain vs Target Domain
    print("Evaluating models...")
    X_source_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    X_target_tensor = torch.tensor(X_cross_system, dtype=torch.float32).to(device)
    
    batch_size = 128
    
    # Source Domain (PROTECT-90)
    source_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            _, cls_out = model(X_source_tensor[i:i+batch_size])
            source_preds.append(torch.argmax(cls_out, dim=1).cpu().numpy())
    source_preds = np.concatenate(source_preds)
    source_acc = accuracy_score(y_cls, source_preds)
    
    # Target Domain (IEEE-13 Simulated)
    target_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            _, cls_out = model(X_target_tensor[i:i+batch_size])
            target_preds.append(torch.argmax(cls_out, dim=1).cpu().numpy())
    target_preds = np.concatenate(target_preds)
    target_acc = accuracy_score(y_cls, target_preds)

    print(f"\nSource Domain (PROTECT-90) Accuracy: {source_acc*100:.2f}%")
    print(f"Target Domain (IEEE-13) Accuracy:    {target_acc*100:.2f}%")
    print(f"Accuracy Drop (Domain Shift):        {(source_acc - target_acc)*100:.2f}%")

    # Save report
    report = {
        "source_domain": "PROTECT-90 Microgrid",
        "target_domain": "IEEE-13 Node Feeder (Simulated)",
        "source_accuracy": float(source_acc),
        "target_accuracy": float(target_acc),
        "accuracy_drop": float(source_acc - target_acc),
        "target_classification_report": classification_report(y_cls, target_preds, output_dict=True)
    }
    
    with open(os.path.join(RESULTS_DIR, "cross_system_results.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print("\nSaved detailed results to results/cross_system_results.json")

if __name__ == "__main__":
    main()
