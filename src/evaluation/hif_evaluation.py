import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.models.tcn import TCN
from src.models.config import MODEL_DIR, DATA_DIR, RAW_DATA_DIR, RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("=" * 50)
    print("HIGH-IMPEDANCE FAULT (HIF) EVALUATION")
    print("=" * 50)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    # Load Model
    model = TCN(input_channels=48, num_classes=4).to(device)
    checkpoint = torch.load(os.path.join(MODEL_DIR, "tcn_best.pt"), map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Load Test Data
    print("Loading test data...")
    try:
        test_data = np.load(os.path.join(DATA_DIR, "test.npz"))
        X = test_data["X"]
        y_det = test_data["detection_labels"]
        y_cls = test_data["fault_types"]
        episode_ids = test_data["episode_ids"]
    except KeyError:
        print("Error: test.npz does not contain episode_ids. Run create_windows.py again.")
        return

    # Load Labels to map fault resistance
    labels = pd.read_csv(os.path.join(RAW_DATA_DIR, "hv_double_line_90kv_labels.csv"))

    # Validate that fault_resistance column exists
    if "fault_resistance" not in labels.columns:
        print("Error: 'fault_resistance' column not found in labels CSV.")
        print(f"Available columns: {list(labels.columns)}")
        return
    
    # Extract resistance for each sample in test set
    resistances = []
    for eid in episode_ids:
        # Match sample_id in labels
        match = labels[labels["sample_id"] == eid]
        if not match.empty:
            resistances.append(match.iloc[0]["fault_resistance"])
        else:
            resistances.append(0.0)
            
    resistances = np.array(resistances)

    # Run Inference
    print("Running inference...")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    batch_size = 128
    all_det_preds = []
    all_cls_preds = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            det_out, cls_out = model(batch_X)
            
            all_det_preds.append(torch.argmax(det_out, dim=1).cpu().numpy())
            all_cls_preds.append(torch.argmax(cls_out, dim=1).cpu().numpy())
            
    pred_det = np.concatenate(all_det_preds)
    pred_cls = np.concatenate(all_cls_preds)

    # Filter only fault events (y_det == 1) for classification and resistance analysis
    fault_mask = (y_det == 1)
    
    resistances_faults = resistances[fault_mask]
    y_cls_faults = y_cls[fault_mask]
    pred_cls_faults = pred_cls[fault_mask]

    # Bin the resistances
    bins = [0, 2, 5, 8, 10.1]
    labels_bins = ['< 2 Ohms', '2 - 5 Ohms', '5 - 8 Ohms', '> 8 Ohms']
    
    binned = pd.cut(resistances_faults, bins=bins, labels=labels_bins, right=False)
    
    accuracies = []
    for bin_label in labels_bins:
        idx = (binned == bin_label)
        if np.sum(idx) > 0:
            acc = accuracy_score(y_cls_faults[idx], pred_cls_faults[idx])
        else:
            acc = 0.0
        accuracies.append(acc * 100)
        print(f"Resistance {bin_label}: Accuracy {acc*100:.2f}% (N={np.sum(idx)})")

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels_bins, accuracies, color='#3b82f6', edgecolor='black')
    
    # Dynamic y-limit based on actual values
    min_acc = min(accuracies)
    plt.ylim(max(0, min_acc - 5), 101)
    
    plt.ylabel('Classification Accuracy (%)')
    plt.xlabel('Fault Resistance Range')
    plt.title('TCN Performance Across Fault Resistances (HIF Evaluation)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "hif_performance.png"), dpi=300)
    print("\nSaved HIF performance plot to results/hif_performance.png")

if __name__ == "__main__":
    main()
