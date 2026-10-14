import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

from src.models.tcn import TCN
from src.models.config import MODEL_DIR, DATA_DIR, RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("=" * 50)
    print("XAI ANALYSIS (INTEGRATED GRADIENTS)")
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

    # Create a wrapper model that outputs only classification logic
    class ClassificationModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            _, cls = self.model(x)
            return cls
            
    wrapped_model = ClassificationModelWrapper(model)

    # Load Test Data
    print("Loading test data...")
    test_data = np.load(os.path.join(DATA_DIR, "test.npz"))
    X = test_data["X"]
    y_cls = test_data["fault_types"]

    # Find a sample that is a fault
    # We will pick the first sample with fault type 0 (SLG)
    target_idx = np.where(y_cls == 0)[0][0]
    sample = torch.tensor(X[target_idx], dtype=torch.float32).unsqueeze(0).to(device)
    
    # Run prediction to make sure it is correct
    pred = wrapped_model(sample)
    pred_class = torch.argmax(pred, dim=1).item()
    print(f"Sample {target_idx}: True class: {y_cls[target_idx]}, Predicted class: {pred_class}")

    # Integrated Gradients
    print("Computing Integrated Gradients...")
    ig = IntegratedGradients(wrapped_model)
    
    # Compute attribution for the predicted class
    attributions, delta = ig.attribute(sample, target=pred_class, return_convergence_delta=True)
    
    attributions = attributions.squeeze(0).cpu().detach().numpy()
    original_signal = sample.squeeze(0).cpu().numpy()

    # We will plot the first 3 channels (Va, Vb, Vc) and their attributions
    # The actual order in the first 6 channels of raw data is Ia, Ib, Ic, Va, Vb, Vc
    # So Va, Vb, Vc are indices 3, 4, 5.
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    channel_names = ["Va", "Vb", "Vc"]
    
    for i in range(3):
        ax = axes[i]
        
        # Plot original signal
        ax.plot(original_signal[i + 3], label=f'Original {channel_names[i]}', color='#3b82f6', alpha=0.9, linewidth=2)
        
        # Plot attribution as a bar/area chart at the bottom
        ax2 = ax.twinx()
        
        # Take absolute value of attribution for importance magnitude
        attr_mag = np.abs(attributions[i + 3])
        ax2.bar(np.arange(256), attr_mag, color='#ef4444', alpha=0.4, label='Importance (IG)')
        
        ax.set_ylabel('Amplitude')
        ax2.set_ylabel('Importance Score')
        ax.set_title(f'{channel_names[i]} - Waveform vs Model Attention')
        
        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "xai_attributions.png"), dpi=300)
    print("Saved XAI visualization to results/xai_attributions.png")

if __name__ == "__main__":
    main()
