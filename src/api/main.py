from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import json
import numpy as np
import torch
from pydantic import BaseModel
import sys

# Add src to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.tcn import TCN
from src.models.cnn_baseline import CNN1D
from src.models.lstm_baseline import LSTMClassifier
from src.models.config import FAULT_TYPE_NAMES

# Global variables for model and data
model = None
cnn_model = None
lstm_model = None
test_data = None
device = torch.device("cpu")

SUPPORTED_MODELS = {"TCN", "CNN", "LSTM"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and test data on startup."""
    global model, cnn_model, lstm_model, test_data, device

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"Using device: {device}")

    # Load TCN Model
    model = TCN(input_channels=48, num_classes=4).to(device)
    checkpoint = torch.load("models/tcn_best.pt", map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Load CNN Model
    cnn_model = CNN1D(input_channels=48, num_classes=4).to(device)
    cnn_model.load_state_dict(torch.load("models/cnn_best.pt", map_location=device, weights_only=True))
    cnn_model.eval()

    # Load LSTM Model
    lstm_model = LSTMClassifier(input_features=48, hidden_size=128, num_layers=2, num_classes=4).to(device)
    lstm_model.load_state_dict(torch.load("models/lstm_best.pt", map_location=device, weights_only=True))
    lstm_model.eval()

    # Load Test Data for inference sampling
    try:
        test_data = np.load("data/processed/test.npz")
        print(f"Loaded test data with {test_data['X'].shape[0]} samples")
    except Exception as e:
        print(f"Failed to load test data: {e}")

    yield  # Application runs here

    # Cleanup (if needed) goes here
    print("Shutting down FDCS API.")


app = FastAPI(title="TCN Microgrid FDCS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the results directory statically so the frontend can load the plots
app.mount("/api/static/results", StaticFiles(directory="results"), name="results")

# Serve the dashboard statically
app.mount("/dashboard", StaticFiles(directory="dashboard"), name="dashboard")


@app.get("/")
def read_root():
    return RedirectResponse(url="/dashboard/index.html")

@app.get("/api/results/performance")
def get_performance():
    try:
        with open("results/tcn_evaluation_results.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Performance results not found")

@app.get("/api/results/comparison")
def get_comparison():
    try:
        with open("results/model_comparison.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Comparison results not found")

@app.get("/api/results/noise")
def get_noise():
    try:
        with open("results/noise_results.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Noise results not found")

@app.get("/api/inference/random")
def run_random_inference(model_type: str = "TCN"):
    # Validate model type
    model_type_upper = model_type.upper()
    if model_type_upper not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model type '{model_type}'. "
                   f"Supported models: {', '.join(sorted(SUPPORTED_MODELS))}"
        )

    if test_data is None:
        raise HTTPException(status_code=500, detail="Test data not loaded")

    idx = np.random.randint(0, test_data['X'].shape[0])

    x = test_data['X'][idx]
    true_det = int(test_data['detection_labels'][idx])
    true_cls = int(test_data['fault_types'][idx])

    # Run inference
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    # Select the model
    if model_type_upper == "CNN":
        active_model = cnn_model
    elif model_type_upper == "LSTM":
        active_model = lstm_model
    else:
        active_model = model

    with torch.no_grad():
        det_out, cls_out = active_model(x_tensor)

        det_prob = torch.softmax(det_out, dim=1)[0]
        cls_prob = torch.softmax(cls_out, dim=1)[0]

        det_conf, det_pred = torch.max(det_prob, dim=0)
        cls_conf, cls_pred = torch.max(cls_prob, dim=0)

        pred_det = det_pred.item()
        pred_cls = cls_pred.item()
        det_conf = det_conf.item()
        cls_conf = cls_conf.item()

        needs_review = bool(det_conf < 0.85 or (pred_det == 1 and cls_conf < 0.85))

    # Downsample channels for visualization (take first 6 as Ia,Ib,Ic,Va,Vb,Vc proxy)
    # The real order is Ia,Ib,Ic,Va,Vb,Vc for the first line.
    waveform_data = x[:6, :].tolist()

    return {
        "sample_id": idx,
        "true_detection": "Fault" if true_det == 1 else "Normal",
        "true_classification": FAULT_TYPE_NAMES.get(true_cls, "Unknown"),
        "predicted_detection": "Fault" if pred_det == 1 else "Normal",
        "predicted_classification": FAULT_TYPE_NAMES.get(pred_cls, "Unknown"),
        "detection_confidence": float(det_conf),
        "classification_confidence": float(cls_conf),
        "needs_review": needs_review,
        "waveforms": {
            "Ia": waveform_data[0],
            "Ib": waveform_data[1],
            "Ic": waveform_data[2],
            "Va": waveform_data[3],
            "Vb": waveform_data[4],
            "Vc": waveform_data[5],
        },
        "time_axis": np.arange(256).tolist()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
