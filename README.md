# Deep Learning-Based Fault Detection and Classification in Microgrids using TCN

An end-to-end deep-learning-based Fault Detection and Classification Scheme (FDCS) for modern microgrids that learns temporal fault signatures directly from three-phase voltage and current waveforms using a Temporal Convolutional Network (TCN).

## Project Overview

This project develops an intelligent fault detection and classification system for microgrids using the PROTECT-90 dataset. The system processes synchronized three-phase voltage and current waveforms and performs two tasks:

1. **Fault Detection** — Binary classification (Fault / No Fault)
2. **Fault Classification** — Four-class classification (LLL / LL / SLG / LLG)

The TCN learns temporal representations directly from waveform data without requiring manually engineered features.

## Project Structure

```text
data/
├── raw/                  # Original PROTECT-90 dataset files
└── processed/            # Normalized, windowed dataset arrays
models/                   # Trained PyTorch model checkpoints
results/                  # Evaluation metrics, confusion matrices, plots
src/
├── data_analysis/        # Dataset exploration and validation
├── evaluation/           # TCN evaluation, noise robustness, response time, model comparison
├── models/               # TCN, CNN, LSTM architectures, training, baselines
└── preprocessing/        # Episode splitting, window generation, normalization
tests/                    # Unit tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

### 1. Dataset Analysis
```bash
python -m src.data_analysis.explore_dataset
```

### 2. Preprocessing (Episode Split → Windowing → Normalization)
```bash
python -m src.preprocessing.create_split
python -m src.preprocessing.preprocess_dataset
```

### 3. TCN Training
```bash
python -m src.models.train
```

### 4. TCN Evaluation
```bash
python -m src.evaluation.evaluate
```

### 5. Baseline Models
```bash
python -m src.models.baselines           # Decision Tree
python -m src.models.svm_baseline        # SVM
python -m src.models.cnn_baseline        # CNN
python -m src.models.lstm_baseline       # LSTM
```

### 6. Robustness and Performance
```bash
python -m src.evaluation.noise_robustness
python -m src.evaluation.response_time
python -m src.evaluation.compare_models
```

### 7. Tests
```bash
python -m pytest tests/ -v
```

### 8. Interactive SCADA Dashboard (Local)
Start the FastAPI backend with PyTorch model inference:
```bash
python -m src.api.main
```
Then open `http://127.0.0.1:8000/` in your browser.

### 9. Vercel Deployment (Cloud)
This repository is configured for one-click Vercel deployment:
- **Zero Configuration**: Simply import the repository in [Vercel](https://vercel.com).
- `.vercelignore` prevents heavy PyTorch build collisions and solves `PYTHON_ENTRYPOINT_NOT_FOUND`.
- `vercel.json` and serverless API endpoints in `api/` automatically serve the dashboard, evaluation metrics, and real test-set waveform inference interactively.
- Optional: Set `window.API_BASE_URL` in `dashboard/config.js` if pointing to an external dedicated backend.

## Dataset

- **Source**: PROTECT-90 (HV Double Line 90kV)
- **Episodes**: 9,022
- **Sampling Rate**: 6.4 kHz
- **Channels**: 48 (voltage + current measurements)
- **Window Size**: 256 samples (40 ms)
- **Fault Types**: LLL, LL, SLG, LLG

## Model Architecture

The TCN uses causal dilated convolutions with residual connections:

```
Input (48 channels, 256 timesteps)
  → Residual Block (dilation=1, 64 channels)
  → Residual Block (dilation=2, 64 channels)
  → Residual Block (dilation=4, 128 channels)
  → Residual Block (dilation=8, 128 channels)
  → Global Average Pooling
  → Detection Head (2 classes)
  → Classification Head (4 classes)
```

## Technologies

| Component | Technology |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Machine Learning | Scikit-learn |
| Data Processing | NumPy, Pandas, SciPy |
| Visualization | Matplotlib |

## Future Work (Hardware-in-the-Loop Simulation Concept)

With a demonstrated inference time of just **0.09 ms**, the TCN model is highly suitable for real-time digital relay applications. Future work will focus on **Hardware-in-the-Loop (HIL) Simulation**:
- **Deployment Target:** FPGA, Raspberry Pi, or dSPACE real-time simulator.
- **Goal:** Act as a physical digital relay, receiving analog waveform signals, digitizing them, running the TCN inference, and physically triggering a trip signal to a breaker within the required 1-cycle fault clearance time.
- **Expected Outcome:** Prove that the model's speed and accuracy translates from offline testing to a physical, real-time power systems environment.
