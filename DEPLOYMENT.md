# Deployment Guide for CloudIntellMal

This guide outlines how to deploy the **CloudIntellMal** project (Deep Learning Fault Detection and Classification for Microgrids) to Vercel and how to optionally connect an external Python FastAPI backend.

---

## 1. Quick Deployment to Vercel (Recommended)

The repository has been configured to deploy out-of-the-box to **Vercel** with full interactive functionality:

1. Push your changes to GitHub:
   ```bash
   git add .
   git commit -m "Configure Vercel deployment and dynamic API routing"
   git push origin main
   ```
2. Log in to [Vercel](https://vercel.com) and click **"Add New..." → "Project"**.
3. Select your `CloudIntellMal` repository.
4. Leave all settings at default (**Root Directory: `./`**, Framework Preset: Other).
5. Click **"Deploy"**.

### What happens behind the scenes:
- `.vercelignore` ignores `requirements.txt` and heavy local model weights (`models/`, `data/`). This prevents Vercel from trying to build heavy PyTorch C-extensions, completely resolving the `PYTHON_ENTRYPOINT_NOT_FOUND` build failure.
- `vercel.json` routes the root URL `/` to `/dashboard/index.html`.
- Vercel Serverless Functions in `api/inference/random.js` and `api/results/[type].js` serve:
  - Evaluation results (`/api/results/performance`, `/api/results/comparison`, `/api/results/noise`)
  - Real waveform multi-model inference (`/api/inference/random?model_type=TCN|CNN|LSTM`) sampled from pre-evaluated test-set episodes with real signals.
  - Model explanation and HIF performance images (`/api/static/results/...`).

---

## 2. Running Locally

To run the complete pipeline with live PyTorch models on your local machine:

1. Activate your virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Start the FastAPI server:
   ```bash
   python -m src.api.main
   ```
3. Open `http://127.0.0.1:8000/` in your browser. The dashboard will automatically connect to your local PyTorch models (`models/tcn_best.pt`, `models/cnn_best.pt`, `models/lstm_best.pt`) and `data/processed/test.npz`.

---

## 3. Optional: Hosting a Dedicated PyTorch Backend

If you wish to host the heavy PyTorch inference engine 24/7 on a dedicated Python service (e.g., Render, Railway, Hugging Face Spaces, or AWS EC2):

1. **Deploy the backend** to a platform with full Python/PyTorch support:
   - **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   - **Dependencies**: `pip install -r requirements.txt`
   - Ensure `models/tcn_best.pt` and `data/processed/test.npz` are included or downloaded on startup.

2. **Connect the Vercel Frontend**:
   - In `dashboard/config.js`, set your deployed backend URL:
     ```javascript
     window.API_BASE_URL = "https://your-backend.onrender.com/api";
     ```
   - Commit and push to GitHub. The Vercel frontend will now route all live requests to your dedicated backend.

