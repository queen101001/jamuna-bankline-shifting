# Jamuna River Bankline Prediction System

A full-stack machine learning application for predicting the bankline shifting of the Jamuna River (Bangladesh) using a **Temporal Fusion Transformer (TFT)** and **10 baseline algorithms**. Built with a Python FastAPI backend and a Next.js 16 frontend.

The system comes with **pre-trained models** — no training is required to run. Just clone, install dependencies, and start.

---

## Quick Start (Clone & Run)

```bash
# 1. Clone the repository
git clone https://github.com/queen101001/jamuna-bankline-shifting.git
cd jamuna-bankline-shifting

# 2. Run the start script (installs dependencies automatically)
python start.py
```

That's it. The script will:
- Install Python dependencies (via `uv`)
- Install frontend dependencies (via `pnpm`)
- Start the backend API server on **http://localhost:8000**
- Start the frontend dev server on **http://localhost:3000**

Open **http://localhost:3000** in your browser.

---

## Prerequisites

Before running `start.py`, you need these tools installed:

### Python 3.12 + uv (Python package manager)

**Windows:**
```powershell
# Install uv (it will auto-install Python 3.12)
winget install --id=astral-sh.uv
# OR
pip install uv
```

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Node.js + pnpm (Frontend package manager)

**Windows:**
```powershell
# Install Node.js (includes npm)
winget install OpenJS.NodeJS.LTS

# Install pnpm globally
npm install -g pnpm
```

**Linux / macOS:**
```bash
# Install Node.js via your package manager, then:
npm install -g pnpm
```

### Verify Installation

```bash
uv --version       # Should show uv X.Y.Z
node --version      # Should show v20+ or v22+
pnpm --version      # Should show 8+ or 9+
```

---

## Docker Setup (Alternative)

If you prefer Docker, you can run the entire system without installing Python or Node.js locally:

```bash
# Build and start both services
docker compose up --build

# Or run in background
docker compose up --build -d
```

This starts:
- **Backend** (FastAPI) on http://localhost:8000
- **Frontend** (Next.js) on http://localhost:3000

To stop:
```bash
docker compose down
```

### Docker Requirements
- Docker Desktop (Windows/macOS) or Docker Engine (Linux)
- Docker Compose v2 (included with Docker Desktop)

---

## Project Structure

```
jamuna-bankline-shifting/
├── start.py                  # Cross-platform startup script
├── Dockerfile                # Backend Docker image
├── docker-compose.yml        # Full-stack Docker orchestration
├── pyproject.toml             # Python dependencies & project config
├── configs/
│   └── config.yaml           # All configuration (paths, hyperparams, splits)
├── data/
│   └── raw/                  # Raw Excel measurement data
├── dataset/
│   ├── proper-dataset.xlsx   # Primary dataset (referenced by config)
│   └── dataset.xlsx          # Alternative dataset
├── models/
│   ├── tft/
│   │   └── last.ckpt         # Pre-trained TFT model (production)
│   ├── baselines/            # 1000 pre-trained baseline models
│   │   ├── arima/            # 100 .joblib files (50 reaches × 2 banks)
│   │   ├── linear/
│   │   ├── persistence/
│   │   ├── random_forest/
│   │   ├── exp_smoothing/
│   │   ├── xgboost/
│   │   ├── svr/
│   │   ├── gradient_boosting/
│   │   ├── elastic_net/
│   │   └── knn/
│   └── predictions_cache.json # Pre-computed predictions (2021–2100)
├── src/                       # Python backend source
│   ├── serving/
│   │   ├── api.py            # FastAPI app (all endpoints)
│   │   └── schemas.py        # Pydantic request/response models
│   ├── models/
│   │   ├── tft_wrapper.py    # TFT model construction & loading
│   │   └── baselines.py      # 10 baseline algorithm implementations
│   ├── training/
│   │   ├── train.py          # TFT training with Lightning
│   │   ├── train_all_baselines.py  # Train all 10 baselines
│   │   └── train_*.py        # Individual baseline training scripts
│   ├── data/
│   │   ├── loader.py         # Excel data loading
│   │   ├── preprocessing.py  # Feature engineering
│   │   └── dataset.py        # PyTorch Forecasting dataset builder
│   ├── anomaly/
│   │   ├── changepoint.py    # PELT changepoint detection
│   │   └── autoencoder.py    # VAE anomaly scoring
│   └── config.py             # Pydantic settings from config.yaml
└── frontend/                  # Next.js 16 frontend
    ├── Dockerfile             # Frontend Docker image
    ├── package.json
    └── src/
        ├── app/               # Next.js App Router pages
        │   ├── page.js        # Dashboard
        │   ├── evaluate/      # Model metrics
        │   ├── compare/       # Multi-model comparison
        │   ├── anomaly/       # Changepoint detection
        │   ├── validate/      # Upload & validate predictions
        │   └── series/        # Time series detail
        ├── components/        # React components
        ├── lib/               # API client, algorithms config, page info
        └── store/             # Zustand state management
```

---

## Pages & Features

| Page | URL | Description |
|------|-----|-------------|
| **Dashboard** | `/` | Year selector + algorithm tabs + 50-reach prediction grid |
| **Series Detail** | `/series/[reach]/[bank]` | Historical chart + forecast with quantile bands |
| **Metrics** | `/evaluate` | NSE, KGE, RMSE, MAE comparison across all 11 algorithms |
| **Anomalies** | `/anomaly` | PELT changepoint detection table with protection signatures |
| **Compare** | `/compare` | Multi-model forecast overlay chart + metrics table |
| **Validate** | `/validate` | Upload 2021–2025 Excel data to test predictions against reality |

Every page has a floating **info button** (bottom-left) that opens a comprehensive learning guide explaining every element on the page.

---

## Algorithms

The system uses 11 prediction algorithms:

| # | Algorithm | Type | Output |
|---|-----------|------|--------|
| 1 | **TFT** (Temporal Fusion Transformer) | Deep Learning | Probabilistic (q10, q50, q90) |
| 2 | Persistence (Naive) | Baseline | Point prediction |
| 3 | Linear Extrapolation | Baseline | Point prediction |
| 4 | ARIMA | Statistical | Point prediction |
| 5 | Random Forest | ML Ensemble | Point prediction |
| 6 | Holt's Exponential Smoothing | Statistical | Point prediction |
| 7 | XGBoost | ML Ensemble | Point prediction |
| 8 | SVR (Support Vector Regression) | ML | Point prediction |
| 9 | Gradient Boosting | ML Ensemble | Point prediction |
| 10 | Elastic Net | Regularized Linear | Point prediction |
| 11 | KNN Regression | Instance-based | Point prediction |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/predict/year/{year}` | TFT predictions for all 100 series |
| `GET` | `/predict/baseline/year/{year}?model_name=...` | Baseline predictions |
| `POST` | `/predict/baseline` | Single series baseline prediction |
| `GET` | `/evaluate/compare` | Side-by-side metrics for all algorithms |
| `GET` | `/anomaly/changepoints` | PELT changepoint detection results |
| `POST` | `/train` | Start background training job |
| `GET` | `/train/{job_id}/status` | Poll training progress |

Full API documentation: **http://localhost:8000/docs** (Swagger UI)

---

## Training Models (Optional)

Pre-trained models are included — you do **not** need to train to use the system. If you want to retrain:

### From the UI
Click the **Train** button (top-right corner) in the web interface. You can train:
- TFT model only
- All baseline models
- Individual baseline algorithms

### From the Command Line
```bash
# Train TFT
uv run python -m src.training.train

# Train all 10 baselines (1000 models)
uv run python -m src.training.train_all_baselines

# Train a specific baseline
uv run python -m src.training.train_arima
uv run python -m src.training.train_xgboost
# etc.

# Hyperparameter tuning (Optuna, 50 trials)
uv run python -m src.training.tune
```

---

## Configuration

All settings are in `configs/config.yaml`:

- **Paths**: Dataset location, model output directories
- **Data splits**: Train (≤2010), Validation (2011–2015), Test (≥2016)
- **TFT hyperparameters**: encoder_length, prediction_length, hidden_size, etc.
- **Training**: Early stopping patience, learning rate, max epochs
- **Anomaly detection**: PELT penalty, VAE architecture
- **API**: Host, port settings

---

## Tech Stack

### Backend
- **Python 3.12** — Runtime
- **FastAPI** — REST API framework
- **PyTorch + PyTorch Lightning** — Deep learning training
- **PyTorch Forecasting** — TFT model architecture
- **scikit-learn, XGBoost, statsmodels** — Baseline algorithms
- **ruptures** — PELT changepoint detection
- **MLflow** — Experiment tracking
- **Optuna** — Hyperparameter optimization
- **uv** — Fast Python package manager

### Frontend
- **Next.js 16** (App Router, Turbopack) — React framework
- **React 19** — UI library
- **TanStack React Query** — Server state management
- **Zustand** — Client state management
- **Recharts** — Data visualization
- **TailwindCSS 4** — Styling
- **Lucide React** — Icons
- **pnpm** — Node.js package manager

---

## Data

- **Source**: Satellite-derived bankline distance measurements
- **Coverage**: 50 river reaches × 2 banks (left + right) = 100 time series
- **Historical period**: 1991–2020 (27 non-contiguous observation years)
- **Prediction range**: 2021–2100
  - Direct forecast: 2021–2025 (single model pass)
  - Rolling forecast: 2026–2100 (iterative, predictions feed back as input)

---

## Troubleshooting

### `uv` not found
Install uv: `pip install uv` or see [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/)

### `pnpm` not found
Install pnpm: `npm install -g pnpm`

### Port already in use
The start script automatically frees ports 8000 and 3000 on Linux/macOS. On Windows, manually close any process using those ports.

### Backend starts but frontend can't connect
Ensure the backend is running on port 8000. Check the browser console for CORS errors. The frontend expects the API at `http://localhost:8000` by default.

### Models not loading
The `models/` directory must contain:
- `models/tft/last.ckpt` — TFT checkpoint
- `models/baselines/` — 10 subdirectories with .joblib files
- `models/predictions_cache.json` — Pre-computed prediction cache

These are included in the git repository. If missing, re-clone or retrain.
