# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack ML application for predicting Jamuna River bankline shifting using a Temporal Fusion Transformer (TFT). Python FastAPI backend serves TFT predictions; Next.js 16 frontend displays forecasts, anomaly detection, and model evaluation.

## Commands

### Run Everything
```bash
python start.py          # Installs deps + starts backend (:8000) and frontend (:3000)
```

### Backend (Python, uv)
```bash
uv sync                  # Install dependencies
uv sync --dev            # Install with dev dependencies (ruff, mypy, pytest)
uv run uvicorn src.serving.api:app --host 0.0.0.0 --port 8000   # Start API server
uv run python -m src.training.train      # Train TFT model
uv run python -m src.training.tune       # Optuna hyperparameter search
uv run python -m src.training.evaluate   # Evaluate model metrics
```

### Frontend (Next.js, pnpm)
```bash
cd frontend && pnpm install    # Install dependencies
cd frontend && pnpm dev        # Dev server on :3000
cd frontend && pnpm build      # Production build
```

### Linting & Testing
```bash
uv run ruff check src/         # Lint Python
uv run ruff format src/        # Format Python
uv run mypy src/               # Type check
uv run pytest tests/           # Run all tests
uv run pytest tests/test_foo.py::test_bar  # Single test
```

### Docker
```bash
docker build -t jamuna .
docker run -p 8000:8000 jamuna
```

## Architecture

### Backend (`src/`)

The backend is a **FastAPI** app (`src/serving/api.py`) that loads a trained TFT checkpoint on startup, pre-computes a prediction cache for years 2021-2040, and serves predictions via REST endpoints.

**Module structure:**
- `src/config.py` — Pydantic settings loaded from `configs/config.yaml` (paths, data splits, TFT hyperparams, training, MLflow, Optuna, anomaly, API config)
- `src/serving/api.py` — FastAPI app with `AppState` holding all loaded models/data in memory. Endpoints use sync `def` (not async) because PyTorch inference is CPU-bound
- `src/serving/schemas.py` — Pydantic v2 request/response models
- `src/models/tft_wrapper.py` — TFT construction from `TimeSeriesDataSet`, QuantileLoss([0.1, 0.5, 0.9])
- `src/models/baselines.py` — Persistence, ARIMA, random forest, linear baselines
- `src/training/train.py` — Lightning training with MLflow logging and checkpoint saving
- `src/training/tune.py` — Optuna optimization (50 trials, SQLite storage)
- `src/training/evaluate.py` — Metrics: NSE, RMSE, MAE, KGE, quantile coverage
- `src/data/loader.py` — Loads Excel with 2-row header into tidy long-format DataFrame (50 reaches × 2 banks × 27 years)
- `src/data/preprocessing.py` — Imputation, feature engineering (erosion_indicator, rate_of_change, rolling_mean_3, net_channel_erosion), categorical encoding, optional RobustScaler
- `src/data/dataset.py` — PyTorch Forecasting `TimeSeriesDataSet` builder (encoder_length=10, prediction_length=5)
- `src/anomaly/changepoint.py` — PELT change-point detection with protection signature identification
- `src/anomaly/autoencoder.py` — VAE for reconstruction-based anomaly scoring

**Prediction modes:**
- Direct forecast (2021-2025): Single TFT pass from historical data
- Rolling forecast (2026+): Iterative multi-step — appends predicted q50 values as synthetic rows, recomputes features, feeds back into model

**Background training:** POST `/train` runs training in a `ThreadPoolExecutor`. Poll `/train/{job_id}/status` and `/train/{job_id}/logs` for progress. Phases: training → model_reloading → cache_building → ready.

### Frontend (`frontend/src/`)

Next.js 16 App Router with React 19, TanStack React Query (staleTime: 30s), Zustand state, TailwindCSS 4.

**Pages:**
- `/` — Dashboard: YearSelector + 5×10 ReachGrid showing predictions for all 50 reaches
- `/series/[reach_id]/[bank_side]` — Historical time series + forecast with quantile bands (Recharts)
- `/anomaly` — PELT change-point detection results table
- `/evaluate` — Model metrics panel (NSE, RMSE, MAE, KGE) per train/val/test split

**Key files:**
- `src/lib/api.js` — Fetch wrapper functions for all backend endpoints; base URL from `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`)
- `src/store/index.js` — Zustand store: selectedYear, selectedReach, trainingJobId, toastMessage
- `src/components/` — Organized by feature: `dashboard/`, `series/`, `training/`, `anomaly/`, `evaluate/`, `ui/`, `nav/`
- `src/app/globals.css` — CSS variables for dark theme; color semantics: `--erosion` (red) for bank erosion, `--deposition` (green) for deposition

### Data

- Source: `data/raw/Distances (1).xlsx` — 50 river reaches, 27 observed years (1991-2020, non-contiguous), left/right bank distances
- Processed output: `data/processed/`
- Temporal split: train (≤2010), validation (2011-2015), test (≥2016)
- All config in `configs/config.yaml`

## Code Conventions

- Python: ruff with `target-version = "py312"`, line-length 100, select rules E/F/I/UP/B
- Python: mypy strict mode, ignore_missing_imports
- Frontend: path alias `@/*` → `./src/*` (jsconfig.json)
- Frontend: no API routes — components call backend directly via fetch
