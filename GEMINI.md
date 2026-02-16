# Jamuna Bankline Shifting Prediction System

## Project Overview
This project is a full-stack Machine Learning application designed to predict the bankline shifting of the Jamuna River. It utilizes a **Temporal Fusion Transformer (TFT)** model to forecast erosion and deposition patterns. The system consists of a **FastAPI** backend for serving predictions and model training, and a **Next.js 16** frontend for visualization and interaction.

## Tech Stack

### Backend
*   **Language:** Python 3.12+
*   **Framework:** FastAPI
*   **ML Core:** PyTorch, PyTorch Lightning, PyTorch Forecasting (TFT)
*   **Experiment Tracking:** MLflow
*   **Hyperparameter Tuning:** Optuna
*   **Data Processing:** Pandas, NumPy, Scikit-learn
*   **Package Manager:** `uv`

### Frontend
*   **Framework:** Next.js 16 (App Router)
*   **Library:** React 19
*   **Styling:** Tailwind CSS 4
*   **State Management:** Zustand
*   **Data Fetching:** TanStack React Query
*   **Charts:** Recharts

## Getting Started

### Prerequisites
*   Python 3.12 or higher
*   Node.js & pnpm
*   `uv` (Universal Python Package Manager)

### Quick Start
To install dependencies and start both the backend (port 8000) and frontend (port 3000):

```bash
python start.py
```

### Manual Setup

#### Backend
```bash
# Install dependencies
uv sync
# Run API server
uv run uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

#### Frontend
```bash
cd frontend
# Install dependencies
pnpm install
# Run development server
pnpm dev
```

## Project Structure

### Backend (`src/`)
*   **`src/serving/`**: FastAPI application and Pydantic schemas.
    *   `api.py`: Main entry point, loads models, serves endpoints.
    *   `schemas.py`: Request/Response models.
*   **`src/models/`**: Model definitions.
    *   `tft_wrapper.py`: Temporal Fusion Transformer implementation.
    *   `baselines.py`: Baseline models (ARIMA, Persistence, etc.).
*   **`src/training/`**: Training and evaluation scripts.
    *   `train.py`: PyTorch Lightning training loop with MLflow logging.
    *   `tune.py`: Hyperparameter optimization using Optuna.
    *   `evaluate.py`: Model evaluation metrics (NSE, RMSE, MAE, KGE).
*   **`src/data/`**: Data loading and preprocessing.
    *   `loader.py`: Loads raw Excel data.
    *   `preprocessing.py`: Feature engineering and imputation.
    *   `dataset.py`: PyTorch Forecasting dataset builder.
*   **`src/anomaly/`**: Anomaly detection modules.
    *   `changepoint.py`: PELT change-point detection.
    *   `autoencoder.py`: VAE for anomaly scoring.
*   **`src/config.py`**: Centralized configuration loaded from `configs/config.yaml`.

### Frontend (`frontend/src/`)
*   **`app/`**: Next.js App Router pages.
    *   `/`: Dashboard with ReachGrid.
    *   `/series/[reach_id]/[bank_side]`: Detailed charts.
    *   `/anomaly`: Change-point detection results.
    *   `/evaluate`: Model metrics.
*   **`components/`**: React components organized by feature (`dashboard`, `series`, `ui`, etc.).
*   **`lib/api.js`**: API client for backend communication.
*   **`store/`**: Zustand state management.

## Key Commands

### Backend Operations
*   **Train Model:** `uv run python -m src.training.train`
*   **Tune Hyperparameters:** `uv run python -m src.training.tune`
*   **Evaluate Model:** `uv run python -m src.training.evaluate`

### Code Quality
*   **Lint:** `uv run ruff check src/`
*   **Format:** `uv run ruff format src/`
*   **Type Check:** `uv run mypy src/`
*   **Test:** `uv run pytest tests/`

### Docker
```bash
docker build -t jamuna .
docker run -p 8000:8000 jamuna
```

## Development Conventions
*   **Python:** Strictly follow PEP 8. Use `ruff` for linting/formatting and `mypy` for static type checking. Target Python version is 3.12.
*   **Frontend:** Use functional components and hooks. Place all components in `src/components`. Use standard `fetch` in `src/lib/api.js` for API calls.
*   **Configuration:** All configurable parameters (paths, model params) should be in `configs/config.yaml`, not hardcoded.
