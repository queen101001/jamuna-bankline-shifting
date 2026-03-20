# src/ — Python Backend

FastAPI backend for Jamuna River bankline shift prediction.

## Module Structure
- `config.py` — Pydantic `Settings` from `configs/config.yaml` (paths, splits, TFT hyperparams, API config)
- `__init__.py` — Applies `torch.load` safety patch (must import before other torch usage)
- `serving/` — FastAPI app, Pydantic schemas, REST endpoints
- `models/` — TFT wrapper + 10 baseline model classes
- `training/` — Training scripts (TFT, baselines), Optuna tuning, evaluation
- `data/` — Excel data loading, preprocessing pipeline, PyTorch dataset builder
- `anomaly/` — PELT changepoint detection, VAE autoencoder

## Key Patterns
- All models are loaded into `AppState` at startup (TFT checkpoint + joblib baselines)
- Predictions cached in memory for years 2021–2040 on startup
- Training runs in background `ThreadPoolExecutor`
- Data: 50 reaches × 2 banks × 27 years (1991–2020)
- Temporal split: train ≤2010, val 2011–2015, test ≥2016
