# src/serving/ — FastAPI Application

## Files
- `api.py` — Main FastAPI app with all endpoints and `AppState` singleton
- `schemas.py` — Pydantic v2 request/response models
- `__init__.py` — Empty

## Key Endpoints
- `GET /health` — Model status and metadata
- `POST /predict` — TFT forecast with quantile intervals (q10, q50, q90)
- `POST /predict/baseline` — Named baseline model forecast
- `GET /predict/year/{year}` — All predictions for a specific year
- `GET /predict/baseline/year/{year}` — All baseline predictions for a year
- `GET /evaluate` — TFT evaluation metrics (val or test split)
- `GET /evaluate/compare` — Side-by-side metrics for all 11 algorithms
- `GET /anomaly/changepoints` — PELT changepoint detection results
- `GET /series/{reach_id}/{bank_side}` — Historical series + latest forecast
- `POST /train` — Background retraining (target: tft/baselines/all/specific model)

## AppState
Holds in memory: TFT model, baseline models dict, preprocessed DataFrame, prediction cache, settings. All endpoints use sync `def` (CPU-bound PyTorch inference).

## Baseline Models Loading
On startup, loads all `.joblib` files from `models/baselines/{model_name}/` into `state.baseline_models[model_name]["{reach_id}_{bank_side}"]`.
