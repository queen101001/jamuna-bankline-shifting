# src/training/ — Training & Evaluation Scripts

## Files
- `train.py` — TFT training with PyTorch Lightning, MLflow logging, checkpoint saving
- `train_all_baselines.py` — Bulk script: trains all 10 baselines for all 50×2 series
- `train_{model}.py` — Individual baseline trainers (arima, rf, xgboost, etc.)
- `_baseline_trainer.py` — Shared helper: `train_single_baseline(model_name, ...)` used by all individual scripts
- `evaluate.py` — Metrics computation: NSE, RMSE, MAE, KGE, quantile coverage
- `tune.py` — Optuna hyperparameter optimization (50 trials, SQLite storage)
- `__init__.py` — Empty

## Usage
```bash
uv run python -m src.training.train           # Train TFT
uv run python -m src.training.train_all_baselines  # Train all baselines
uv run python -m src.training.train_arima      # Train specific baseline
uv run python -m src.training.evaluate         # Generate eval metrics
uv run python -m src.training.tune             # Optuna search
```

## Output
- TFT checkpoints: `models/checkpoints/`
- Baseline models: `models/baselines/{model_name}/{reach_id}_{bank_side}.joblib`
- Eval results: `data/processed/eval_results/`
