# src/models/ — Model Definitions

## Files
- `tft_wrapper.py` — TFT construction from `TimeSeriesDataSet`, `QuantileLoss([0.1, 0.5, 0.9])`, checkpoint loading
- `baselines.py` — `BaseBaseline` ABC + 10 subclasses, `compute_metrics()`, `BASELINE_NAME_MAP`
- `__init__.py` — Empty

## Baseline Classes (all in baselines.py)
`Persistence`, `LinearBaseline`, `ARIMABaseline`, `RandomForestBaseline`, `ExpSmoothingBaseline`, `XGBoostBaseline`, `SVRBaseline`, `GradientBoostingBaseline`, `ElasticNetBaseline`, `KNNBaseline`

Each has: `fit(train_df)`, `predict(train_df, n_steps)`, `save(path)`, `load(path)` (classmethod)
- Save/load uses `joblib` to `models/baselines/{model_name}/{reach_id}_{bank_side}.joblib`

## compute_metrics(actuals, preds)
Returns dict: `{NSE, RMSE, MAE, KGE}` — used by evaluation and API compare endpoint.

## BASELINE_NAME_MAP
Maps string keys (e.g., `"arima"`, `"random_forest"`) to their class objects.
