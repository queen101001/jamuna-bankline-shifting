# src/data/ — Data Loading & Preprocessing

## Files
- `loader.py` — Loads Excel with 2-row hierarchical header into tidy long-format DataFrame
- `preprocessing.py` — Imputation, feature engineering, categorical encoding, optional RobustScaler
- `dataset.py` — PyTorch Forecasting `TimeSeriesDataSet` builder (encoder_length=10, prediction_length=5)
- `__init__.py` — Empty

## Data Source
`data/raw/Distances (1).xlsx` — 50 river reaches, 27 observed years (1991–2020), left/right bank distances

## Excel Format (2-row header)
Row 1: "Reaches" | "Distance(1991)" | (merged) | "Distance(1993)" | ...
Row 2: (empty)   | "Right Bank (m)" | "Left Bank (m)" | "Right Bank (m)" | ...

## Key Functions
- `load_long_dataframe()` → DataFrame with columns: reach_id, bank_side, year, bank_distance
- `get_series_id(reach_id, bank_side)` → e.g., "R01_left"
- `run_preprocessing_pipeline()` → adds features: erosion_indicator, rate_of_change, rolling_mean_3, net_channel_erosion
- `temporal_split(df, settings)` → train/val/test DataFrames
- `build_prediction_dataset()` → TimeSeriesDataSet for PyTorch Forecasting
