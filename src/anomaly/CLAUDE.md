# src/anomaly/ — Anomaly & Changepoint Detection

## Files
- `changepoint.py` — PELT changepoint detection with protection signature identification
- `autoencoder.py` — VAE for reconstruction-based anomaly scoring
- `__init__.py` — Empty

## PELT Changepoint Detection
- Detects structural breaks in bankline variance using Pruned Exact Linear Time algorithm
- `detect_changepoints_pelt(df)` → list of `ChangepointResult`
- Each result: reach_id, bank_side, changepoint_year, variance_before, variance_after, variance_reduction, is_protection_signature

## Protection Signatures
Variance reduction ≥ 70% after changepoint suggests bank protection works (revetments, groynes). Flagged as `is_protection_signature=True`.

## VAE Autoencoder
Reconstruction-based anomaly scoring for detecting unusual bankline patterns.
