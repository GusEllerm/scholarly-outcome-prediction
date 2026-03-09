# Experiment log

Lightweight log for manual runs and observations.  
Run artifacts in `artifacts/metrics/*.json` are self-describing (target_mode, model, features, split, dataset_id, etc.).

## Template

| Date | Experiment | Data config | Target mode | Model | Notes | RMSE | MAE | R² |
|------|------------|-------------|-------------|-------|-------|------|-----|-----|
| YYYY-MM-DD | baseline_regression | openalex_sample_100 | proxy | baseline | … | … | … | … |
| YYYY-MM-DD | ridge_regression | openalex_pilot | proxy | ridge | … | … | … | … |
| YYYY-MM-DD | xgb_regression | openalex_pilot | proxy | xgboost | … | … | … | … |

## Runs

(Record runs and any changes to data, features, or code that affect results.)
