# Aluminium Price Forecasting

Forecasting the daily percentage change of the aluminium price (1 day up to 3 weeks ahead) using historical prices of aluminium, other metals (copper, gold, lead, nickel, silver, zinc) and the S&P 500 (2014–2026).
University project at OTH Regensburg, comparing classical baselines with ML and deep-learning models.

## Approach

1. **Data pipeline** – cleaning, merging and resampling of multiple market time series, chronological train/val/test split
2. **Feature engineering** – lags, rolling statistics, log returns, cross-market covariates
3. **Models**
   - Baselines: naive (last value), mean
   - ML: Linear/Ridge, SVR, KNN, Random Forest, XGBoost, LightGBM
   - DL: LSTM, GRU, Transformer (PyTorch, walk-forward evaluation)
4. **Evaluation** – MAE, RMSE, R², directional accuracy

## Key finding

Daily metal returns have a very low signal-to-noise ratio. On the test set, neither the ML nor the DL models clearly beat the simple baselines (R² ≈ 0, directional accuracy ≈ 50 %).
The project shows *why* honest baselines and walk-forward validation matter in financial forecasting.

| Model (test, log returns) | MAE | RMSE | Directional acc. |
|---|---|---|---|
| LSTM | 0.0195 | 0.0257 | 0.49 |
| GRU | 0.0200 | 0.0261 | 0.49 |
| Transformer | 0.0200 | 0.0264 | 0.49 |

Full results: [`results/`](results/)

## Structure

```
src/timeseries_forecast/   # data prep, features, training, evaluation
Notebooks/                 # exploration, feature engineering, model experiments
data/                      # raw, cleaned and final feature sets
results/                   # metrics and predictions
Latex/                     # project report (German)
```

## Quickstart

```bash
uv sync
uv run jupyter lab   # notebooks 01 → 07
```

## Tech

Python · pandas · scikit-learn · XGBoost · LightGBM · PyTorch · Darts · uv
