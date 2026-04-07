# Bayesian Cointegrated Basket Optimizer

## Overview
This repository implements a hybrid statistical arbitrage strategy. It utilizes the **Johansen Test** to identify structural equilibrium between non-stationary assets and leverages **Bayesian Optimization** to maximize the risk-adjusted returns (Sharpe Ratio) of the resulting synthetic spread.

## Key Features
- **Econometric Core:** Vector Error Correction Model (VECM) and Johansen trace statistics for baseline cointegration.
- **Bayesian Weight Optimization:** 300 trials using **Optuna** to refine portfolio weights beyond simple eigenvectors.
- **Robust Validation:** Strict In-Sample (2021-2024) and Out-Of-Sample (2024-2025) data split to mitigate look-ahead bias.
- **Mean-Reversion Analytics:** Calculation of the spread half-life and maximum drawdown for tradability assessment.

## Architecture
- `MarketData`: Ingestion and preprocessing of 1,255 daily adjusted closing prices.
- `CointegrationEngine`: Implementation of the Johansen procedure to find the primary cointegrating vector.
- `BayesianOptimizer`: Objective function targeting the In-Sample Sharpe Ratio (peaking at 0.8436).
- `Visualizer`: Generates a 4-panel dashboard including normalization, spread performance, and optimization history.

## Performance Summary (AAPL/MSFT/GOOGL)
The optimizer shifted the baseline toward a heavy MSFT long and an AAPL short, resulting in:
- **OOS Sharpe Ratio:** 0.4759
- **Max Drawdown:** -0.2319
- **Half-Life:** 163.97 days

## Quick Start
1. `pip install -r requirements.txt`
2. `python basket_optimizer.py`

---
**Author:** TUNG Alexandre | **Affiliation:** EFREI Paris
