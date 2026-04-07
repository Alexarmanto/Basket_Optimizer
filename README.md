# Bayesian Cointegrated Basket Optimizer

## Overview
This repository implements a hybrid statistical arbitrage strategy. [cite_start]It utilizes the **Johansen Test** to identify structural equilibrium between non-stationary assets and leverages **Bayesian Optimization** to maximize the risk-adjusted returns (Sharpe Ratio) of the resulting synthetic spread. [cite: 8, 14, 18]

## Key Features
- [cite_start]**Econometric Core:** Vector Error Correction Model (VECM) and Johansen trace statistics for baseline cointegration. [cite: 18, 30]
- [cite_start]**Bayesian Weight Optimization:** 300 trials using **Optuna** to refine portfolio weights beyond simple eigenvectors. [cite: 10, 31]
- [cite_start]**Robust Validation:** Strict 70/30 In-Sample (2021-2024) and Out-Of-Sample (2024-2025) data split to mitigate look-ahead bias. [cite: 26, 27, 28]
- [cite_start]**Mean-Reversion Analytics:** Calculation of the spread half-life and maximum drawdown for tradability assessment. [cite: 41]

## Architecture
- [cite_start]`MarketData`: Ingestion and preprocessing of 1,255 daily adjusted closing prices. [cite: 26]
- [cite_start]`CointegrationEngine`: Implementation of the Johansen procedure to find the primary cointegrating vector. [cite: 18]
- [cite_start]`BayesianOptimizer`: Objective function targeting the In-Sample Sharpe Ratio (peaking at 0.8436). [cite: 23, 31]
- [cite_start]`Visualizer`: Generates a 4-panel dashboard including normalization, spread performance, and optimization history. [cite: 43, 94]

## Performance Summary (AAPL/MSFT/GOOGL)
[cite_start]The optimizer shifted the baseline toward a heavy MSFT long and an AAPL short, resulting in: [cite: 34, 35]
- [cite_start]**OOS Sharpe Ratio:** 0.4759 [cite: 39]
- [cite_start]**Max Drawdown:** -0.2319 [cite: 40]
- [cite_start]**Half-Life:** 163.97 days [cite: 41]

## Quick Start
1. `pip install -r requirements.txt`
2. `python basket_optimizer.py`

---
**Author:** TUNG Alexandre | [cite_start]**Affiliation:** EFREI Paris [cite: 3, 4]
