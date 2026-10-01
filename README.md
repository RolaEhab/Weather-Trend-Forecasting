# Weather Trend Forecasting

Global weather analysis and next-day temperature forecasting using machine learning and ensemble modeling.

## Overview

This project analyzes the Kaggle Global Weather Repository dataset and builds models to predict temperature approximately 24 hours ahead.

The project includes:

- Data cleaning and preprocessing
- Exploratory data analysis
- Anomaly and outlier investigation
- Feature engineering
- Multiple forecasting models
- Ensemble modeling
- Feature importance and SHAP analysis
- Forecast error analysis

## Dataset

The project uses the Global Weather Repository dataset from Kaggle, containing daily weather observations for cities around the world.

## Models

The following approaches were evaluated:

- Persistence baseline
- Ridge Regression
- HistGradientBoosting
- XGBoost
- CatBoost
- Weighted CatBoost + Ridge ensemble

## Final Results

The best-performing model was a weighted CatBoost + Ridge ensemble.

Test performance:

- MAE: **1.36°C**
- RMSE: **1.94°C**
- R²: **0.922**

## Key Findings

- Recent temperature history was the strongest source of predictive signal.
- Multi-day temperature lags improved forecasting performance.
- Air-quality features provided little additional benefit for temperature prediction.
- Forecast errors increased during abrupt temperature changes.
- Combining CatBoost with Ridge produced a small improvement over CatBoost alone.

## Repository Contents

- Main analysis notebook
- Project report / presentation
- Requirements file
- Supporting documentation

## Tools

Python, pandas, NumPy, scikit-learn, CatBoost, XGBoost, SHAP, Matplotlib, and Seaborn.

## How to Run

To be added.

## Dataset Source

Kaggle Global Weather Repository
