# Weather Trend Forecasting

Global weather analysis and next-day temperature forecasting using machine learning and ensemble modeling.

## Overview

This project analyzes the Kaggle Global Weather Repository dataset and builds models to predict temperature approximately 24 hours ahead.

The project includes:

- Data cleaning and preprocessing
- Exploratory data analysis
- Anomaly and outlier investigation
- Climate and geographical analysis
- Air-quality analysis
- Feature engineering
- Multiple forecasting models
- Ensemble modeling
- Feature importance and SHAP analysis
- Forecast error analysis
- A FastAPI inference application for live predictions

## Dataset

The project uses the **Global Weather Repository** dataset from Kaggle, which contains daily weather observations for cities around the world and includes more than 40 weather, geographical, and air-quality features.

[Global Weather Repository on Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)

## Models

The following approaches were evaluated:

- Persistence baseline
- Ridge Regression
- HistGradientBoosting
- XGBoost
- CatBoost
- Weighted CatBoost + Ridge ensemble

## Final Results

The best-performing model was a weighted **CatBoost + Ridge ensemble** using:

- 70% CatBoost
- 30% Ridge

Test performance:

- MAE: **1.35°C**
- RMSE: **1.93°C**
- R²: **0.923**

## Key Findings

- Recent temperature history was the strongest source of predictive signal.
- Multi-day temperature lags improved forecasting performance.
- Air-quality features provided little additional benefit for next-day temperature prediction.
- Forecast errors increased during larger day-to-day temperature changes.
- The model tended to overpredict during cooling events and underpredict during warming events.
- Missing recent temperature history was associated with higher forecast error.
- Combining CatBoost with Ridge produced a small improvement over CatBoost alone.

## Repository Contents

```text
Weather-Trend-Forecasting/
│
├── models/
│   ├── final_catboost.cbm
│   ├── ridge_pipeline.joblib
│   └── ensemble_weights.joblib
│
├── api.py
├── inference.py
├── locations.csv
├── requirements.txt
├── WeatherForecastingNotebook.ipynb
├── WeatherForecast_presentation.pptx
├── README.md
└── .gitignore
```

- **Notebook** — complete data cleaning, EDA, feature engineering, modeling, and evaluation
- **Presentation** — summary of the methodology, analysis, and results
- **api.py** — FastAPI backend and browser interface
- **inference.py** — feature preparation and model inference logic
- **locations.csv** — countries, locations, and coordinates used by the application
- **models/** — saved CatBoost model, Ridge preprocessing/model pipeline, and ensemble weights
- **requirements.txt** — Python package requirements

## Tools

Python, pandas, NumPy, scikit-learn, CatBoost, XGBoost, SHAP, Matplotlib, Seaborn, FastAPI, Uvicorn, and Requests.

## How to Run

> **Python 3.11 is recommended.**

### 1. Clone the repository

```powershell
git clone https://github.com/RolaEhab/Weather-Trend-Forecasting.git
cd Weather-Trend-Forecasting
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate the environment again:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the required packages

```powershell
pip install -r requirements.txt
```

The saved Ridge pipeline was created using `scikit-learn==1.6.1`, so this version is pinned in the requirements file for compatibility.

### 4. Run the FastAPI application

```powershell
uvicorn api:app --reload
```

If `uvicorn` is not recognized:

```powershell
python -m uvicorn api:app --reload
```

### 5. Open the application

Open the following address in your browser:

```text
http://127.0.0.1:8000/
```

The application allows you to:

- Select a country
- Select a location available within that country
- Retrieve recent weather observations
- Generate the features required by the trained models
- Produce CatBoost and Ridge predictions
- Combine them using the saved ensemble weights
- View the final next-day temperature forecast

### API Documentation

FastAPI also provides interactive Swagger documentation at:

```text
http://127.0.0.1:8000/docs
```

## Inference Pipeline

The application retrieves recent weather observations from Open-Meteo and recreates the features required by the trained models, including:

- Current weather conditions
- Temperature lags
- Recent temperature change
- Three-day mean temperature
- Seasonal features
- Geographic information

The CatBoost and Ridge models generate separate predictions, which are combined using the saved ensemble weights.

## Dataset Source

[Global Weather Repository — Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)

## Application Screenshots

### Forecast Input and Prediction

The application allows the user to select a country and location, retrieve recent weather data, and generate the next-day temperature forecast.

![Forecast Input and Prediction](screenshots/Screenshot%202026-10-01%20162934.png)

### Model Input Features

The application also displays the features generated for the selected location before they are passed to the forecasting models.

![Model Input Features](screenshots/Screenshot%202026-10-01%20162933.png)

