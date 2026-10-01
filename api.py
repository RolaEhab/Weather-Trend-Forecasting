"""
FastAPI + browser UI for the Weather Trend Forecasting project.

Run:
    uvicorn api:app --reload

Open in browser:
    http://127.0.0.1:8000/
"""

from numbers import Number
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from inference import (
    build_features,
    fetch_recent_weather,
    load_models,
    predict_temperature,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
LOCATIONS_FILE = BASE_DIR / "locations.csv"

app = FastAPI(
    title="Weather Trend Forecasting API",
    description=(
        "Next-day temperature forecasting using the trained "
        "CatBoost + Ridge ensemble."
    ),
    version="1.1.0",
)

# Load model artifacts
# -------------------------------------------------

try:
    catboost_model, ridge_model, ensemble_weights = load_models(MODEL_DIR)
except Exception as exc:
    catboost_model = None
    ridge_model = None
    ensemble_weights = None
    MODEL_LOAD_ERROR = str(exc)
else:
    MODEL_LOAD_ERROR = None



# Load list 
# -------------------------------------------------

try:
    locations_df = pd.read_csv(LOCATIONS_FILE)

    required_columns = {
        "country",
        "location_name",
        "latitude",
        "longitude",
    }

    missing_columns = required_columns - set(locations_df.columns)

    if missing_columns:
        raise ValueError(
            "locations.csv is missing: "
            + ", ".join(sorted(missing_columns))
        )

    locations_df = (
        locations_df[
            ["country", "location_name", "latitude", "longitude"]
        ]
        .dropna()
        .drop_duplicates()
        .sort_values(["country", "location_name"])
        .reset_index(drop=True)
    )

except Exception as exc:
    locations_df = None
    LOCATION_LOAD_ERROR = str(exc)
else:
    LOCATION_LOAD_ERROR = None


# Helpers
#-------------------------------------------------

def json_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, Number):
        return round(float(value), 4)

    return str(value)


def find_location(country: str, location_name: str) -> dict:
    if LOCATION_LOAD_ERROR:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load locations.csv: {LOCATION_LOAD_ERROR}",
        )

    match = locations_df[
        (locations_df["country"] == country)
        & (locations_df["location_name"] == location_name)
    ]

    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"'{location_name}' was not found under '{country}' "
                "in locations.csv."
            ),
        )

    row = match.iloc[0]

    return {
        "name": str(row["location_name"]),
        "country": str(row["country"]),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
    }


# Browser interface
# -------------------------------------------------

APP_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weather Trend Forecasting</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background: #f5f8fc;
            color: #17263f;
        }

        .hero {
            background: linear-gradient(120deg, #16395f, #317bb5);
            color: white;
            padding: 42px 24px;
        }

        .hero-inner {
            max-width: 980px;
            margin: auto;
        }

        .hero h1 {
            margin: 0 0 8px;
            font-size: 34px;
        }

        .hero p {
            margin: 0;
            opacity: 0.9;
            line-height: 1.5;
        }

        .container {
            max-width: 980px;
            margin: 30px auto;
            padding: 0 18px 40px;
        }

        .card {
            background: white;
            border: 1px solid #dde5ef;
            border-radius: 14px;
            padding: 24px;
            box-shadow: 0 6px 22px rgba(24, 52, 83, 0.06);
            margin-bottom: 22px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr auto;
            gap: 14px;
            align-items: end;
        }

        label {
            display: block;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 7px;
            color: #40536b;
        }

        select, button {
            width: 100%;
            min-height: 44px;
            border-radius: 9px;
            font-size: 15px;
        }

        select {
            border: 1px solid #cbd6e3;
            background: white;
            padding: 0 12px;
            color: #17263f;
        }

        select:disabled {
            background: #edf1f5;
            color: #8795a5;
        }

        button {
            border: none;
            background: #2d67b5;
            color: white;
            font-weight: 700;
            padding: 0 22px;
            cursor: pointer;
        }

        button:hover {
            background: #245796;
        }

        button:disabled {
            background: #9aabba;
            cursor: default;
        }

        .status {
            margin-top: 16px;
            font-size: 14px;
            color: #5b6c80;
        }

        .error {
            color: #a13c3c;
            font-weight: 700;
        }

        .result {
            display: none;
        }

        .forecast {
            display: grid;
            grid-template-columns: 1.2fr 1fr 1fr;
            gap: 14px;
            margin-top: 18px;
        }

        .metric {
            background: #f7fafc;
            border: 1px solid #e1e8f0;
            border-radius: 10px;
            padding: 16px;
        }

        .metric .label {
            color: #6a798b;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
        }

        .metric .value {
            margin-top: 6px;
            font-size: 25px;
            font-weight: 700;
        }

        .metric.primary {
            background: #edf5fd;
            border-color: #b9d5f2;
        }

        .details {
            margin-top: 22px;
        }

        details {
            border-top: 1px solid #e1e7ee;
            padding-top: 16px;
        }

        summary {
            cursor: pointer;
            font-weight: 700;
        }

        pre {
            white-space: pre-wrap;
            word-break: break-word;
            background: #f6f8fa;
            border-radius: 8px;
            padding: 14px;
            font-size: 12px;
            overflow-x: auto;
        }

        .small {
            font-size: 12px;
            color: #6c7b8e;
            margin-top: 14px;
            line-height: 1.5;
        }

        .links {
            margin-top: 12px;
            font-size: 13px;
        }

        .links a {
            color: #2d67b5;
        }

        @media (max-width: 760px) {
            .form-grid,
            .forecast {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>

<div class="hero">
    <div class="hero-inner">
        <h1>Weather Trend Forecasting</h1>
        <p>
            Select a country and one of its available locations to generate
            an approximately 24-hour-ahead temperature forecast.
        </p>
    </div>
</div>

<div class="container">

    <div class="card">
        <div class="form-grid">

            <div>
                <label for="country">Country</label>
                <select id="country">
                    <option value="">Select a country</option>
                </select>
            </div>

            <div>
                <label for="location">City / location</label>
                <select id="location" disabled>
                    <option value="">Select a country first</option>
                </select>
            </div>

            <div>
                <button id="predictButton" disabled>
                    Predict
                </button>
            </div>

        </div>

        <div id="status" class="status">
            Loading available countries...
        </div>

        <div class="links">
            API documentation:
            <a href="/docs" target="_blank">Swagger /docs</a>
        </div>
    </div>


    <div id="resultCard" class="card result">

        <h2 id="resultLocation"></h2>
        <div id="forecastOrigin" class="status"></div>

        <div class="forecast">

            <div class="metric primary">
                <div class="label">Ensemble forecast</div>
                <div id="ensemblePrediction" class="value"></div>
            </div>

            <div class="metric">
                <div class="label">CatBoost</div>
                <div id="catboostPrediction" class="value"></div>
            </div>

            <div class="metric">
                <div class="label">Ridge</div>
                <div id="ridgePrediction" class="value"></div>
            </div>

        </div>

        <div class="forecast">

            <div class="metric">
                <div class="label">Current temperature</div>
                <div id="currentTemperature" class="value"></div>
            </div>

            <div class="metric">
                <div class="label">24h previous temperature</div>
                <div id="previousTemperature" class="value"></div>
            </div>

            <div class="metric">
                <div class="label">Model weights</div>
                <div id="weights" class="value" style="font-size:18px"></div>
            </div>

        </div>

        <div class="details">
            <details>
                <summary>Show model input features</summary>
                <pre id="features"></pre>
            </details>
        </div>

        <p class="small">
            Weather inputs are retrieved from Open-Meteo. The trained model
            was evaluated on the Kaggle Global Weather Repository, so this
            interface demonstrates the inference pipeline rather than
            separately validated production performance.
        </p>

    </div>

</div>


<script>
const countrySelect = document.getElementById("country");
const locationSelect = document.getElementById("location");
const predictButton = document.getElementById("predictButton");
const statusBox = document.getElementById("status");
const resultCard = document.getElementById("resultCard");


async function loadCountries() {
    try {
        const response = await fetch("/countries");

        if (!response.ok) {
            throw new Error(await response.text());
        }

        const countries = await response.json();

        for (const country of countries) {
            const option = document.createElement("option");
            option.value = country;
            option.textContent = country;
            countrySelect.appendChild(option);
        }

        statusBox.textContent =
            `${countries.length} countries available from the project dataset.`;

    } catch (error) {
        statusBox.innerHTML =
            `<span class="error">Could not load countries: ${error.message}</span>`;
    }
}


async function loadLocations(country) {
    locationSelect.innerHTML =
        '<option value="">Loading locations...</option>';

    locationSelect.disabled = true;
    predictButton.disabled = true;
    resultCard.style.display = "none";

    if (!country) {
        locationSelect.innerHTML =
            '<option value="">Select a country first</option>';
        return;
    }

    try {
        const response = await fetch(
            `/locations?country=${encodeURIComponent(country)}`
        );

        if (!response.ok) {
            throw new Error(await response.text());
        }

        const locations = await response.json();

        locationSelect.innerHTML =
            '<option value="">Select a city / location</option>';

        for (const location of locations) {
            const option = document.createElement("option");
            option.value = location;
            option.textContent = location;
            locationSelect.appendChild(option);
        }

        locationSelect.disabled = false;

        statusBox.textContent =
            `${locations.length} location(s) available for ${country}.`;

    } catch (error) {
        statusBox.innerHTML =
            `<span class="error">Could not load locations: ${error.message}</span>`;
    }
}


async function makePrediction() {
    const country = countrySelect.value;
    const location = locationSelect.value;

    if (!country || !location) {
        return;
    }

    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";
    statusBox.textContent = "Fetching recent weather and running the ensemble...";

    try {
        const response = await fetch(
            `/predict?country=${encodeURIComponent(country)}`
            + `&location_name=${encodeURIComponent(location)}`
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Prediction request failed."
            );
        }

        document.getElementById("resultLocation").textContent =
            data.location;

        document.getElementById("forecastOrigin").textContent =
            `Forecast origin: ${data.forecast_origin}`;

        document.getElementById("ensemblePrediction").textContent =
            `${data.predictions.ensemble_celsius.toFixed(2)} °C`;

        document.getElementById("catboostPrediction").textContent =
            `${data.predictions.catboost_celsius.toFixed(2)} °C`;

        document.getElementById("ridgePrediction").textContent =
            `${data.predictions.ridge_celsius.toFixed(2)} °C`;

        document.getElementById("currentTemperature").textContent =
            `${data.current_temperature_celsius.toFixed(2)} °C`;

        document.getElementById("previousTemperature").textContent =
            `${data.previous_day_temperature_celsius.toFixed(2)} °C`;

        document.getElementById("weights").textContent =
            `${Math.round(data.ensemble_weights.catboost * 100)}% CatBoost`
            + ` + ${Math.round(data.ensemble_weights.ridge * 100)}% Ridge`;

        document.getElementById("features").textContent =
            JSON.stringify(data.input_features, null, 2);

        resultCard.style.display = "block";

        statusBox.textContent =
            "Prediction completed successfully.";

    } catch (error) {
        statusBox.innerHTML =
            `<span class="error">${error.message}</span>`;
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Predict";
    }
}


countrySelect.addEventListener("change", () => {
    loadLocations(countrySelect.value);
});


locationSelect.addEventListener("change", () => {
    predictButton.disabled =
        !countrySelect.value || !locationSelect.value;
});


predictButton.addEventListener("click", makePrediction);


loadCountries();
</script>

</body>
</html>
"""


# API endpoints
#-------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def browser_app():
    return HTMLResponse(APP_HTML)


@app.get("/health")
def health():
    return {
        "status": (
            "ok"
            if not MODEL_LOAD_ERROR and not LOCATION_LOAD_ERROR
            else "error"
        ),
        "models_loaded": MODEL_LOAD_ERROR is None,
        "locations_loaded": LOCATION_LOAD_ERROR is None,
        "model_error": MODEL_LOAD_ERROR,
        "location_error": LOCATION_LOAD_ERROR,
    }


@app.get("/countries")
def countries():
    if LOCATION_LOAD_ERROR:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load locations.csv: {LOCATION_LOAD_ERROR}",
        )

    return (
        locations_df["country"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


@app.get("/locations")
def locations(
    country: str = Query(..., description="Selected country"),
):
    if LOCATION_LOAD_ERROR:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load locations.csv: {LOCATION_LOAD_ERROR}",
        )

    available = (
        locations_df.loc[
            locations_df["country"] == country,
            "location_name",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not available:
        raise HTTPException(
            status_code=404,
            detail=f"No locations found for '{country}'.",
        )

    return available


@app.get("/predict")
def predict(
    country: str = Query(..., description="Country from the project dataset"),
    location_name: str = Query(
        ...,
        description="Location belonging to the selected country",
    ),
):
    if MODEL_LOAD_ERROR:
        raise HTTPException(
            status_code=500,
            detail=f"Model artifacts could not be loaded: {MODEL_LOAD_ERROR}",
        )

    try:
        # Use saved dataset coordinates
        location = find_location(country, location_name)

        # Fetch recent weather history
        weather = fetch_recent_weather(
            location["latitude"],
            location["longitude"],
        )

        catboost_frame, ridge_frame, metadata = build_features(
            weather,
            location,
        )

        predictions = predict_temperature(
            catboost_model,
            ridge_model,
            ensemble_weights,
            catboost_frame,
            ridge_frame,
        )

        catboost_features = {
            key: json_value(value)
            for key, value in catboost_frame.iloc[0].to_dict().items()
        }

        ridge_additional = {
            key: json_value(value)
            for key, value in ridge_frame.iloc[0].to_dict().items()
            if key not in catboost_features
        }

        return {
            "location": metadata["location"],
            "coordinates": {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
            },
            "forecast_origin": str(metadata["origin_time"]),
            "current_temperature_celsius": round(
                float(metadata["current_temperature"]), 2
            ),
            "previous_day_temperature_celsius": round(
                float(metadata["lag_1"]), 2
            ),
            "input_features": {
                "catboost": catboost_features,
                "ridge_additional": ridge_additional,
            },
            "predictions": {
                "catboost_celsius": round(
                    float(predictions["catboost"]), 2
                ),
                "ridge_celsius": round(
                    float(predictions["ridge"]), 2
                ),
                "ensemble_celsius": round(
                    float(predictions["ensemble"]), 2
                ),
            },
            "ensemble_weights": {
                "catboost": float(predictions["catboost_weight"]),
                "ridge": float(predictions["ridge_weight"]),
            },
            "prediction_horizon": "approximately 24 hours",
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
