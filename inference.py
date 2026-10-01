"""
Inference script for the Weather Trend Forecasting project.

This script:
1. Gets latitude/longitude with Open-Meteo geocoding.
2. Fetches current + recent hourly weather from Open-Meteo.
3. Recreates the features used by the trained models.
4. Loads the saved models and weights.
5. Produces a next-day temperature forecast.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
from catboost import CatBoostRegressor


DEFAULT_MODEL_DIR = Path("models")

CATBOOST_FILENAME = "final_catboost.cbm"
RIDGE_FILENAME = "ridge_pipeline.joblib"
WEIGHTS_FILENAME = "ensemble_weights.joblib"

FINAL_CATBOOST_FEATURES = [
    "temperature_celsius",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "temp_change_1d",
    "temp_mean_3d",
    "humidity",
    "pressure_mb",
    "wind_kph",
    "log_precip_mm",
    "cloud",
    "uv_index",
    "day_of_year_sin",
    "day_of_year_cos",
    "latitude",
    "longitude",
    "country",
]

RIDGE_NUMERIC_FEATURES = [
    "temperature_celsius",
    "lag_1",
    "humidity",
    "pressure_mb",
    "wind_kph",
    "log_precip_mm",
    "visibility_km",
    "cloud",
    "uv_index",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "latitude",
    "longitude",
]

RIDGE_CATEGORICAL_FEATURES = [
    "country",
    "location_name",
]

RIDGE_FEATURES = RIDGE_NUMERIC_FEATURES + RIDGE_CATEGORICAL_FEATURES

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "cloud_cover",
    "visibility",
    "uv_index",
    "wind_speed_10m",
]


class InferenceError(RuntimeError):
    pass


def geocode_location(city: str, country: str | None = None) -> dict:
    params = {
        "name": city,
        "count": 10,
        "language": "en",
        "format": "json",
    }

    response = requests.get(GEOCODING_URL, params=params, timeout=20)
    response.raise_for_status()
    results = response.json().get("results", [])

    if not results:
        raise InferenceError(f"No location found for '{city}'.")

    if country:
        country_lower = country.strip().lower()
        matching = [
            item
            for item in results
            if str(item.get("country", "")).strip().lower() == country_lower
            or str(item.get("country_code", "")).strip().lower() == country_lower
        ]
        if matching:
            results = matching

    result = results[0]

    return {
        "name": result["name"],
        "country": result.get("country", country or "Unknown"),
        "latitude": float(result["latitude"]),
        "longitude": float(result["longitude"]),
    }


def fetch_recent_weather(latitude: float, longitude: float) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "past_days": 8,
        "forecast_days": 1,
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    response = requests.get(FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise InferenceError("Weather API returned no hourly data.")

    weather = pd.DataFrame(hourly)
    weather["time"] = pd.to_datetime(weather["time"])
    weather = weather.sort_values("time").reset_index(drop=True)

    return weather


def nearest_value(
    weather: pd.DataFrame,
    target_time: pd.Timestamp,
    column: str,
    tolerance_hours: float = 2.0,
) -> float:
    deltas = (weather["time"] - target_time).abs()
    idx = deltas.idxmin()

    if deltas.loc[idx] > pd.Timedelta(hours=tolerance_hours):
        raise InferenceError(
            f"Could not find '{column}' close enough to {target_time}."
        )

    value = weather.loc[idx, column]

    if pd.isna(value):
        raise InferenceError(
            f"Missing '{column}' value near {target_time}."
        )

    return float(value)


def choose_forecast_origin(weather: pd.DataFrame) -> pd.Timestamp:
    now_local_naive = pd.Timestamp.now().floor("h")
    eligible = weather.loc[weather["time"] <= now_local_naive, "time"]

    if eligible.empty:
        return weather["time"].iloc[0]

    return eligible.iloc[-1]


def build_features(
    weather: pd.DataFrame,
    location: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    origin_time = choose_forecast_origin(weather)

    current_temp = nearest_value(weather, origin_time, "temperature_2m")
    lag_1 = nearest_value(
        weather, origin_time - pd.Timedelta(hours=24), "temperature_2m"
    )
    lag_2 = nearest_value(
        weather, origin_time - pd.Timedelta(hours=48), "temperature_2m"
    )
    lag_3 = nearest_value(
        weather, origin_time - pd.Timedelta(hours=72), "temperature_2m"
    )
    lag_7 = nearest_value(
        weather, origin_time - pd.Timedelta(hours=168), "temperature_2m"
    )

    humidity = nearest_value(weather, origin_time, "relative_humidity_2m")
    pressure_mb = nearest_value(weather, origin_time, "pressure_msl")
    wind_kph = nearest_value(weather, origin_time, "wind_speed_10m")
    precip_mm = nearest_value(weather, origin_time, "precipitation")
    cloud = nearest_value(weather, origin_time, "cloud_cover")
    visibility_m = nearest_value(weather, origin_time, "visibility")
    uv_index = nearest_value(weather, origin_time, "uv_index")

    day_of_year = origin_time.dayofyear
    month = origin_time.month

    features = {
        "temperature_celsius": current_temp,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "lag_7": lag_7,
        "temp_change_1d": current_temp - lag_1,
        "temp_mean_3d": np.mean([current_temp, lag_1, lag_2]),
        "humidity": humidity,
        "pressure_mb": pressure_mb,
        "wind_kph": wind_kph,
        "precip_mm": precip_mm,
        "log_precip_mm": np.log1p(max(precip_mm, 0.0)),
        "visibility_km": visibility_m / 1000.0,
        "cloud": cloud,
        "uv_index": uv_index,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "country": str(location["country"]),
        "location_name": str(location["name"]),
    }

    catboost_frame = pd.DataFrame(
        [{feature: features[feature] for feature in FINAL_CATBOOST_FEATURES}]
    )

    ridge_frame = pd.DataFrame(
        [{feature: features[feature] for feature in RIDGE_FEATURES}]
    )

    metadata = {
        "origin_time": origin_time,
        "current_temperature": current_temp,
        "lag_1": lag_1,
        "location": f'{location["name"]}, {location["country"]}',
    }

    return catboost_frame, ridge_frame, metadata


def load_models(model_dir: Path):
    catboost_path = model_dir / CATBOOST_FILENAME
    ridge_path = model_dir / RIDGE_FILENAME
    weights_path = model_dir / WEIGHTS_FILENAME

    missing = [
        str(path)
        for path in [catboost_path, ridge_path, weights_path]
        if not path.exists()
    ]

    if missing:
        joined = "\n  - ".join(missing)
        raise InferenceError(
            "Missing model artifact(s):\n  - "
            + joined
            + "\nPlace them inside the models/ folder or pass --model-dir."
        )

    catboost_model = CatBoostRegressor()
    catboost_model.load_model(str(catboost_path))

    ridge_model = joblib.load(ridge_path)
    weights = joblib.load(weights_path)

    if "catboost_weight" not in weights or "ridge_weight" not in weights:
        raise InferenceError(
            "ensemble_weights.joblib must contain "
            "'catboost_weight' and 'ridge_weight'."
        )

    return catboost_model, ridge_model, weights


def predict_temperature(
    catboost_model,
    ridge_model,
    weights: dict,
    catboost_frame: pd.DataFrame,
    ridge_frame: pd.DataFrame,
) -> dict:
    catboost_prediction = float(catboost_model.predict(catboost_frame)[0])
    ridge_prediction = float(ridge_model.predict(ridge_frame)[0])

    catboost_weight = float(weights["catboost_weight"])
    ridge_weight = float(weights["ridge_weight"])

    ensemble_prediction = (
        catboost_weight * catboost_prediction
        + ridge_weight * ridge_prediction
    )

    return {
        "catboost": catboost_prediction,
        "ridge": ridge_prediction,
        "ensemble": ensemble_prediction,
        "catboost_weight": catboost_weight,
        "ridge_weight": ridge_weight,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict temperature approximately 24 hours ahead."
    )
    parser.add_argument(
        "--city",
        required=True,
        help='City name, e.g. "Alexandria"',
    )
    parser.add_argument(
        "--country",
        default=None,
        help='Optional country name or code, e.g. "Egypt"',
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Folder containing saved model artifacts (default: models)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        location = geocode_location(args.city, args.country)

        print(
            f'Location: {location["name"]}, {location["country"]} '
            f'({location["latitude"]:.4f}, {location["longitude"]:.4f})'
        )
        print("Fetching recent weather data...")

        weather = fetch_recent_weather(
            location["latitude"],
            location["longitude"],
        )

        catboost_frame, ridge_frame, metadata = build_features(
            weather,
            location,
        )

        catboost_model, ridge_model, weights = load_models(
            Path(args.model_dir)
        )

        predictions = predict_temperature(
            catboost_model,
            ridge_model,
            weights,
            catboost_frame,
            ridge_frame,
        )

        print()
        print("Weather input")
        print("-------------")
        print(f'Forecast origin:      {metadata["origin_time"]}')
        print(f'Current temperature:  {metadata["current_temperature"]:.2f} °C')
        print(f'Previous-day temp:    {metadata["lag_1"]:.2f} °C')

        print()
        print("Model predictions")
        print("-----------------")
        print(
            f'CatBoost ({predictions["catboost_weight"]:.0%}): '
            f'{predictions["catboost"]:.2f} °C'
        )
        print(
            f'Ridge    ({predictions["ridge_weight"]:.0%}): '
            f'{predictions["ridge"]:.2f} °C'
        )
        print(
            f'Ensemble forecast:    {predictions["ensemble"]:.2f} °C'
        )
        print()
        print("Prediction horizon: approximately 24 hours ahead.")

    except requests.RequestException as exc:
        print(f"API/network error: {exc}", file=sys.stderr)
        sys.exit(1)
    except InferenceError as exc:
        print(f"Inference error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
