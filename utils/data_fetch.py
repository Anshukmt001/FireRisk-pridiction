"""
utils/data_fetch.py
Fetches real-time weather data from OpenWeatherMap API.
Falls back to simulated data if the API key is missing / quota exceeded.
Reverse geocoding via Nominatim (OpenStreetMap) for location names.
"""

import os
import random
import logging
import time
import requests

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

# ── Set your API key here OR export it as an env variable ─────────────────────
# export OPENWEATHER_API_KEY="your_key_here"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def fetch_weather(lat: float, lon: float) -> dict:
    """
    Fetch current weather for (lat, lon).
    Returns a normalised dict with keys:
        temperature, humidity, wind_speed, city, description, icon
    Raises RuntimeError if fetch fails and no fallback is desired.
    """
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "your_key_here":
        logger.warning("No API key set – returning simulated weather data.")
        return _simulated_weather(lat, lon)

    params = {
        "lat":   lat,
        "lon":   lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        return {
            "temperature": round(data["main"]["temp"],    1),
            "humidity":    round(data["main"]["humidity"], 1),
            "wind_speed":  round(data["wind"]["speed"] * 3.6, 1),   # m/s → km/h
            "city":        data.get("name", "Unknown"),
            "description": data["weather"][0]["description"].title(),
            "icon":        data["weather"][0]["icon"],
            "source":      "live",
        }

    except requests.RequestException as exc:
        logger.error("OpenWeatherMap request failed: %s – falling back to simulation.", exc)
        result = _simulated_weather(lat, lon)
        result["error"] = str(exc)
        return result


def _simulated_weather(lat: float, lon: float) -> dict:
    """
    Generate plausible weather data based on latitude (tropical islands are hot & humid).
    """
    # Tropical / subtropical bias for island latitudes
    abs_lat = abs(lat)
    base_temp  = max(15, 38 - abs_lat * 0.5)
    base_hum   = max(20, 85 - abs_lat * 0.4)

    temperature = round(random.gauss(base_temp, 4),  1)
    humidity    = round(random.gauss(base_hum,  10), 1)
    wind_speed  = round(random.uniform(2, 55),        1)

    # Clamp
    temperature = max(10,  min(50, temperature))
    humidity    = max(5,   min(98, humidity))
    wind_speed  = max(0,   min(80, wind_speed))

    return {
        "temperature": temperature,
        "humidity":    humidity,
        "wind_speed":  wind_speed,
        "city":        "Simulated Location",
        "description": "Simulated Data",
        "icon":        "01d",
        "source":      "simulated",
    }


def get_drought_index(lat: float, lon: float) -> float:
    """
    Approximate KBDI-like drought index.
    In a production system you would call a specialised drought API
    (e.g. US Drought Monitor, Copernicus Climate Data Store).
    Here we simulate it based on latitude/season.
    """
    # Northern-hemisphere summer proxy
    import datetime
    month = datetime.datetime.utcnow().month
    summer_factor = abs(6.5 - month) / 6.5          # peaks at month 1 & 12
    base = 200 + (1 - summer_factor) * 400
    return round(random.gauss(base, 80), 1)


def get_vegetation_index(lat: float, lon: float) -> float:
    """
    Simulate NDVI-like vegetation density (0–1).
    Tropical islands → denser vegetation.
    """
    abs_lat = abs(lat)
    base = max(0.2, 0.85 - abs_lat * 0.01)
    return round(random.gauss(base, 0.08), 2)


def get_slope(lat: float, lon: float) -> float:
    """
    Simulate average terrain slope in degrees.
    Islands are often mountainous.
    """
    return round(random.uniform(5, 35), 1)


def fetch_all_features(lat: float, lon: float) -> dict:
    """
    Aggregate all features needed by the ML model.
    """
    weather  = fetch_weather(lat, lon)
    drought  = get_drought_index(lat, lon)
    veg      = get_vegetation_index(lat, lon)
    slope    = get_slope(lat, lon)

    return {
        # ML model features
        "temperature":   weather["temperature"],
        "humidity":      weather["humidity"],
        "wind_speed":    weather["wind_speed"],
        "drought_index": drought,
        "vegetation":    veg,
        "slope":         slope,
        # Display extras
        "city":          weather.get("city", ""),
        "description":   weather.get("description", ""),
        "icon":          weather.get("icon", "01d"),
        "source":        weather.get("source", "unknown"),
    }


def reverse_geocode(lat: float, lon: float) -> dict:
    """
    Reverse geocode (lat, lon) to a human-readable location name using Nominatim.
    Returns a dict with display_name, city, state, country.
    """
    try:
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "addressdetails": 1,
            "zoom": 10,
        }
        headers = {
            "User-Agent": "ForestGuard/1.0 (Fire Risk Prediction App)",
            "Accept": "application/json"
        }
        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        address = data.get("address", {})

        city = address.get("city") or address.get("town") or address.get("village") or address.get("county") or ""
        state = address.get("state") or address.get("region") or ""
        country = address.get("country") or ""
        display_name = data.get("display_name", "")

        return {
            "display_name": display_name,
            "city": city,
            "state": state,
            "country": country,
            "success": True
        }
    except Exception as e:
        logger.warning(f"Reverse geocoding failed: {e}")
        return {"display_name": "", "city": "", "state": "", "country": "", "success": False}