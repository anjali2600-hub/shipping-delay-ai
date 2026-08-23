"""
weather_service.py
Fetches current weather (temperature, rain, condition, wind, humidity)
from OpenWeather's Current Weather API. Falls back to a mock weather
generator if no API key is set or the API call fails.
"""

import os
import random
import requests
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Heavy Rain", "Storm", "Fog"]


def _mock_weather_data(location: str) -> dict:
    """
    Deterministic mock weather, seeded by location name so repeated
    calls for the same city return consistent results in a demo.
    """
    seed = abs(hash(location.lower())) % (10 ** 6)
    rng = random.Random(seed)

    condition = rng.choice(WEATHER_CONDITIONS)
    temperature = round(rng.uniform(15, 42), 1)

    rainfall = 0.0
    if condition in ("Rain", "Heavy Rain", "Storm"):
        rainfall = round(rng.uniform(5, 80), 1)
    elif condition == "Cloudy":
        rainfall = round(rng.uniform(0, 5), 1)

    wind_speed = round(rng.uniform(2, 35), 1)  # km/h
    humidity = rng.randint(30, 95)  # %

    return {
        "source": "mock",
        "condition": condition,
        "temperature": temperature,
        "rainfall_mm": rainfall,
        "wind_speed_kmh": wind_speed,
        "humidity_percent": humidity,
    }


def _map_owm_condition(owm_main: str, rain_1h: float) -> str:
    """
    Maps OpenWeather's 'main' condition field to our project's
    weather_condition categories.
    """
    owm_main = owm_main.lower()

    if owm_main in ("thunderstorm",):
        return "Storm"
    if owm_main in ("rain", "drizzle"):
        return "Heavy Rain" if rain_1h and rain_1h > 10 else "Rain"
    if owm_main in ("fog", "mist", "haze"):
        return "Fog"
    if owm_main in ("clouds",):
        return "Cloudy"
    return "Clear"


def get_weather_data(location: str) -> dict:
    """
    Main entry point. Returns weather info for a given location
    (city name). Uses live OpenWeather API if a key is configured,
    otherwise falls back to mock data.
    """
    if not OPENWEATHER_API_KEY:
        return _mock_weather_data(location)

    try:
        params = {
            "q": location,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",  # Celsius, km/h-equivalent
        }
        response = requests.get(CURRENT_WEATHER_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()

        owm_main = data["weather"][0]["main"]
        rain_1h = data.get("rain", {}).get("1h", 0.0)
        condition = _map_owm_condition(owm_main, rain_1h)

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed_ms = data["wind"]["speed"]
        wind_speed_kmh = round(wind_speed_ms * 3.6, 1)

        return {
            "source": "openweather_api",
            "condition": condition,
            "temperature": round(temperature, 1),
            "rainfall_mm": round(rain_1h, 1),
            "wind_speed_kmh": wind_speed_kmh,
            "humidity_percent": humidity,
        }

    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        print(f"[weather_service] OpenWeather API call failed ({e}); using mock data.")
        return _mock_weather_data(location)


if __name__ == "__main__":
    result = get_weather_data("Mumbai")
    print("Weather data:")
    for k, v in result.items():
        print(f"  {k}: {v}")