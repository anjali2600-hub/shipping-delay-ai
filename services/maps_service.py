"""
maps_service.py
Fetches distance/travel-time/traffic info from Google Maps Distance
Matrix API. Falls back to a mock estimator if no API key is set or
the API call fails, so the app never crashes due to a missing key.
"""

import os
import random
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"


def _mock_route_data(origin: str, destination: str) -> dict:
    """
    Deterministic-ish mock: same origin/destination pair always
    produces the same mock distance (seeded by string hash), so
    repeated demo runs behave consistently.
    """
    seed = abs(hash((origin.lower(), destination.lower()))) % (10 ** 6)
    rng = random.Random(seed)

    distance_km = round(rng.uniform(50, 2000), 1)
    avg_speed_kmh = rng.uniform(35, 60)
    duration_minutes = round((distance_km / avg_speed_kmh) * 60, 1)

    traffic_options = ["Low", "Medium", "High", "Severe"]
    traffic_level = rng.choice(traffic_options)

    # Traffic adds proportional extra time, mirrors real Distance Matrix behavior
    traffic_multiplier = {"Low": 1.0, "Medium": 1.15, "High": 1.35, "Severe": 1.6}
    duration_in_traffic_minutes = round(duration_minutes * traffic_multiplier[traffic_level], 1)

    return {
        "source": "mock",
        "distance_km": distance_km,
        "duration_minutes": duration_minutes,
        "duration_in_traffic_minutes": duration_in_traffic_minutes,
        "traffic_level": traffic_level,
    }


def _classify_traffic_from_ratio(ratio: float) -> str:
    """Classifies traffic severity from (duration_in_traffic / duration) ratio."""
    if ratio <= 1.05:
        return "Low"
    elif ratio <= 1.20:
        return "Medium"
    elif ratio <= 1.45:
        return "High"
    else:
        return "Severe"


def get_route_data(origin: str, destination: str) -> dict:
    """
    Main entry point. Returns distance, duration, and traffic level
    for a given origin/destination pair. Uses live Google Maps API
    if a key is configured, otherwise falls back to mock data.
    """
    if not GOOGLE_MAPS_API_KEY:
        return _mock_route_data(origin, destination)

    try:
        params = {
            "origins": origin,
            "destinations": destination,
            "departure_time": "now",  # required to get duration_in_traffic
            "key": GOOGLE_MAPS_API_KEY,
        }
        response = requests.get(DISTANCE_MATRIX_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()

        element = data["rows"][0]["elements"][0]
        if element.get("status") != "OK":
            # e.g. NOT_FOUND, ZERO_RESULTS -> fall back safely
            return _mock_route_data(origin, destination)

        distance_km = element["distance"]["value"] / 1000
        duration_minutes = element["duration"]["value"] / 60
        duration_in_traffic_minutes = element.get(
            "duration_in_traffic", element["duration"]
        )["value"] / 60

        ratio = duration_in_traffic_minutes / duration_minutes if duration_minutes else 1.0
        traffic_level = _classify_traffic_from_ratio(ratio)

        return {
            "source": "google_maps_api",
            "distance_km": round(distance_km, 1),
            "duration_minutes": round(duration_minutes, 1),
            "duration_in_traffic_minutes": round(duration_in_traffic_minutes, 1),
            "traffic_level": traffic_level,
        }

    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        # Any failure (network issue, bad response shape, quota exceeded, etc.)
        # -> fall back to mock rather than crashing the app
        print(f"[maps_service] Google Maps API call failed ({e}); using mock data.")
        return _mock_route_data(origin, destination)


if __name__ == "__main__":
    result = get_route_data("Delhi", "Mumbai")
    print("Route data:")
    for k, v in result.items():
        print(f"  {k}: {v}")