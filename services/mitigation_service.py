"""
mitigation_service.py  (real-data version)
Rule-based engine: lists the likely causes of a delay and suggests actions
a dispatcher can take. The rules follow patterns seen in the Amazon Delivery
Dataset (traffic, weather, agent rating, peak hours and distance).
"""

from datetime import datetime

BAD_WEATHER = ("Cloudy", "Fog", "Stormy", "Sandstorms", "Windy")
BAD_TRAFFIC = ("High", "Jam")
PEAK_HOURS = list(range(7, 11)) + list(range(17, 22))
LONG_DISTANCE_KM = 14   # roughly the longest quarter of distances in the data
LOW_RATING = 4.5        # roughly the lowest quarter of agent ratings
TIME_MARGIN_MIN = 22    # about the typical error (RMSE) of the time model


def _order_hour(raw_input: dict) -> int:
    order_time = raw_input.get("order_time")
    if order_time:
        try:
            return datetime.fromisoformat(str(order_time)).hour
        except ValueError:
            pass
    return datetime.now().hour


def identify_delay_factors(raw_input: dict, prediction: dict) -> list:
    """Short, readable list of what is likely slowing this delivery down."""
    factors = []
    traffic = raw_input.get("traffic_level", "Low")
    weather = raw_input.get("weather_condition", "Sunny")
    distance = float(raw_input.get("distance_km", 0))
    rating = float(raw_input.get("agent_rating", 5))

    if traffic in BAD_TRAFFIC:
        factors.append(f"{traffic} traffic")
    if weather in BAD_WEATHER:
        factors.append(f"{weather} weather")
    if _order_hour(raw_input) in PEAK_HOURS:
        factors.append("Peak-hour delivery")
    if distance > LONG_DISTANCE_KM:
        factors.append(f"Long distance ({distance:.1f} km)")
    if rating < LOW_RATING:
        factors.append(f"Lower-rated agent ({rating:.1f})")

    if not factors:
        factors.append("No major risk factors identified")
    return factors


def get_actions(raw_input: dict, prediction: dict) -> list:
    """Actions a dispatcher can take, most useful first (max 3 returned)."""
    actions = []
    traffic = raw_input.get("traffic_level", "Low")
    weather = raw_input.get("weather_condition", "Sunny")
    distance = float(raw_input.get("distance_km", 0))
    rating = float(raw_input.get("agent_rating", 5))

    if traffic in BAD_TRAFFIC:
        actions.append("Pick a route that avoids congested roads.")
    if rating < LOW_RATING:
        actions.append("Assign a higher-rated agent.")
    if weather in BAD_WEATHER:
        actions.append("Add a time buffer and watch the weather.")
    if _order_hour(raw_input) in PEAK_HOURS:
        actions.append("Dispatch before the peak window if possible.")
    if distance > LONG_DISTANCE_KM:
        actions.append("Use the nearest available agent.")
    return actions[:3]


def get_recommendation(raw_input: dict, prediction: dict) -> str:
    """One short recommendation text (kept under the database column limit)."""
    risk = prediction.get("risk_level", "LOW")
    minutes = prediction.get("predicted_delivery_time_minutes")
    actions = get_actions(raw_input, prediction)

    if risk == "LOW" and not actions:
        return "Conditions look favourable. Standard dispatch is sufficient."

    intro = {
        "HIGH": "High risk of a long delay.",
        "MEDIUM": "Some risk of delay.",
    }.get(risk, "Low delay risk.")

    parts = [intro] + actions
    if risk in ("MEDIUM", "HIGH") and minutes:
        low = max(10, round(minutes - TIME_MARGIN_MIN))
        high = round(minutes + TIME_MARGIN_MIN)
        parts.append(
            f"Tell the customer to expect delivery in about {low}-{high} minutes."
        )
    return " ".join(parts)[:490]


def get_mitigation_summary(raw_input: dict, prediction: dict) -> dict:
    """Factors, actions and the recommendation text, ready for the app."""
    return {
        "main_factors": identify_delay_factors(raw_input, prediction),
        "actions": get_actions(raw_input, prediction),
        "recommendation": get_recommendation(raw_input, prediction),
    }


if __name__ == "__main__":
    tests = [
        (
            "Jam, cloudy, evening rush",
            {"traffic_level": "Jam", "weather_condition": "Cloudy",
             "distance_km": 8.5, "agent_rating": 4.5,
             "order_time": "2026-10-19 18:30:00"},
            {"risk_level": "MEDIUM", "predicted_delivery_time_minutes": 141.9},
        ),
        (
            "Low traffic, sunny, late morning",
            {"traffic_level": "Low", "weather_condition": "Sunny",
             "distance_km": 3.0, "agent_rating": 4.9,
             "order_time": "2026-10-19 11:00:00"},
            {"risk_level": "LOW", "predicted_delivery_time_minutes": 95.6},
        ),
        (
            "Fog, high traffic, long distance, low-rated agent",
            {"traffic_level": "High", "weather_condition": "Fog",
             "distance_km": 16.0, "agent_rating": 4.2,
             "order_time": "2026-10-19 08:00:00"},
            {"risk_level": "HIGH", "predicted_delivery_time_minutes": 190.0},
        ),
    ]

    for name, sample, pred in tests:
        result = get_mitigation_summary(sample, pred)
        print(f"\n{name}")
        print(f"  Factors: {result['main_factors']}")
        print(f"  Recommendation: {result['recommendation']}")