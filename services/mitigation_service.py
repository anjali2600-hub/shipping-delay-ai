"""
mitigation_service.py
Rule-based engine that identifies likely delay factors and suggests
mitigation actions, given the shipment input and the ML prediction
result. Architecture is intentionally simple/rule-based per spec,
but kept isolated in its own module so a smarter (e.g. model-driven)
version could replace it later without touching the rest of the app.
"""

def identify_delay_factors(raw_input: dict, prediction: dict) -> list:
    """
    Returns a list of human-readable strings describing the main
    contributors to the predicted delay, ranked roughly by severity.
    """
    factors = []

    traffic_level = raw_input.get("traffic_level", "Low")
    weather_condition = raw_input.get("weather_condition", "Clear")
    distance_km = float(raw_input.get("distance_km", 0))
    route_type = raw_input.get("route_type", "Highway")
    historical_delay_rate = float(raw_input.get("historical_delay_rate", 0))

    if traffic_level in ("High", "Severe"):
        factors.append(f"{traffic_level} traffic conditions")

    if weather_condition in ("Rain", "Heavy Rain", "Storm", "Fog"):
        factors.append(f"{weather_condition} weather")

    if distance_km > 1000:
        factors.append("Long delivery distance")
    elif distance_km > 500:
        factors.append("Moderate delivery distance")

    if route_type in ("City Road", "Rural Road"):
        factors.append(f"{route_type} routing (slower than highway)")

    if historical_delay_rate > 0.3:
        factors.append("Route has a history of frequent delays")

    if not factors:
        factors.append("No major risk factors identified")

    return factors


def get_recommendation(raw_input: dict, prediction: dict) -> str:
    """
    Returns a single, actionable recommendation string based on the
    combination of traffic, weather, and predicted risk level.
    """
    traffic_level = raw_input.get("traffic_level", "Low")
    weather_condition = raw_input.get("weather_condition", "Clear")
    risk_level = prediction.get("risk_level", "LOW")

    high_traffic = traffic_level in ("High", "Severe")
    bad_weather = weather_condition in ("Rain", "Heavy Rain", "Storm", "Fog")

    # Rule priority: worst combinations first
    if high_traffic and bad_weather:
        return (
            "High risk of significant delay. Consider an alternate route, "
            "dispatch earlier than usual, and proactively notify the customer "
            "of a revised delivery window."
        )

    if high_traffic and not bad_weather:
        return (
            "Traffic is the main risk factor. Consider an alternate route "
            "or an earlier dispatch time to avoid peak congestion."
        )

    if bad_weather and not high_traffic:
        return (
            "Weather is the main risk factor. Increase the estimated delivery "
            "time buffer and monitor conditions in case of further deterioration."
        )

    if risk_level == "MEDIUM":
        return (
            "Some risk of delay present. Monitor traffic and weather updates "
            "closer to dispatch time."
        )

    if risk_level == "HIGH":
        return (
            "High risk of delay. Recommend proactive customer communication "
            "and evaluating alternate routing options."
        )

    return "Conditions look favorable. Standard dispatch procedure is sufficient."


def get_mitigation_summary(raw_input: dict, prediction: dict) -> dict:
    """
    Combined entry point: returns both the delay factors and the
    recommendation together, ready to merge into the final API
    response shown to the user.
    """
    return {
        "main_factors": identify_delay_factors(raw_input, prediction),
        "recommendation": get_recommendation(raw_input, prediction),
    }


if __name__ == "__main__":
    sample_input = {
        "traffic_level": "High",
        "weather_condition": "Heavy Rain",
        "distance_km": 1400,
        "route_type": "City Road",
        "historical_delay_rate": 0.35,
    }
    sample_prediction = {"risk_level": "HIGH"}

    result = get_mitigation_summary(sample_input, sample_prediction)
    print("Mitigation summary:")
    print(f"  Main factors: {result['main_factors']}")
    print(f"  Recommendation: {result['recommendation']}")