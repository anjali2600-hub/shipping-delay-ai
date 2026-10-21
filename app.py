"""
app.py  (real-data version, Amazon Delivery Dataset)
Flask backend: form -> maps/weather auto-fill -> ML prediction ->
mitigation -> database -> result page.
"""

from datetime import datetime

from flask import Flask, render_template, request

from database.db import SessionLocal, init_db
from database.models import Delivery, delivery_to_dict
from services.maps_service import get_route_data
from services.mitigation_service import get_mitigation_summary
from services.prediction_service import OPTIONS, predict_shipment
from services.weather_service import get_weather_data

app = Flask(__name__)

# Create the tables on start-up if they do not exist yet
init_db()

# The model was trained on city deliveries of up to about 21 km
MAX_DISTANCE_KM = 25

# The maps / weather services use their own labels, so translate them
# into the labels used in the training data.
WEATHER_MAP = {
    "Clear": "Sunny", "Cloudy": "Cloudy", "Fog": "Fog",
    "Rain": "Stormy", "Heavy Rain": "Stormy", "Storm": "Stormy",
}
TRAFFIC_MAP = {"Low": "Low", "Medium": "Medium", "High": "High", "Severe": "Jam"}


def render_form(error=None, form=None):
    """Shows the input form (with an error message if there is one)."""
    return render_template(
        "index.html",
        options=OPTIONS,
        error=error,
        form=form or {},
        now=datetime.now().strftime("%Y-%m-%dT%H:%M"),
    )


@app.route("/", methods=["GET"])
def index():
    return render_form()
@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    store = form.get("store_location", "").strip()
    drop = form.get("drop_location", "").strip()

    # Distance and traffic: use what the user typed, otherwise ask the maps service
    distance_text = form.get("distance_km", "").strip()
    traffic = form.get("traffic_level", "").strip()
    if not distance_text or not traffic:
        if not (store and drop):
            return render_form(
                "Enter the distance and traffic, or fill in both the store "
                "and drop locations so they can be fetched.", form), 400
        route = get_route_data(store, drop)
        distance_text = distance_text or str(route["distance_km"])
        traffic = traffic or TRAFFIC_MAP.get(route["traffic_level"], "Medium")

    # Weather: use what the user chose, otherwise ask the weather service
    weather = form.get("weather_condition", "").strip()
    if not weather:
        place = drop or store
        if not place:
            return render_form(
                "Choose the weather, or enter a location so it can be fetched.",
                form), 400
        weather = WEATHER_MAP.get(get_weather_data(place)["condition"], "Sunny")

    # Numbers and dates
    try:
        distance_km = float(distance_text)
        agent_age = int(form.get("agent_age", "").strip() or 30)
        agent_rating = float(form.get("agent_rating", "").strip() or 4.7)
    except ValueError:
        return render_form(
            "Distance, agent age and agent rating must be numbers.", form), 400

    if not 0.5 <= distance_km <= MAX_DISTANCE_KM:
        return render_form(
            f"Distance must be between 0.5 and {MAX_DISTANCE_KM} km. The model "
            "is trained on city deliveries of up to about 21 km.", form), 400
    if not 1 <= agent_rating <= 5:
        return render_form("Agent rating must be between 1 and 5.", form), 400
    if not 15 <= agent_age <= 60:
        return render_form("Agent age must be between 15 and 60.", form), 400

    order_time_text = form.get("order_time", "").strip()
    try:
        order_dt = (datetime.fromisoformat(order_time_text)
                    if order_time_text else datetime.now())
    except ValueError:
        return render_form("The order time is not valid.", form), 400

    raw_input = {
        "distance_km": distance_km,
        "weather_condition": weather,
        "traffic_level": traffic,
        "vehicle": form.get("vehicle", "motorcycle"),
        "area": form.get("area", "Metropolitan"),
        "category": form.get("category", "Electronics"),
        "agent_age": agent_age,
        "agent_rating": agent_rating,
        "order_time": order_dt.strftime("%Y-%m-%d %H:%M:%S"),
    }

    prediction = predict_shipment(raw_input)
    mitigation = get_mitigation_summary(raw_input, prediction)

    # Save to the database
    db = SessionLocal()
    try:
        db.add(Delivery(
            store_location=store or None,
            drop_location=drop or None,
            distance_km=distance_km,
            weather_condition=weather,
            traffic_level=traffic,
            vehicle=raw_input["vehicle"],
            area=raw_input["area"],
            category=raw_input["category"],
            agent_age=agent_age,
            agent_rating=agent_rating,
            order_time=order_dt,
            predicted_delivery_time_minutes=prediction["predicted_delivery_time_minutes"],
            expected_time_minutes=prediction["expected_time_minutes"],
            predicted_delay_minutes=prediction["predicted_delay_minutes"],
            delay_category=prediction["delay_category"],
            delay_probability_percent=prediction["delay_probability_percent"],
            serious_delay_probability_percent=prediction["serious_delay_probability_percent"],
            risk_level=prediction["risk_level"],
            will_be_delayed=prediction["will_be_delayed"],
            main_factors="|".join(mitigation["main_factors"]),
            recommendation=mitigation["recommendation"],
        ))
        db.commit()
    finally:
        db.close()

    return render_template(
        "result.html",
        raw_input=raw_input,
        prediction=prediction,
        mitigation=mitigation,
        store=store,
        drop=drop,
    )


@app.route("/history", methods=["GET"])
def history():
    db = SessionLocal()
    try:
        rows = (db.query(Delivery)
                  .order_by(Delivery.created_at.desc())
                  .limit(50)
                  .all())
        deliveries = [delivery_to_dict(r) for r in rows]
    finally:
        db.close()
    return render_template("history.html", deliveries=deliveries)


if __name__ == "__main__":
    app.run(debug=True)