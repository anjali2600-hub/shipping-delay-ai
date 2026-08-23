"""
app.py
Flask backend for the AI-Based Shipping Delay Classification and
Mitigation System. Wires together the ML prediction service, maps/
weather services, mitigation engine, and database.
"""

from flask import Flask, render_template, request, redirect, url_for

from services.prediction_service import predict_shipment
from services.maps_service import get_route_data
from services.weather_service import get_weather_data
from services.mitigation_service import get_mitigation_summary
from database.db import init_db, SessionLocal
from database.models import Shipment, shipment_to_dict

app = Flask(__name__)

# Create tables on startup if they don't already exist
init_db()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    # Optionally auto-fill distance/traffic from Maps API (mock or live)
    # if the user didn't manually enter a distance.
    distance_km = form.get("distance_km", "").strip()
    traffic_level = form.get("traffic_level", "").strip()

    if not distance_km or not traffic_level:
        route_info = get_route_data(form.get("origin"), form.get("destination"))
        distance_km = distance_km or route_info["distance_km"]
        traffic_level = traffic_level or route_info["traffic_level"]

    # Optionally auto-fill weather from OpenWeather (mock or live)
    weather_condition = form.get("weather_condition", "").strip()
    temperature = form.get("temperature", "").strip()
    rainfall = form.get("rainfall", "").strip()

    if not weather_condition:
        weather_info = get_weather_data(form.get("destination"))
        weather_condition = weather_info["condition"]
        temperature = temperature or weather_info["temperature"]
        rainfall = rainfall or weather_info["rainfall_mm"]

    raw_input = {
        "origin": form.get("origin"),
        "destination": form.get("destination"),
        "distance_km": float(distance_km),
        "weather_condition": weather_condition,
        "temperature": float(temperature) if temperature else 25.0,
        "rainfall": float(rainfall) if rainfall else 0.0,
        "traffic_level": traffic_level,
        "route_type": form.get("route_type", "Highway"),
        "package_weight": float(form.get("package_weight", 5.0)),
        "shipment_type": form.get("shipment_type", "Standard"),
        "historical_delay_rate": float(form.get("historical_delay_rate", 0.2)),
    }

    prediction = predict_shipment(raw_input)
    mitigation = get_mitigation_summary(raw_input, prediction)

    # Save to database
    db = SessionLocal()
    shipment_record = Shipment(
        origin=raw_input["origin"],
        destination=raw_input["destination"],
        distance_km=raw_input["distance_km"],
        weather_condition=raw_input["weather_condition"],
        temperature=raw_input["temperature"],
        rainfall=raw_input["rainfall"],
        traffic_level=raw_input["traffic_level"],
        route_type=raw_input["route_type"],
        package_weight=raw_input["package_weight"],
        shipment_type=raw_input["shipment_type"],
        historical_delay_rate=raw_input["historical_delay_rate"],
        predicted_delivery_time_minutes=prediction["predicted_delivery_time_minutes"],
        predicted_delay_minutes=prediction["predicted_delay_minutes"],
        delay_category=prediction["delay_category"],
        delay_probability_percent=prediction["delay_probability_percent"],
        risk_level=prediction["risk_level"],
        will_be_delayed=prediction["will_be_delayed"],
        main_factors="|".join(mitigation["main_factors"]),
        recommendation=mitigation["recommendation"],
    )
    db.add(shipment_record)
    db.commit()
    db.close()

    return render_template(
        "result.html",
        raw_input=raw_input,
        prediction=prediction,
        mitigation=mitigation,
    )


@app.route("/history", methods=["GET"])
def history():
    db = SessionLocal()
    records = db.query(Shipment).order_by(Shipment.created_at.desc()).limit(50).all()
    db.close()

    shipments = [shipment_to_dict(r) for r in records]
    return render_template("history.html", shipments=shipments)


if __name__ == "__main__":
    app.run(debug=True)