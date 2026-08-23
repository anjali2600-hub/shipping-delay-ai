"""
models.py
SQLAlchemy ORM table definitions for the shipping-delay-ai app.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.sql import func

from database.db import Base


class Shipment(Base):
    """
    Stores every prediction request: the raw input the user submitted,
    plus the full prediction result. Doubles as prediction history.
    """
    __tablename__ = "shipments"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # --- Input fields (what the user submitted) ---
    origin = Column(String(100), nullable=False)
    destination = Column(String(100), nullable=False)
    distance_km = Column(Float, nullable=False)
    weather_condition = Column(String(50), nullable=False)
    temperature = Column(Float)
    rainfall = Column(Float)
    traffic_level = Column(String(50), nullable=False)
    route_type = Column(String(50), nullable=False)
    package_weight = Column(Float)
    shipment_type = Column(String(50))
    historical_delay_rate = Column(Float)

    # --- Prediction output fields ---
    predicted_delivery_time_minutes = Column(Float)
    predicted_delay_minutes = Column(Float)
    delay_category = Column(String(50))
    delay_probability_percent = Column(Float)
    risk_level = Column(String(20))
    will_be_delayed = Column(Boolean)

    # --- Mitigation output ---
    main_factors = Column(String(500))   # stored as comma-joined text
    recommendation = Column(String(500))

    # --- Metadata ---
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def shipment_to_dict(shipment: "Shipment") -> dict:
    """Converts a Shipment ORM row into a plain dict for templates/JSON responses."""
    return {
        "id": shipment.id,
        "origin": shipment.origin,
        "destination": shipment.destination,
        "distance_km": shipment.distance_km,
        "weather_condition": shipment.weather_condition,
        "traffic_level": shipment.traffic_level,
        "route_type": shipment.route_type,
        "shipment_type": shipment.shipment_type,
        "predicted_delivery_time_minutes": shipment.predicted_delivery_time_minutes,
        "predicted_delay_minutes": shipment.predicted_delay_minutes,
        "delay_category": shipment.delay_category,
        "delay_probability_percent": shipment.delay_probability_percent,
        "risk_level": shipment.risk_level,
        "will_be_delayed": shipment.will_be_delayed,
        "main_factors": shipment.main_factors.split("|") if shipment.main_factors else [],
        "recommendation": shipment.recommendation,
        "created_at": shipment.created_at.strftime("%Y-%m-%d %H:%M:%S") if shipment.created_at else None,
    }