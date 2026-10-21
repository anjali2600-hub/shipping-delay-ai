"""
models.py  (real-data version)
SQLAlchemy table for the Amazon-data version of the app.
Uses a NEW table called "deliveries", so the old synthetic-data table
("shipments") is left untouched in MySQL.
"""

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from database.db import Base


class Delivery(Base):
    """One prediction request: the input the user gave plus the full result."""
    __tablename__ = "deliveries"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # --- Input ---
    store_location = Column(String(200))
    drop_location = Column(String(200))
    distance_km = Column(Float, nullable=False)
    weather_condition = Column(String(30), nullable=False)
    traffic_level = Column(String(30), nullable=False)
    vehicle = Column(String(30))
    area = Column(String(30))
    category = Column(String(50))
    agent_age = Column(Integer)
    agent_rating = Column(Float)
    order_time = Column(DateTime)

    # --- Prediction ---
    predicted_delivery_time_minutes = Column(Float)
    expected_time_minutes = Column(Float)
    predicted_delay_minutes = Column(Float)
    delay_category = Column(String(30))
    delay_probability_percent = Column(Float)
    serious_delay_probability_percent = Column(Float)
    risk_level = Column(String(10))
    will_be_delayed = Column(Boolean)

    # --- Mitigation ---
    main_factors = Column(String(500))   # factors joined with "|"
    recommendation = Column(String(500))

    created_at = Column(DateTime, server_default=func.now())


def delivery_to_dict(d: "Delivery") -> dict:
    """Turns a Delivery row into a plain dict for the templates."""
    return {
        "id": d.id,
        "store_location": d.store_location,
        "drop_location": d.drop_location,
        "distance_km": d.distance_km,
        "weather_condition": d.weather_condition,
        "traffic_level": d.traffic_level,
        "vehicle": d.vehicle,
        "area": d.area,
        "category": d.category,
        "agent_age": d.agent_age,
        "agent_rating": d.agent_rating,
        "order_time": d.order_time.strftime("%Y-%m-%d %H:%M") if d.order_time else None,
        "predicted_delivery_time_minutes": d.predicted_delivery_time_minutes,
        "expected_time_minutes": d.expected_time_minutes,
        "predicted_delay_minutes": d.predicted_delay_minutes,
        "delay_category": d.delay_category,
        "delay_probability_percent": d.delay_probability_percent,
        "serious_delay_probability_percent": d.serious_delay_probability_percent,
        "risk_level": d.risk_level,
        "will_be_delayed": d.will_be_delayed,
        "main_factors": d.main_factors.split("|") if d.main_factors else [],
        "recommendation": d.recommendation,
        "created_at": d.created_at.strftime("%Y-%m-%d %H:%M:%S") if d.created_at else None,
    }