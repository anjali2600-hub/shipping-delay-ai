"""
test_db.py
One-off script to verify the database connection, create tables,
and insert/read a test row. Not part of the final app — just for
confirming Step 12 works before moving on.
"""

from database.db import init_db, SessionLocal
from database.models import Shipment, shipment_to_dict

print("Creating tables (if they don't exist)...")
init_db()
print("Tables created successfully.")

db = SessionLocal()

test_shipment = Shipment(
    origin="Delhi",
    destination="Mumbai",
    distance_km=1400,
    weather_condition="Heavy Rain",
    temperature=28,
    rainfall=35,
    traffic_level="High",
    route_type="City Road",
    package_weight=12,
    shipment_type="Express",
    historical_delay_rate=0.3,
    predicted_delivery_time_minutes=2003.8,
    predicted_delay_minutes=137.1,
    delay_category="Moderate Delay",
    delay_probability_percent=72.4,
    risk_level="MEDIUM",
    will_be_delayed=True,
    main_factors="High traffic|Heavy Rain weather|Long distance",
    recommendation="Consider an alternate route and earlier dispatch.",
)

db.add(test_shipment)
db.commit()
db.refresh(test_shipment)

print(f"\nInserted test row with id={test_shipment.id}")

all_shipments = db.query(Shipment).all()
print(f"\nTotal rows in shipments table: {len(all_shipments)}")
print("Latest row as dict:")
print(shipment_to_dict(all_shipments[-1]))

db.close()