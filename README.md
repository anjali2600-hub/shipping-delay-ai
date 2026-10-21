# AI-Based Delivery Delay Classification and Mitigation System

A machine learning web app that predicts how long a last-mile delivery will take, classifies how late it is likely to be, and suggests what a dispatcher can do about it. It is trained on **real delivery data** (Amazon Delivery Dataset, Kaggle).

## What it does

For one delivery it returns:

- **Predicted delivery time** (minutes)
- **Delay category**: On Time / Slight Delay / Moderate Delay / Severe Delay
- **Delay probability** (chance of being more than 15 minutes late) and **risk level** (LOW / MEDIUM / HIGH)
- **Main delay factors** (traffic, weather, peak hour, distance, agent rating)
- **Recommendation**: actions to take and a revised delivery window for the customer

Every prediction is saved in MySQL and shown on a **History** page.

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python |
| Machine learning | scikit-learn (Linear/Logistic Regression, Random Forest), XGBoost |
| APIs | Google Maps (distance, traffic), OpenWeather (weather), with automatic mock fallback |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Database | MySQL via SQLAlchemy |

## Dataset

**Amazon Delivery Dataset** (Kaggle, author: sujalsuthar): 43,739 deliveries with weather, traffic, vehicle, area, item category, agent age and rating, store and drop coordinates, order and pickup time, and the delivery time in minutes.

Cleaning (`src/clean_amazon.py`) leaves **39,997 rows**:

- removed spaces and a spelling error in text values, and the text "NaN" in Traffic
- dropped rows with missing weather or traffic
- dropped rows whose coordinates fall outside India (3,651 rows)
- computed the straight-line (haversine) distance between store and drop point, which comes out to 1.5 to 21 km, so this is **city-level last-mile delivery**
- added time features (order hour, day of week, weekend, peak hour)

### How the delay label is built

The dataset has no promised delivery time, so a delay cannot be read directly. It is derived (`src/make_labels.py`):

1. Take deliveries made in the best conditions (Low traffic and Sunny weather, 2,201 rows).
2. Fit expected time = 76.8 + 3.25 x distance_km.
3. delay = actual delivery time - expected time.

| Category | Delay vs expected | Share of data |
|---|---|---|
| On Time | up to 15 min | 49.2% |
| Slight Delay | 15 to 45 min | 22.3% |
| Moderate Delay | 45 to 90 min | 21.8% |
| Severe Delay | over 90 min | 6.7% |

## Models and results

Data is split 80/20 (stratified on the delay category).

**Delivery time (regression), 8,000 test deliveries**

| Model | MAE (min) | RMSE (min) | R2 |
|---|---|---|---|
| Random Forest (best) | 17.1 | 22.1 | 0.82 |
| XGBoost | 17.7 | 22.8 | 0.81 |
| Linear Regression | 25.3 | 32.2 | 0.62 |

**Delay category (classification), class-weighted**

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| XGBoost (best) | 0.67 | 0.68 | 0.60 |
| Random Forest | 0.66 | 0.67 | 0.59 |
| Logistic Regression | 0.55 | 0.57 | 0.48 |

Most mistakes are between neighbouring categories, and On Time deliveries are rarely mistaken for Severe ones. Severe Delay has high recall (0.80) but lower precision (0.37) because the model is weighted to catch rare, serious delays.

## Limitations

- The delay label is derived from an estimate of expected time, not a real promised time. The delivery-time regression is the most reliable result; the category classifier partly learns the labelling rule.
- Weather values in the dataset look evenly spread, and Cloudy shows slower deliveries than Stormy, so the weather effect should not be read as real-world meteorology.
- The data covers a short period in 2022 and city deliveries up to about 21 km. The app rejects distances above 25 km.
- Rain, Heavy Rain and Storm from OpenWeather are mapped to "Stormy", the closest label in the dataset.
- The risk level uses thresholds chosen by the author on the model's estimated chance of a delay of 45 minutes or more.

## Project structure

```
shipping-delay-ai/
├── app.py                        Flask app
├── requirements.txt
├── src/
│   ├── explore_amazon.py         first look at the raw data
│   ├── clean_amazon.py           cleaning and features
│   ├── make_labels.py            builds the delay label
│   └── train_amazon.py           trains and compares models
├── services/
│   ├── prediction_service.py     loads models, makes a prediction
│   ├── mitigation_service.py     rule-based factors and recommendations
│   ├── maps_service.py           Google Maps with mock fallback
│   └── weather_service.py        OpenWeather with mock fallback
├── database/                     SQLAlchemy engine and table
├── models/amazon/                trained models
├── templates/ and static/        web pages
```

The first, synthetic-data version of this project (generated dataset, `generate_data.py`, `train.py`, `evaluate.py`) is kept on the `synthetic-version` branch.

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/anjali2600-hub/shipping-delay-ai.git
cd shipping-delay-ai
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a MySQL database named `shipping_db`, then create a `.env` file:

```env
GOOGLE_MAPS_API_KEY=
OPENWEATHER_API_KEY=
DATABASE_URL=mysql+pymysql://root:your_password@localhost:3306/shipping_db
```

The API keys can stay empty; the app then uses mock data.

Run the app (trained models are included):

```powershell
python app.py
```

Open http://127.0.0.1:5000.

To rebuild everything from the raw data, download `amazon_delivery.csv` from Kaggle into `data/raw/`, then run:

```powershell
python src\clean_amazon.py
python src\make_labels.py
python src\train_amazon.py
```

## Author

Anjali Agrawal