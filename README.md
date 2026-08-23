
## Setup & Installation (Windows)

### 1. Clone the repository

```powershell
git clone https://github.com/anjali2600-hub/shipping-delay-ai.git
cd shipping-delay-ai
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Set up MySQL

Create a database:

```sql
CREATE DATABASE shipping_db;
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_MAPS_API_KEY=
OPENWEATHER_API_KEY=
DATABASE_URL=mysql+pymysql://root:your_password@localhost:3306/shipping_db
```

Leaving the API keys blank is fine — the app automatically falls back to realistic mock data for traffic and weather.

### 6. (Optional) Retrain models from scratch

Trained models are already included in `models/`. To regenerate them:

```powershell
python src\generate_data.py
python src\train.py
python src\evaluate.py
```

### 7. Run the application

```powershell
python app.py
```

Visit **http://127.0.0.1:5000** in your browser.

## Usage

1. Enter shipment details (origin, destination, and optionally distance/weather/traffic — left blank, these are auto-fetched)
2. Click **Predict Delivery**
3. View predicted delivery time, delay category, risk level, contributing factors, and a recommended action
4. Visit **History** to see all past predictions

## Model Performance (Example Results)

| Model | RMSE (minutes) | R² |
|---|---|---|
| XGBoost | 19.0 | 0.83 |
| Linear Regression | 19.1 | 0.83 |
| Random Forest | 19.1 | 0.83 |

| Model | Accuracy | F1-score |
|---|---|---|
| Random Forest | 0.69 | 0.69 |
| XGBoost | 0.69 | 0.68 |
| Logistic Regression | 0.56 | 0.56 |

## Future Improvements

- Replace rule-based mitigation engine with a learned recommendation model
- Add real-time route re-optimization
- Expand dataset with real-world logistics data if/when available

## Author

Anjali Agrawal