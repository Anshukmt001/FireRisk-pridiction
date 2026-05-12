# 🔥 ForestGuard — Island Fire Risk Intelligence

A full-stack Machine Learning web application that predicts **forest fire risk levels** (Low / Medium / High) for islands worldwide using real-time weather data.

---

## 📁 Project Structure

```
forest_fire_project/
├── app.py                  # Flask backend (API server)
├── train_model.py          # Model training script (run once)
├── model.pkl               # Saved trained model (generated)
├── requirements.txt
├── templates/
│   └── index.html          # Dashboard UI
├── static/
│   └── style.css           # Styles
└── utils/
    ├── __init__.py
    └── data_fetch.py       # Weather API + feature engineering
```

---

## 🚀 Quick Start

### 1. Clone / copy the project
```bash
cd forest_fire_project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Set your OpenWeatherMap API key
```bash
export OPENWEATHER_API_KEY="your_key_here"   # Linux / macOS
set OPENWEATHER_API_KEY=your_key_here         # Windows CMD
```
Get a free key at https://openweathermap.org/api  
Without a key the app uses **simulated weather data** — everything still works perfectly.

### 5. Train the model (one-time step)
```bash
python train_model.py
```
This generates `model.pkl` and prints accuracy / feature importance.

### 6. Start the Flask server
```bash
python app.py
```

### 7. Open the dashboard
Visit **http://localhost:5000** in your browser.

---

## 🎯 How It Works

| Step | Description |
|------|-------------|
| 1 | Click an island preset (or enter lat/lon on the map) |
| 2 | Backend fetches live weather from OpenWeatherMap |
| 3 | Additional features (drought index, vegetation, slope) are estimated |
| 4 | Random Forest model predicts fire risk class + probabilities |
| 5 | Dashboard updates with colour-coded result and live stats |

---

## 🌡 ML Features

| Feature | Source |
|---------|--------|
| Temperature (°C) | OpenWeatherMap API |
| Humidity (%) | OpenWeatherMap API |
| Wind Speed (km/h) | OpenWeatherMap API |
| Drought Index (KBDI proxy) | Simulated / extendable |
| Vegetation density (NDVI proxy) | Simulated / extendable |
| Terrain slope (°) | Simulated / extendable |

---

## 🔌 API Reference

### `POST /predict`

**Option A — fetch live weather:**
```json
{ "lat": 28.29, "lon": -16.62 }
```

**Option B — manual feature input:**
```json
{
  "temperature": 40,
  "humidity": 15,
  "wind_speed": 55,
  "drought_index": 700,
  "vegetation": 0.6,
  "slope": 25
}
```

**Response:**
```json
{
  "risk_level": "High",
  "risk_class": 2,
  "risk_color": "#ef4444",
  "risk_emoji": "🔴",
  "alert": true,
  "probabilities": { "Low": 3.2, "Medium": 12.5, "High": 84.3 },
  "features": { ... },
  "source": "live",
  "city": "Tenerife"
}
```

### `GET /islands`  — list of island presets  
### `GET /health`   — server health check

---

## 🧠 Model Details

- **Algorithm**: Random Forest (200 estimators, max_depth=12)
- **Training set**: 2 400 synthetic samples (stratified)
- **Test set**: 600 samples
- **Typical accuracy**: ~95 %
- **Labels**: 0 = Low, 1 = Medium, 2 = High

---

## 🗺 Island Presets

Canary Islands · Hawaii · Sardinia · Sumatra · Tasmania · Corsica · Borneo · Crete · Vancouver Island · Maui

---

## 📦 Production Deployment

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

---

## 📝 License

MIT — free to use, modify, and distribute.