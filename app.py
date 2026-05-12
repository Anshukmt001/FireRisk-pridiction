"""
app.py  –  Forest Fire Risk Detection – Flask Backend
Run:  python app.py
"""

import os
import pickle
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template

# Local utilities
import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.data_fetch import fetch_all_features

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ─── Load model bundle ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        logger.error("model.pkl not found. Run:  python train_model.py")
        return None, None, None
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    logger.info("Model loaded successfully.")
    return bundle["model"], bundle["scaler"], bundle["features"]

MODEL, SCALER, FEATURES = load_model()

# ─── Risk metadata ────────────────────────────────────────────────────────────
RISK_META = {
    0: {"label": "Low",    "color": "#22c55e", "emoji": "🟢", "alert": False},
    1: {"label": "Medium", "color": "#f59e0b", "emoji": "🟡", "alert": False},
    2: {"label": "High",   "color": "#ef4444", "emoji": "🔴", "alert": True },
}

# ─── Island presets ───────────────────────────────────────────────────────────
ISLANDS = {
    "canary":    {"name": "Canary Islands, Spain",  "lat":  28.29, "lon": -16.62},
    "hawaii":    {"name": "Hawaii, USA",             "lat":  19.90, "lon":-155.58},
    "sardinia":  {"name": "Sardinia, Italy",         "lat":  40.12, "lon":   9.01},
    "sumatra":   {"name": "Sumatra, Indonesia",      "lat":  -0.59, "lon": 101.34},
    "tasmania":  {"name": "Tasmania, Australia",     "lat": -42.00, "lon": 146.00},
    "corsica":   {"name": "Corsica, France",         "lat":  42.05, "lon":   9.09},
    "borneo":    {"name": "Borneo",                  "lat":   1.82, "lon": 113.92},
    "crete":     {"name": "Crete, Greece",           "lat":  35.24, "lon":  24.81},
    "vancouver": {"name": "Vancouver Island, Canada","lat":  49.65, "lon":-125.45},
    "maui":      {"name": "Maui, Hawaii",            "lat":  20.80, "lon":-156.33},
}


def _predict(features: dict) -> dict:
    """Run inference and return enriched result dict."""
    if MODEL is None:
        return {"error": "Model not loaded. Run: python train_model.py"}

    row = np.array([[features[f] for f in FEATURES]])
    row_sc = SCALER.transform(row)

    pred_class = int(MODEL.predict(row_sc)[0])
    proba      = MODEL.predict_proba(row_sc)[0].tolist()

    meta = RISK_META[pred_class]
    return {
        "risk_level":   meta["label"],
        "risk_class":   pred_class,
        "risk_color":   meta["color"],
        "risk_emoji":   meta["emoji"],
        "alert":        meta["alert"],
        "probabilities": {
            "Low":    round(proba[0] * 100, 1),
            "Medium": round(proba[1] * 100, 1),
            "High":   round(proba[2] * 100, 1),
        },
        "features": {k: round(float(v), 2) for k, v in features.items()
                     if k in FEATURES},
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", islands=ISLANDS)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON with either:
      (a) lat + lon  → fetch live weather then predict
      (b) full feature dict → predict directly (manual override)
    """
    data = request.get_json(force=True, silent=True) or {}

    # Manual feature input
    if all(k in data for k in FEATURES):
        features = {k: float(data[k]) for k in FEATURES}
        result   = _predict(features)
        result["source"] = "manual"
        return jsonify(result)

    # Live weather fetch
    lat = data.get("lat")
    lon = data.get("lon")
    if lat is None or lon is None:
        return jsonify({"error": "Provide 'lat' & 'lon' or all feature values."}), 400

    try:
        features = fetch_all_features(float(lat), float(lon))
    except Exception as exc:
        logger.exception("Feature fetch failed")
        return jsonify({"error": str(exc)}), 500

    result = _predict(features)
    result["source"]      = features.get("source", "unknown")
    result["city"]        = features.get("city", "")
    result["description"] = features.get("description", "")
    result["icon"]        = features.get("icon", "01d")
    result["lat"]         = lat
    result["lon"]         = lon
    return jsonify(result)


@app.route("/islands")
def islands():
    return jsonify(ISLANDS)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)