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
from utils.data_fetch import fetch_all_features, reverse_geocode

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

DEEP_MODEL_PATH = os.path.join(os.path.dirname(__file__), "deep_model.pkl")

import torch
import torch.nn as nn

class FireRiskNN(nn.Module):
    def __init__(self, input_size):
        super(FireRiskNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def load_deep_model():
    if not os.path.exists(DEEP_MODEL_PATH):
        logger.warning("deep_model.pkl not found. Run: python train_deep_model.py")
        return None, None, None, None
    with open(DEEP_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    logger.info("Deep learning model loaded successfully.")
    return bundle["model"], bundle["scaler"], bundle["features"], bundle.get("model_type", "deep_learning")

DEEP_MODEL, DEEP_SCALER, DEEP_FEATURES, DEEP_MODEL_TYPE = load_deep_model()

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


def _predict(features: dict, model_type: str = "random_forest") -> dict:
    """Run inference using ML or Deep Learning model."""
    if model_type == "deep_learning":
        if DEEP_MODEL is None:
            return {"error": "Deep model not loaded. Run: python train_deep_model.py"}
        row = np.array([[features[f] for f in DEEP_FEATURES]])
        row_sc = DEEP_SCALER.transform(row)

        if DEEP_MODEL_TYPE == "deep_learning_pytorch" or not hasattr(DEEP_MODEL, 'predict'):
            import torch
            DEEP_MODEL.eval()
            with torch.no_grad():
                tensor_input = torch.FloatTensor(row_sc)
                output = DEEP_MODEL(tensor_input)
                proba = torch.softmax(output, dim=1).numpy()[0].tolist()
                pred_class = int(np.argmax(proba))
        else:
            proba = DEEP_MODEL.predict(row_sc)[0].tolist()
            pred_class = int(np.argmax(proba))

        meta = RISK_META[pred_class]
        return {
            "model_type":   "deep_learning",
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
                         if k in DEEP_FEATURES},
        }
    else:
        if MODEL is None:
            return {"error": "Model not loaded. Run: python train_model.py"}

        row = np.array([[features[f] for f in FEATURES]])
        row_sc = SCALER.transform(row)

        pred_class = int(MODEL.predict(row_sc)[0])
        proba      = MODEL.predict_proba(row_sc)[0].tolist()

        meta = RISK_META[pred_class]
        return {
            "model_type":   "random_forest",
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

    Optional: "model_type" can be "random_forest" or "deep_learning"
    """
    data = request.get_json(force=True, silent=True) or {}
    model_type = data.get("model_type", "random_forest")

    all_features = ["temperature", "humidity", "wind_speed", "drought_index", "vegetation", "slope"]

    # Manual feature input
    if all(k in data for k in all_features):
        features = {k: float(data[k]) for k in all_features}
        result   = _predict(features, model_type)
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

    result = _predict(features, model_type)
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


@app.route("/geocode", methods=["GET"])
def geocode():
    """Reverse geocode lat/lon to location name."""
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    if lat is None or lon is None:
        return jsonify({"error": "Provide lat and lon parameters"}), 400
    result = reverse_geocode(lat, lon)
    return jsonify(result)


@app.route("/models")
def models():
    return jsonify({
        "available_models": ["random_forest", "deep_learning"],
        "random_forest": {"loaded": MODEL is not None, "type": "Random Forest Classifier"},
        "deep_learning": {"loaded": DEEP_MODEL is not None, "type": DEEP_MODEL_TYPE if DEEP_MODEL else "Not available"}
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "random_forest_loaded": MODEL is not None,
        "deep_learning_loaded": DEEP_MODEL is not None,
        "deep_model_type": DEEP_MODEL_TYPE if DEEP_MODEL else None
    })


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)