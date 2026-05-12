import pickle
import os

MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        print("Model bundle loaded successfully.")
        print("Features:", bundle["features"])
    except Exception as e:
        print("Error loading model:", e)
else:
    print("model.pkl not found.")
