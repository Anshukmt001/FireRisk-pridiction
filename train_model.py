"""
train_model.py
Trains a Random Forest classifier for forest fire risk detection and saves it as model.pkl
Run this once before starting the Flask app: python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# ─── 1. Generate synthetic dataset ────────────────────────────────────────────
np.random.seed(42)
N = 3000

def generate_dataset(n):
    """
    Features:
        temperature   : °C  (15 – 50)
        humidity      : %   (5  – 95)
        wind_speed    : km/h(0  – 80)
        drought_index : 0 – 800 (Keetch-Byram Drought Index proxy)
        vegetation    : 0 – 1  (normalised vegetation density)
        slope         : degrees (0 – 45)
    Labels:
        0 = Low, 1 = Medium, 2 = High
    """
    temperature   = np.random.uniform(15, 50, n)
    humidity      = np.random.uniform(5,  95, n)
    wind_speed    = np.random.uniform(0,  80, n)
    drought_index = np.random.uniform(0, 800, n)
    vegetation    = np.random.uniform(0,   1, n)
    slope         = np.random.uniform(0,  45, n)

    # Rule-based labels (deterministic + some noise)
    risk_score = (
          0.35 * (temperature   / 50)
        + 0.25 * (1 - humidity  / 95)
        + 0.15 * (wind_speed    / 80)
        + 0.15 * (drought_index / 800)
        + 0.05 * vegetation
        + 0.05 * (slope         / 45)
    )

    # Add small noise
    risk_score += np.random.normal(0, 0.03, n)
    risk_score = np.clip(risk_score, 0, 1)

    labels = np.where(risk_score < 0.35, 0,
             np.where(risk_score < 0.65, 1, 2))

    df = pd.DataFrame({
        "temperature":   temperature,
        "humidity":      humidity,
        "wind_speed":    wind_speed,
        "drought_index": drought_index,
        "vegetation":    vegetation,
        "slope":         slope,
        "risk":          labels,
    })
    return df

df = generate_dataset(N)
print(f"Dataset shape : {df.shape}")
print(f"Class balance :\n{df['risk'].value_counts().sort_index()}\n")

# ─── 2. Train / test split ────────────────────────────────────────────────────
X = df.drop("risk", axis=1)
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── 3. Scale features ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── 4. Train Random Forest ───────────────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train_sc, y_train)

# ─── 5. Evaluate ──────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy : {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

# Feature importance
fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(fi.round(4))

# ─── 6. Save model + scaler ───────────────────────────────────────────────────
bundle = {"model": clf, "scaler": scaler, "features": list(X.columns)}
with open("model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\n✅  model.pkl saved successfully!")