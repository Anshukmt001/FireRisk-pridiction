"""
train_deep_model.py
Trains a Deep Neural Network (DNN) for forest fire risk detection
Run: python train_deep_model.py
"""

import numpy as np
import pandas as pd
import pickle
import os

np.random.seed(42)

N = 3000

def generate_dataset(n):
    """Generate synthetic dataset with features for fire risk prediction."""
    temperature   = np.random.uniform(15, 50, n)
    humidity      = np.random.uniform(5,  95, n)
    wind_speed    = np.random.uniform(0,  80, n)
    drought_index = np.random.uniform(0, 800, n)
    vegetation    = np.random.uniform(0,   1, n)
    slope         = np.random.uniform(0,  45, n)

    risk_score = (
          0.35 * (temperature   / 50)
        + 0.25 * (1 - humidity  / 95)
        + 0.15 * (wind_speed    / 80)
        + 0.15 * (drought_index / 800)
        + 0.05 * vegetation
        + 0.05 * (slope         / 45)
    )

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

X = df.drop("risk", axis=1).values
y = pd.get_dummies(df["risk"]).values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical

    print("Using TensorFlow/Keras for Deep Learning")
    num_classes = 3
    y_train_cat = to_categorical(np.argmax(y_train, axis=1), num_classes)
    y_test_cat = to_categorical(np.argmax(y_test, axis=1), num_classes)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(6,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        X_train_sc, y_train_cat,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test_sc, y_test_cat, verbose=0)
    print(f"\nTest accuracy : {accuracy:.4f}")
    print(f"Test loss     : {loss:.4f}")

    bundle = {
        "model": model,
        "scaler": scaler,
        "features": ["temperature", "humidity", "wind_speed", "drought_index", "vegetation", "slope"],
        "model_type": "deep_learning"
    }

    with open("deep_model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print("\n✅  deep_model.pkl saved successfully!")

except ImportError:
    print("TensorFlow not installed. Install with: pip install tensorflow")
    print("Falling back to PyTorch...")

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        print("Using PyTorch for Deep Learning")

        class FireRiskNN(nn.Module):
            def __init__(self, input_dim=6, num_classes=3):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.bn1 = nn.BatchNorm1d(128)
                self.fc2 = nn.Linear(128, 64)
                self.bn2 = nn.BatchNorm1d(64)
                self.fc3 = nn.Linear(64, 32)
                self.bn3 = nn.BatchNorm1d(32)
                self.fc4 = nn.Linear(32, num_classes)
                self.dropout = nn.Dropout(0.3)

            def forward(self, x):
                x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
                x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
                x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
                x = self.fc4(x)
                return x

        X_train_t = torch.FloatTensor(X_train_sc)
        y_train_t = torch.LongTensor(np.argmax(y_train, axis=1))
        X_test_t = torch.FloatTensor(X_test_sc)
        y_test_t = torch.LongTensor(np.argmax(y_test, axis=1))

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = FireRiskNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(100):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/100 - Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test_t).sum().item() / len(y_test_t)

        print(f"\nTest accuracy : {accuracy:.4f}")

        bundle = {
            "model": model,
            "scaler": scaler,
            "features": ["temperature", "humidity", "wind_speed", "drought_index", "vegetation", "slope"],
            "model_type": "deep_learning_pytorch"
        }

        with open("deep_model.pkl", "wb") as f:
            pickle.dump(bundle, f)

        print("\n✅  deep_model.pkl saved successfully!")

    except ImportError:
        print("Neither TensorFlow nor PyTorch is installed.")
        print("Please install one: pip install tensorflow  OR  pip install torch")