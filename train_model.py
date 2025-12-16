# ==============================
# Mood Music AI - Train Model
# ==============================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# ------------------------------
# 1. Load Dataset
# ------------------------------
data = pd.read_csv("dataset/spotify.csv")
print("Dataset loaded successfully")
print(data.head())

# Handle missing values
data = data.dropna()
print(f"Dataset shape after dropping missing values: {data.shape}")


# ------------------------------
# 2. Select Features & Labels
# ------------------------------
features = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo'
]

X = data[features]
y = data['labels']  # Labels are integers: 0=Sad, 1=Happy, 2=Energetic, 3=Calm


# ------------------------------
# 3. Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ------------------------------
# 4. Feature Scaling
# ------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------------------
# 5. Build Deep Learning Model
# ------------------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(len(np.unique(y)), activation='softmax')
])


# ------------------------------
# 6. Compile Model
# ------------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ------------------------------
# 7. Train Model
# ------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)


# ------------------------------
# 8. Evaluate Model
# ------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Classification Report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
target_names = ['Sad', 'Happy', 'Energetic', 'Calm']
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=target_names))


# ------------------------------
# 9. Save Model & Scaler
# ------------------------------
model.save("model/mood_model.h5")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler saved successfully!")
