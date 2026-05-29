import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib 

# --- CONFIGURATION ---
LOOKBACK = 12  # Look at past 60 mins (12 * 5 mins)
PREDICT_AHEAD = 6 # Predict next 30 mins (6 * 5 mins)

# 1. LOAD DATA
try:
    df = pd.read_csv('training_data.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("ERROR: Could not find 'training_data.csv'. Did you rename and move it?")
    exit()

# Select only relevant columns
# Note: Column names might vary based on simulator version. 
# We use standard names: 'CGM' (Glucose), 'CHO' (Carbs), 'insulin' (Insulin)
data = df[['CGM', 'CHO', 'insulin']].fillna(0) 

# 2. NORMALIZE DATA (Scale to 0-1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
joblib.dump(scaler, 'scaler.pkl') # Save scaler for the app later

# 3. CREATE SLIDING WINDOWS
X, y = [], []
for i in range(LOOKBACK, len(scaled_data) - PREDICT_AHEAD):
    X.append(scaled_data[i-LOOKBACK:i]) # The past 12 steps
    y.append(scaled_data[i+PREDICT_AHEAD, 0]) # The future Glucose only (Index 0)

X, y = np.array(X), np.array(y)

# 4. BUILD LSTM MODEL
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1) # Predicts 1 value (Glucose)
])

model.compile(optimizer='adam', loss='mse')

# 5. TRAIN
print("Starting Training... (This will take a few minutes)")
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1)

# 6. SAVE MODEL
model.save('glucose_model.h5')
print("SUCCESS: Model Saved as 'glucose_model.h5'!")