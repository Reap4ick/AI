import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

csv_path = "workspace/csv_files/Cleaned_Daily_Energy_Consumption.csv"
df = pd.read_csv(csv_path, parse_dates=["date"])

df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["weekday"] = df["date"].dt.weekday
df["is_weekend"] = df["weekday"] >= 5

x_train = df[["day", "month", "weekday", "is_weekend"]].astype(float).values
y_train = df["consumption_kwh"].values.reshape(-1, 1)

if np.isnan(x_train).any() or np.isnan(y_train).any():
    raise ValueError("У даних є пропущені значення.")

model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=["mae"])

model.fit(x_train, y_train, epochs=200, verbose=1)

test_days = pd.DataFrame({
    "day": [10, 20, 31],
    "month": [4, 7, 12],
    "weekday": [3, 5, 1],   
    "is_weekend": [0, 1, 0]
})

x_test = test_days.astype(float).values
predictions = model.predict(x_test)

for i, row in test_days.iterrows():
    date_str = f"{int(row['day'])}.{int(row['month'])}"
    print(f"{date_str}: {predictions[i][0]:.2f} kWh")
