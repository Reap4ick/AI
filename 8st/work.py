import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Зчитування CSV
csv_path = "workspace/csv_files/Cleaned_Daily_Energy_Consumption.csv"
df = pd.read_csv(csv_path, parse_dates=["date"])

# Перевірка на наявність необхідних колонок
if "date" not in df.columns or "consumption_kwh" not in df.columns:
    raise ValueError("CSV-файл має містити стовпці 'date' і 'consumption_kwh'.")

# 2. Додавання номера місяця
df["month_num"] = df["date"].dt.month

# 3. Підготовка навчальних даних
x_train = df["month_num"].values.reshape(-1, 1)
y_train = df["consumption_kwh"].values.reshape(-1, 1)

# Перевірка на пропущені значення
if np.isnan(x_train).any() or np.isnan(y_train).any():
    raise ValueError("У даних є пропущені значення. Очистіть їх перед навчанням.")

# 4. Побудова моделі
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(1,)),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 5. Навчання моделі
model.fit(x_train, y_train, epochs=200, verbose=0)

# 6. Прогноз для April (4), July (7), December (12)
months_to_predict = np.array([[4], [7], [12]])
predictions = model.predict(months_to_predict)

# 7. Виведення результатів
for i, month in enumerate(["April", "July", "December"]):
    print(f"{month}: {predictions[i][0]:.2f} kWh")
