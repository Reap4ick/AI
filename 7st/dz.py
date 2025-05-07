import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from tensorflow.keras import layers

df = pd.read_csv("workspace\csv_files\Small_Trip_Duration_Dataset.csv")

def time_to_minutes(t):
    h, m = map(int, t.split(":"))
    return h * 60 + m

df["minutes"] = df["time"].apply(time_to_minutes)
X = df["minutes"].values.reshape(-1, 1)
y = df["duration"].values.reshape(-1, 1)

def build_nn_model():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

nn_model = build_nn_model()
nn_model.fit(X, y, epochs=200, verbose=0)

X_test = np.array([10*60 + 30, 0, 2*60 + 40]).reshape(-1, 1)
nn_preds = nn_model.predict(X_test)

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
linreg = LinearRegression()
linreg.fit(X_poly, y)

X_test_poly = poly.transform(X_test)
poly_preds = linreg.predict(X_test_poly)

for t, nn_p, poly_p in zip(["10:30", "00:00", "02:40"], nn_preds, poly_preds):
    print(f"Час: {t} -> Нейронна мережа: {nn_p[0]:.2f} хв, Поліноміальна регресія: {poly_p[0]:.2f} хв")