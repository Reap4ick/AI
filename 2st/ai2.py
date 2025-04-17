import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Завантаження даних
df = pd.read_csv("energy_usage_plus.csv")
print(df.head())

# Вибір ознак та цільової змінної////temperature,humidity,season,hour,district_type,is_weekend,consumption
X = df[['temperature','humidity','season','hour','district_type','is_weekend']] # features
# temperature,humidity,season,hour,district_type,is_weekend,consumption
y = df['consumption']                                  # target

print("Features: ", X)
print("Target:", y)

# Розділення на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        # Для числових ознак ('year', 'engine_volume', 'mileage', 'horsepower') нічого не змінюємо ('passthrough')
        ('num', 'passthrough', ['temperature','humidity','hour','is_weekend']),
        # Для категоріальних ознак ('brand', 'model') застосовуємо OneHotEncoder
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['season', 'district_type'])
    ])


# Створення та навчання моделі
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
model.fit(X_train, y_train)

# Прогноз та оцінка
# your_apartment = pd.DataFrame([{
#     'area': 81,           # площа в м²
#     'rooms': 2,           # кількість кімнат
#     'floor': 5,           # поверх
#     'year_built': 1990    # рік будівництва
# }])  2006,2.6,37,325,353978.54

# your_apartment = pd.DataFrame([{
#     'year': 2006,
#     'engine_volume':2.6,
#     'mileage': 325,
#     'horsepower':353978.54,
# }])

# your_car = pd.DataFrame([{
#     'brand': 'BMW',          # Приклад бренду
#     'model': 'M3',           # Приклад моделі
#     'year': 2006,
#     'engine_volume': 2.6,
#     'mileage': 37,
#     'horsepower': 325
# }])

# Прогноз ціни
# predicted_price = model.predict(your_car)
# print(f"Прогнозована ціна Машини: {predicted_price[0]:,.2f} $")

y_pred = model.predict(X_test)

# Evaluate the model
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape:.2f}%")

# Візуалізація: справжні ціни vs прогноз
plt.scatter(y_test, y_pred)
plt.xlabel("Справжня ціна")
plt.ylabel("Прогнозована ціна")
plt.title("Справжня vs Прогнозована ціна")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()
