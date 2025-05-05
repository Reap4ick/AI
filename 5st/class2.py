import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import os

print("Поточна робоча директорія:", os.getcwd())

data = pd.read_csv('workspace/csv_files/fuel_consumption_vs_speed.csv')

speed_kmh = data['speed_kmh'].values.reshape(-1, 1)
fuel_consumption = data['fuel_consumption_l_per_100km'].values

degrees = [3]

mse_values = []
models = []

for degree in degrees:
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(speed_kmh, fuel_consumption)
    
    y_pred = polyreg.predict(speed_kmh)
    
    mse = mean_squared_error(fuel_consumption, y_pred)
    mse_values.append(mse)
    models.append(polyreg)
    
    print(f"Поліном {degree}-го ступеня: MSE = {mse:.4f}")

best_degree = degrees[np.argmin(mse_values)]
best_mse = min(mse_values)
best_model = models[np.argmin(mse_values)]
print(f"\nНайкраща модель: Поліном {best_degree}-го ступеня з MSE = {best_mse:.4f}")

test_speeds = np.array([35, 95, 140]).reshape(-1, 1)
predictions = best_model.predict(test_speeds)

for speed, pred in zip([35, 95, 140], predictions):
    print(f"Передбачені витрати пального при {speed} км/год: {pred:.2f} л/100км")

plt.figure(figsize=(10, 6))
plt.scatter(speed_kmh, fuel_consumption, color='blue', label='Дані', s=100)

x_range = np.linspace(10, 120, 100).reshape(-1, 1)

for degree, model in zip(degrees, models):
    y_range = model.predict(x_range)
    plt.plot(x_range, y_range, label=f'Поліном {degree}-го ступеня', linewidth=2)

plt.ylim(5, 12)

plt.xlabel('Швидкість (км/год)')
plt.ylabel('Витрати пального (л/100км)')
plt.title('Поліноміальна регресія: Витрати пального залежно від швидкості')
plt.legend()
plt.grid(True)
plt.show()