import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def true_function(x):
    return np.sin(x) + 0.1 * x**2

X = np.linspace(-20, 20, 5000).reshape(-1, 1)
y = true_function(X.flatten())

noise = np.random.normal(0, 0.2, size=X.shape[0])
y_noisy = y + noise

model = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42)
model.fit(X, y_noisy)

y_pred = model.predict(X)

mae = mean_absolute_error(y_noisy, y_pred)
mse = mean_squared_error(y_noisy, y_pred)

print(f"Середня абсолютна помилка (MAE): {mae:.4f}")
print(f"Середня квадратична помилка (MSE): {mse:.4f}")

x_value = 7
predicted_value = model.predict(np.array([[x_value]]))[0]
true_value = true_function(x_value)
print(f"Справжнє значення при x={x_value}: {true_value:.4f}")
print(f"Передбачене значення при x={x_value}: {predicted_value:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(X, y, color='green', label='Справжня функція', linewidth=2)
plt.plot(X, y_pred, color='red', label='Передбачена функція (Нейронна мережа)', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Справжня та передбачена функція: sin(x) + 0.1x² (Нейронна мережа)')
plt.legend()
plt.grid(True)
plt.show()