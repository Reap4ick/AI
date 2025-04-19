import matplotlib.pyplot as plt
import numpy as np

# 1
x = np.linspace(-10, 10, 1000)
y = x**2 * np.sin(x)

plt.plot(x, y)
plt.title("Графік функції sin(x)")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid()
plt.show()

# 2
data = np.random.normal(loc=5, scale=2, size=1000)

plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title("Гістограма розподілу")
plt.xlabel("Значення")
plt.ylabel("Частота")
plt.show()

# 3
labels = ['Програмування', 'Спать', 'З кимось вкусно кушать', 'ШАГ',":)"]
sizes = [30, 4.5, 30, 30, 5.5]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Я")
plt.axis('equal')  # рівні осі для круга
plt.show()

# 4
np.random.seed(42)
apple_weights = np.random.normal(loc=150, scale=10, size=100)
banana_weights = np.random.normal(loc=120, scale=15, size=100)
orange_weights = np.random.normal(loc=130, scale=12, size=100)
pear_weights = np.random.normal(loc=140, scale=8, size=100)

data = [apple_weights, banana_weights, orange_weights, pear_weights]
fruit_names = ['Яблука', 'Банани', 'Апельсини', 'Груші']

box = plt.boxplot(data, patch_artist=True, labels=fruit_names)

colors = ['red', 'yellow', 'orange', 'green']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.title('Box-plot маси фруктів')
plt.xlabel('Фрукти')
plt.ylabel('Маса (г)')
plt.grid()

plt.tight_layout()
plt.show()

# 5
x = np.random.uniform(0, 1, 100)
y = np.random.uniform(0, 1, 100)

plt.scatter(x, y, color='green', alpha=0.6)

plt.xlabel('X значення')
plt.ylabel('Y значення')
plt.title('Точкова діаграма з рівномірним розподілом')
plt.grid()

plt.show()

# 6
x = np.linspace(-10, 10, 1000)
y = np.sin(x)
z = np.cos(x)
d = z + y


plt.plot(x, y, label='sin(x)')
plt.plot(x, z, label='cos(x)')
plt.plot(x, d, label='sin(x) + cos(x)')
plt.title("Графік функції sin(x)")
plt.legend(loc='upper right')
plt.xlabel("Функція")
plt.ylabel("Значення")
plt.grid()
plt.show()