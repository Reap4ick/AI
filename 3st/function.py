import matplotlib.pyplot as plt
import numpy as np

# fig = plt.figure()  # створює порожню фігуру

# fig, ax = plt.subplots()  # створює фігуру з однією віссю

# ax.plot([1, 2, 3], [4, 5, 6])  # малює лінійний графік

# графіків функцій
x = np.linspace(-10, 10, 1000)
y = x**2 * np.sin(x)

plt.plot(x, y)
plt.title("Графік функції sin(x)")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()

# гістограм розподілу змінних
data = np.random.normal(loc=5, scale=2, size=1000)

# Побудова гістограми
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title("Гістограма розподілу")
plt.xlabel("Значення")
plt.ylabel("Частота")
plt.show()

# Pie Chart
labels = ['Автомобіль', 'Велосипед', 'Громадський транспорт', 'Пішки']
sizes = [50, 20, 25, 5]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title(" да наверно????")
plt.axis('equal')  # рівні осі для круга
plt.show()


group1 = np.random.normal(70, 10, 100)
group2 = np.random.normal(75, 7, 100)
group3 = np.random.normal(65, 12, 100)

groups = [group1, group2, group3]
labels = ['Group 1', 'Group 2', 'Group 3']

fig = plt.figure(figsize =(10, 7))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(groups)

plt.show()