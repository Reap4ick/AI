import matplotlib.pyplot as plt
import numpy as np

# fig = plt.figure()  # створює порожню фігуру

# fig, ax = plt.subplots()  # створює фігуру з однією віссю

# ax.plot([1, 2, 3], [4, 5, 6])  # малює лінійний графік

# графіків функцій
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
plt.grid(True)
plt.show()
###################################################
group1 = np.random.normal(70, 10, 100)
group2 = np.random.normal(80, 5, 100)
group3 = np.random.normal(65, 15, 100)

groups = [group1, group2, group3]
labels = ['Group A', 'Group B', 'Group C']

fig = plt.figure(figsize=(10, 7))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(groups, labels=labels)
plt.xlabel('Groups')
ax.set_ylabel('Values')
ax.set_title('Boxplot of Group Distributions')

plt.show()
###################################################
x = np.random.uniform(0, 1, 100)
y = np.random.uniform(0, 1, 100)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='green', alpha=0.6)

plt.xlabel('X значення')
plt.ylabel('Y значення')
plt.title('Точкова діаграма з рівномірним розподілом')

plt.show()