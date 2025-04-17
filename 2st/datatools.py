from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# csv read
df = pd.read_csv('csv_files\employees.csv')
df = df.sort_values(by='Salary', ascending=True)
arr = [i + 1 for i in range(1, len(df) + 1)]


print("\n Average salary:")
print(df['Salary'].median())

print("\n Standard deviation of salary:")
print(df['Salary'].std())

print("\n Minimum and maximum age of employees:")
print("Min age:",df['Age'].min(),"Max age:",df['Age'].max())



plt.scatter(arr, df['Salary'])
plt.title("Зарплата")
# plt.xlabel("Вік")
plt.ylabel("Зарплата")
plt.grid(True)
plt.show()




# # methods
# print("Head:")
# print(df.head())

# print("\nTail:")
# print(df.tail())

# print("\nSample:")
# print(df.sample(1))

# print("\nInfo:")
# df.info()

# print("\nDescribe:")
# print(df.describe())