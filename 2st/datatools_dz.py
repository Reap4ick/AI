from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# csv read
df = pd.read_csv('workspace\csv_files\orders_sample.csv')
pd.set_option('display.max_columns', None)

# 1
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
print(df)

# 2
df['TotalAmount'] = df['Quantity'] * df['Price']
print(df)

# 3
# a
print("\nTotal store revenue:")
print(df['TotalAmount'].sum())
# b
print("\nThe average value of TotalAmount:")
print(df['TotalAmount'].median())
# c
print("\nThe number of orders for each client:")

dict = {}
for i in df['Customer']:
    if i not in dict:
        dict[i] = 1
    else:
        dict[i] += 1

for customer, count in dict.items():
    print(f"{customer} have {count} orders")

# 4
print("\nThe purchase amount exceeds 5000:")  # Взяв 5000 бо 500 замало було
for i in range(len(df["TotalAmount"])):
    if df['TotalAmount'][i] > 5000:
        print(f"{df['Customer'][i]} have {df['TotalAmount'][i]} orders price (OrderID: {df['OrderID'][i]})")

# 5
df = df.sort_values(by='OrderDate', ascending=False)
print(df)

# 6
print("\nOrders placed in the period from June 5 to June 10 inclusive:")
for i in range(len(df["OrderDate"])):
    if (pd.to_datetime(f"{df['OrderDate'][0].year}-06-05")) <= df['OrderDate'][i] <= (pd.to_datetime(f"{df['OrderDate'][0].year}-06-10")):
        print(f"{df['Customer'][i]} have {df['TotalAmount'][i]} orders price (OrderID: {df['OrderID'][i]})")

# 7
print("\nWorks with categories:")

dict = {}
for i in range(len(df['Category'])):
    category = df['Category'][i]
    quantity = df['Quantity'][i]
    total = df['TotalAmount'][i]
    
    if category not in dict:
        dict[category] = [quantity, total]
    else:
        dict[category][0] += quantity
        dict[category][1] += total

print("\ta) ")
for category, data in dict.items():
    print(f"\t\t{category} contain {data[0]} products")

print("\tb) ")
for category, data in dict.items():
    print(f"\t\t{category} sold of {data[1]:.2f} $")

# 8
dict = {}
for i in range(len(df['Customer'])):
    customer = df['Customer'][i]
    total_amount = df['TotalAmount'][i]

    if customer not in dict:
        dict[customer] = total_amount
    else:
        dict[customer] += total_amount

sorted_spending = sorted(dict.items(), key=lambda x: x[1], reverse=True)[:3]

for customer, total in sorted_spending:
    print(f"{customer} spend {total} $")


# *
dict = {}

for date in df['OrderDate']:
    if date not in dict:
        dict[date] = 1
    else:
        dict[date] += 1

plt.plot(list(dict.keys()), list(dict.values()), marker='o', color='b')
plt.title('Кількість замовлень по датах')
plt.xlabel('Дата')
plt.ylabel('Кількість замовлень')
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()

dict = {}

for i in range(len(df['Category'])):
    category = df['Category'][i]
    total = df['TotalAmount'][i]

    if category not in dict:
        dict[category] = total
    else:
        dict[category] += total

plt.bar(dict.keys(), dict.values(), color='green')
plt.title('Розподіл доходів по категоріях')
plt.xlabel('Категорія')
plt.ylabel('Доходи ($)')
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()
