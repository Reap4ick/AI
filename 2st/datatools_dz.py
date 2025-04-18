from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('csv_files\orders_sample.csv')
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
order_counts = df['Customer'].value_counts()
for customer, count in order_counts.items():
    print(f"{customer} have {count} orders")

# 4
print("\nThe purchase amount exceeds 5000:")
high_value_orders = df[df['TotalAmount'] > 5000]
for _, row in high_value_orders.iterrows():
    print(f"{row['Customer']} have {row['TotalAmount']} orders price (OrderID: {row['OrderID']})")

# 5
df = df.sort_values(by='OrderDate', ascending=False)
print(df)

# 6
print("\nOrders placed in the period from June 5 to June 10 inclusive:")
june_orders = df[(df['OrderDate'].dt.month == 6) & (df['OrderDate'].dt.day >= 5) & (df['OrderDate'].dt.day <= 10)]
for _, row in june_orders.iterrows():
    print(f"{row['Customer']} have {row['TotalAmount']} orders price (OrderID: {row['OrderID']})")

# 7
print("\nWorks with categories:")
category_stats = df.groupby('Category').agg({'Quantity': 'sum', 'TotalAmount': 'sum'})

print("\ta) ")
for category, quantity in category_stats['Quantity'].items():
    print(f"\t\t{category} contain {quantity} products")

print("\tb) ")
for category, total in category_stats['TotalAmount'].items():
    print(f"\t\t{category} sold of {total:.2f} $")

# 8
print("\nTop 3 customers by spending:")
customer_spending = df.groupby('Customer')['TotalAmount'].sum()
top_3_customers = customer_spending.nlargest(3)
for customer, total in top_3_customers.items():
    print(f"{customer} spend {total} $")

# *
order_counts = df['OrderDate'].value_counts().sort_index()
plt.plot(order_counts.index, order_counts.values, marker='o', color='b')
plt.title('Кількість замовлень по датах')
plt.xlabel('Дата')
plt.ylabel('Кількість замовлень')
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()

category_revenue = df.groupby('Category')['TotalAmount'].sum()
plt.bar(category_revenue.index, category_revenue.values, color='green')
plt.title('Розподіл доходів по категоріях')
plt.xlabel('Категорія')
plt.ylabel('Доходи ($)')
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()