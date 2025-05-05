import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('workspace/csv_files/internship_candidates_cefr_final.csv')

english_level_mapping = {
    'Elementary': 0,
    'Pre-Intermediate': 1,
    'Intermediate': 2,
    'Upper-Intermediate': 3,
    'Advanced': 4
}
df['EnglishLevel'] = df['EnglishLevel'].map(english_level_mapping)

if df['EnglishLevel'].isna().any():
    print("Помилка: Деякі значення EnglishLevel не розпізнані. Доступні рівні:", list(english_level_mapping.keys()))
    exit()

print("Закодовані рівні англійської:", english_level_mapping)

print("\nРозподіл рівнів англійської:")
print(df['EnglishLevel'].value_counts().sort_index())
print("\nВідсоток прийняття за EnglishLevel:")
print(df.groupby('EnglishLevel')['Accepted'].mean())
print("\nСередні значення ознак за EnglishLevel:")
print(df.groupby('EnglishLevel')[['Experience', 'Grade', 'Age', 'EntryTestScore', 'Accepted']].mean())

X = df[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
y = df['Accepted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nТочність:", accuracy_score(y_test, y_pred))
print("Прецизійність:", precision_score(y_test, y_pred))
print("Повнота:", recall_score(y_test, y_pred))
print("Матриця помилок:\n", confusion_matrix(y_test, y_pred))

feature_names = X_train.columns
print("Коефіцієнти моделі:", dict(zip(feature_names, model.coef_[0])))

test_scores = np.linspace(X['EntryTestScore'].min(), X['EntryTestScore'].max(), 100)
probabilities = []
mean_values = X_train[['Experience', 'Grade', 'EnglishLevel', 'Age']].mean().values

for score in test_scores:
    input_data = pd.DataFrame([[mean_values[0], mean_values[1], mean_values[2], mean_values[3], score]],
                              columns=feature_names)
    prob = model.predict_proba(input_data)[0][1]
    probabilities.append(prob)

plt.figure(figsize=(8, 6))
plt.plot(test_scores, probabilities, 'b-', marker='o', markersize=5, label='Ймовірність прийняття')
plt.title('Ймовірність прийняття залежно від балу вступного тесту')
plt.xlabel('Бал вступного тесту')
plt.ylabel('Ймовірність прийняття')
plt.grid(True)
plt.legend()
plt.show()

english_levels = list(english_level_mapping.values())
english_labels = list(english_level_mapping.keys())
probabilities = []

mean_values_by_level = df.groupby('EnglishLevel')[['Experience', 'Grade', 'Age', 'EntryTestScore']].mean()

print("\nЙмовірності прийняття за рівнем англійської:")
for level, label in zip(english_levels, english_labels):
    mean_values = mean_values_by_level.loc[level][['Experience', 'Grade', 'EntryTestScore', 'Age']].values
    input_data = pd.DataFrame([[mean_values[0], mean_values[1], level, mean_values[3], mean_values[2]]],
                              columns=feature_names)
    prob = model.predict_proba(input_data)[0][1]
    probabilities.append(prob)
    print(f"{label}: {prob:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(english_labels, probabilities, 'ro-', marker='o', markersize=10, linewidth=1, label='Ймовірність прийняття')
plt.title('Ймовірність прийняття залежно від рівня англійської мови')
plt.xlabel('Рівень англійської мови')
plt.ylabel('Ймовірність прийняття')
plt.grid(True)
plt.legend()
plt.show()

try:
    new_data = pd.DataFrame({
        'Experience': [1],
        'Grade': [12],
        'EnglishLevel': [english_level_mapping['Pre-Intermediate']],
        'Age': [16],
        'EntryTestScore': [600]
    })
    predictions = model.predict(new_data)
    print("Прогноз для нового кандидата:", predictions)
except KeyError as e:
    print("Помилка: Рівень англійської не розпізнаний. Доступні рівні:", list(english_level_mapping.keys()))
except Exception as e:
    print("Помилка:", e)