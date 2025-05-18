import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Завантаження даних Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Нормалізація даних
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Переформування вхідних даних для CNN (28x28x1)
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# One-hot кодування міток
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Створення моделі CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Компіляція моделі
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Налаштування ранньої зупинки
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Тренування моделі
history = model.fit(
    x_train_cnn, 
    y_train_cat, 
    epochs=30,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Збереження моделі
model.save('fashion_cnn_model.h5')

# Оцінка моделі на тестових даних
test_loss, test_acc = model.evaluate(x_test_cnn, y_test_cat)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Побудова графіків точності та похибки
plt.figure(figsize=(12, 4))

# Графік точності
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Тренувальна точність')
plt.plot(history.history['val_accuracy'], label='Валідаційна точність')
plt.title('Точність моделі')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()
plt.grid(True)

# Графік похибки
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Тренувальна похибка')
plt.plot(history.history['val_loss'], label='Валідаційна похибка')
plt.title('Похибка моделі')
plt.xlabel('Епоха')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()  # Відображення графіків

# Прогнозування для перших 5 тестових зображень
predictions = model.predict(x_test_cnn[:5])
class_names = ['Футболка', 'Штани', 'Светр', 'Сукня', 'Пальто', 
               'Сандалі', 'Сорочка', 'Кросівки', 'Сумка', 'Чоботи']

# Візуалізація прогнозів
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i], cmap='gray')
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    plt.title(f'Прогноз: {class_names[predicted_label]}\nСправжній: {class_names[true_label]}')
    plt.axis('off')
plt.tight_layout()
plt.show()  # Відображення прогнозів

# Функція для обробки нового зображення
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Конвертація в градації сірого
    img = img.resize((28, 28))  # Зміна розміру до 28x28
    img_array = np.array(img).astype('float32') / 255.0  # Нормалізація
    img_array = img_array.reshape(1, 28, 28, 1)  # Формат для моделі
    return img_array

# Функція для прогнозування нового зображення
def predict_image(model, image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100
    
    # Візуалізація результату
    plt.figure(figsize=(4, 4))
    img = Image.open(image_path)
    plt.imshow(img, cmap='gray')
    plt.title(f'Прогноз: {class_names[predicted_class]}\nЙмовірність: {confidence:.2f}%')
    plt.axis('off')
    plt.show()  # Відображення результату
    
    return predicted_class, confidence

# Завантаження натренованої моделі
model = tf.keras.models.load_model('fashion_cnn_model.h5')

# Шлях до вашого зображення
image_path = 'workspace\\12st\\images\\test.jpg'  # Замініть на свій шлях

# Прогнозування
predicted_class, confidence = predict_image(model, image_path)
print(f'Передбачений клас: {class_names[predicted_class]}')
print(f'Ймовірність: {confidence:.2f}%')