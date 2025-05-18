import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Назви класів Fashion MNIST
class_names = ['Футболка', 'Штани', 'Светр', 'Сукня', 'Пальто', 
               'Сандалі', 'Сорочка', 'Кросівки', 'Сумка', 'Чоботи']

def preprocess_image(image_path):
    """
    Обробка вхідного зображення для відповідності формату Fashion MNIST.
    - Конвертація в градації сірого (1 канал).
    - Зміна розміру до 28x28 пікселів.
    - Нормалізація пікселів до діапазону [0, 1].
    - Переформування до (1, 28, 28, 1) для моделі CNN.
    """
    img = Image.open(image_path).convert('L')  # Конвертація в градації сірого
    img = img.resize((28, 28))  # Зміна розміру до 28x28
    img_array = np.array(img).astype('float32') / 255.0  # Нормалізація
    img_array = img_array.reshape(1, 28, 28, 1)  # Формат для моделі
    return img_array

def predict_image(model, image_path):
    # Обробка зображення
    processed_image = preprocess_image(image_path)
    
    # Прогноз
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100
    
    # Візуалізація
    plt.figure(figsize=(4, 4))
    img = Image.open(image_path)
    plt.imshow(img, cmap='gray')
    plt.title(f'Прогноз: {class_names[predicted_class]}\nЙмовірність: {confidence:.2f}%')
    plt.axis('off')
    plt.show()  # Відображення графіка
    
    return predicted_class, confidence

# Завантаження натренованої моделі
model = tf.keras.models.load_model('fashion_cnn_model.h5')

# Шлях до вашого зображення
image_path = 'workspace\\12st\\images\\5282863229699224654.jpg'  # Замініть на свій шлях

# Прогнозування
predicted_class, confidence = predict_image(model, image_path)
print(f'Передбачений клас: {class_names[predicted_class]}')
print(f'Ймовірність: {confidence:.2f}%')