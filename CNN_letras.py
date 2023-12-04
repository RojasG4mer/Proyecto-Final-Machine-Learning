# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:12:46 2023
@author: Rojas Martinez Jonathan Francisco

class_names = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
    "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ",", ";",
    ":", "?", "!", ".", "@",  "#", "$", "%", "&", "(", ")", "{", "}", "[", "]"
]

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import os
from PIL import Image


# Ruta principal que contiene las carpetas 1, 2, ..., 79
data_dir = "C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\letras_computadora"

class_names = ['a', 'e', 'i', 'o', 'u']


## Parametros modificables
batch_size = 32
epocas = 8
seed = 42

# Definir una función de normalización
def normalize_image(image):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image

# Crear generadores de datos de entrenamiento y validación con normalización incorporada
datagen_letters = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    preprocessing_function=normalize_image  # Aquí se aplica la normalización
)

train_generator_letters = datagen_letters.flow_from_directory(
    data_dir,
    target_size=(28, 28),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    subset='training',
    seed=seed
)

validation_generator_letters = datagen_letters.flow_from_directory(
    data_dir,
    target_size=(28, 28),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation',
    seed=seed
)

# Estructura de la red neuronal convolucional (CNN)
model_cnn = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compilar el modelo CNN
model_cnn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Aprender con el conjunto de datos de letras
model_cnn.fit(
    train_generator_letters,
    epochs=epocas,
    validation_data=validation_generator_letters
)

from io import StringIO
import sys
model_cnn.summary()
# Redirigir la salida estándar a un objeto StringIO
buffer = StringIO()
sys.stdout = buffer

# Imprimir el resumen del modelo
model_cnn.summary()

# Restaurar la salida estándar
sys.stdout = sys.__stdout__

# Obtener el contenido del buffer como una cadena
summary_str = buffer.getvalue()

# Crear una imagen con el resumen
fig, ax = plt.subplots()
ax.text(0.1, 0.5, summary_str, wrap=True, fontsize=8, va='center')
ax.axis('off')

# Guardar la imagen
plt.savefig('summary_image.png', format='png', bbox_inches='tight')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Entrenar el modelo y almacenar el historial
history = model_cnn.fit(
    train_generator_letters,
    epochs=epocas,
    validation_data=validation_generator_letters
)


# Obtener las etiquetas reales del conjunto de validación
true_labels = validation_generator_letters.classes
# Obtener las probabilidades predichas para el conjunto de validación
predicted_probs = model_cnn.predict_generator(validation_generator_letters)

# Obtener las etiquetas predichas tomando la clase con la probabilidad más alta
predicted_labels = np.argmax(predicted_probs, axis=1)


# Obtener accuracy y loss del historial de entrenamiento
accuracy = history.history['accuracy'][-1]
loss = history.history['loss'][-1]

# Crear la matriz de confusión
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Configurar el estilo de la visualización
sns.set(font_scale=1.2)
plt.figure(figsize=(10, 8))

# Configurar el estilo de la visualización
sns.set(font_scale=1.2)
plt.figure(figsize=(8, 6))

# Crear el mapa de calor con seaborn
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

# Añadir etiquetas y título
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas Verdaderas')
plt.title('Matriz de Confusión vocales')

# Añadir anotaciones para accuracy y loss
plt.annotate(f'Accuracy: {accuracy:.4f}', xy=(0.5, -0.15), ha='center', va='center', fontsize=12)
plt.annotate(f'Loss: {loss:.4f}', xy=(0.5, -0.20), ha='center', va='center', fontsize=12)


# Guardar la figura como una imagen
plt.savefig('confusion_matrix.png', format='png')
plt.show()


## Guardamos el modelo de las vocales
model_cnn.save('modelo_vocales001.h5')









# Realizar la predicción
folder_path = 'C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\pruebas'
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Crear una figura con subgráficos dinámicamente según la cantidad de imágenes
num_images = len(image_files)
num_rows = 2
num_cols = num_images
fig, axes = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 4 * num_rows))

for i, image_file in enumerate(image_files):
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path).convert('RGB')  # Cambiado a formato RGB
    image = image.resize((28, 28))
    image_array = np.expand_dims(np.array(image), axis=0) / 255.0

    # Realizar la predicción
    predictions = model_cnn.predict(image_array)
    predicted_class = np.argmax(predictions)

    # Mostrar la imagen
    axes[0, i].imshow(image_array[0], cmap=plt.cm.binary)  
    axes[0, i].grid(False)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    axes[0, i].set_xlabel(f'Predicción: {class_names[predicted_class]}')

    # Ajustar el espacio entre los valores del eje x
    fig.subplots_adjust(bottom=0.85)  # Puedes ajustar el valor según tus necesidades
    
    # Mostrar el gráfico de barras de confianza
    x_ticks = np.arange(len(class_names))  # Ajustado al número de clases
    axes[1, i].bar(x_ticks, predictions[0], color="#888888")
    axes[1, i].set_xticks(x_ticks)
    axes[1, i].set_xticklabels(class_names, rotation=0)  # Añadido para mostrar etiquetas
    axes[1, i].set_xlabel('Clase')
    axes[1, i].set_ylabel('Confianza de la red')

    # Ajustar el tamaño de la figura
    fig.set_size_inches(2 * num_cols, 4 * num_rows)

plt.savefig('Prueba_vocales001.jpg')
plt.tight_layout()
plt.show()




