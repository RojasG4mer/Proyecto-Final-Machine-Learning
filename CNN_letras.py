# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:12:46 2023
@author: Rojas Martinez Jonathan Francisco
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
data_dir = "C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\English Alphabet Dataset"

class_names = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
    "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ",", ";",
    ":", "?", "!", ".", "@",  "#", "$", "%", "&", "(", ")", "{", "}", "[", "]"
]
## Parametros modificables
batch_size = 32
epocas = 8


# Crear generadores de datos de entrenamiento y validación
datagen_letters = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator_letters = datagen_letters.flow_from_directory(
    data_dir,
    target_size=(28, 28),
    batch_size=batch_size,
    color_mode='rgb',  # Cambiado a formato RGB
    class_mode='categorical',
    subset='training'
)

validation_generator_letters = datagen_letters.flow_from_directory(
    data_dir,
    target_size=(28, 28),
    batch_size=batch_size,
    color_mode='rgb',  # Cambiado a formato RGB
    class_mode='categorical',
    subset='validation'
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



# Realizar la predicción
folder_path = 'C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\letras_computadora'
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

plt.tight_layout()
plt.show()