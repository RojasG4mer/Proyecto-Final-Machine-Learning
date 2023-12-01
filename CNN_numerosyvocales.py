# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:04:43 2023

@author: PC
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import os
from PIL import Image
import tensorflow_datasets as tfds
import math
import logging
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

"""
¿como combinar datasets de la forma MapDataset y DirectoryIteration?

"""

## Dataset de las vocales

# Ruta principal que contiene las carpetas 1, 2, ..., 79
data_dir = "C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\letras_computadora"

class_names = ['a', 'e', 'i', 'o', 'u', 'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro',
               'Cinco', 'Seis', 'Siete', 'Ocho', 'Nueve']

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

## Dataset de los numeros

# Establecer la semilla para TensorFlow y NumPy
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Configurar el nivel de registro
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Cargar el conjunto de datos MNIST
dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']


num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

#Normalizar: Numeros de 0 a 255, que sean de 0 a 1
def normalize(images, labels):
    images = tf.repeat(images, 3, axis=-1)  # Duplica el canal único en tres canales RGB
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

### Concatenar los datasets de train y test

# Convertir los generadores de imágenes a conjuntos de datos de TensorFlow
train_ds_letters = tf.data.Dataset.from_generator(
    lambda: train_generator_letters,
    output_signature=(
        tf.TensorSpec(shape=(28, 28, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(len(class_names),), dtype=tf.float32)  # Usar tf.float32 para las etiquetas
    )
)

validation_ds_letters = tf.data.Dataset.from_generator(
    lambda: validation_generator_letters,
    output_signature=(
        tf.TensorSpec(shape=(28, 28, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(len(class_names),), dtype=tf.float32)  # Usar tf.float32 para las etiquetas
    )
)

# Convertir el conjunto de datos de números a la misma estructura que el de letras
train_dataset = train_dataset.map(lambda img, label: (img, tf.cast(label, tf.float32)))
test_dataset = test_dataset.map(lambda img, label: (img, tf.cast(label, tf.float32)))

# Concatenar los conjuntos de datos de letras y números
train_juntos = train_ds_letters.concatenate(train_dataset)
test_juntos = validation_ds_letters.concatenate(test_dataset)




##### RED NEURONAL CONVOLUCIONAL

# Estructura de la red neuronal convolucional (CNN)
modelo_lyn = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compilar el modelo CNN
modelo_lyn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Aprender con el conjunto de datos de letras
modelo_lyn.fit(
    train_juntos,
    epochs=epocas,
    validation_data=test_juntos
)


## Guardamos el modelo
modelo_lyn.save('modelo_datasetscombinados.h5')






"""
## Guardamos el modelo
modelo_lyn.save('modelo_datasetscombinados.h5')

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
    predictions = modelo_lyn.predict(image_array)
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
"""