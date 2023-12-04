# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:30:17 2023

@author: PC
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

# Establecer la semilla para TensorFlow y NumPy
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
epocas = 5
# Configurar el nivel de registro
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Cargar el conjunto de datos MNIST
dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = [
    'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis',
    'Siete', 'Ocho', 'Nueve'
]

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

#Estructura de la red
model = tf.keras.Sequential([
	Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

#Indicar las funciones a utilizar
model.compile(
	optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
)

#Aprendizaje por lotes de 32 cada lote
BATCHSIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCHSIZE)
test_dataset = test_dataset.batch(BATCHSIZE)

#Realizar el aprendizaje
model.fit(
	train_dataset, epochs=epocas,
	steps_per_epoch=math.ceil(num_train_examples/BATCHSIZE) #No sera necesario pronto
)

#Evaluar nuestro modelo ya entrenado, contra el dataset de pruebas
test_loss, test_accuracy = model.evaluate(
	test_dataset, steps=math.ceil(num_test_examples/32)
)

print("Resultado en las pruebas: ", test_accuracy)


from io import StringIO
import sys
model.summary()
# Redirigir la salida estándar a un objeto StringIO
buffer = StringIO()
sys.stdout = buffer

# Imprimir el resumen del modelo
model.summary()

# Restaurar la salida estándar
sys.stdout = sys.__stdout__

# Obtener el contenido del buffer como una cadena
summary_str = buffer.getvalue()

# Crear una imagen con el resumen
fig, ax = plt.subplots()
ax.text(0.1, 0.5, summary_str, wrap=True, fontsize=8, va='center')
ax.axis('off')

# Guardar la imagen
plt.savefig('summary_image_numeros.png', format='png', bbox_inches='tight')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Crear conjuntos de entrenamiento y validación
train_dataset = train_dataset.take(num_train_examples)
validation_dataset = train_dataset.skip(num_train_examples)

# Entrenar el modelo y almacenar el historial
history = model.fit(
    train_dataset,
    epochs=epocas,
    validation_data=validation_dataset
)

# Evaluar nuestro modelo ya entrenado contra el dataset de pruebas
test_loss, test_accuracy = model.evaluate(
    test_dataset, steps=math.ceil(num_test_examples/BATCHSIZE)
)

print("Resultado en las pruebas: ", test_accuracy)

# Obtener las etiquetas reales del conjunto de pruebas
true_labels = np.concatenate([y for x, y in test_dataset], axis=0)

# Obtener las probabilidades predichas para el conjunto de pruebas
predicted_probs = model.predict(test_dataset)

# Obtener las etiquetas predichas tomando la clase con la probabilidad más alta
predicted_labels = np.argmax(predicted_probs, axis=1)

# Crear la matriz de confusión
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Obtener accuracy y loss del historial de entrenamiento
accuracy = history.history['accuracy'][-1]
loss = history.history['loss'][-1]

# Configurar el estilo de la visualización
sns.set(font_scale=1.2)
plt.figure(figsize=(10, 8))

# Crear el mapa de calor con seaborn
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

# Añadir etiquetas y título
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas Verdaderas')
plt.title('Matriz de Confusión Numeros')

# Añadir anotaciones para accuracy y loss
#plt.annotate(f'Accuracy: {accuracy:.4f}', xy=(0.5, -0.15), ha='center', va='center', fontsize=12)
#plt.annotate(f'Loss: {loss:.4f}', xy=(0.5, -0.20), ha='center', va='center', fontsize=12)

# Guardar la figura como una imagen
plt.savefig('confusion_matrix_numeros.png', format='png')
plt.show()




#Guardamos el modelo de las letras
model.save('modelo_numeros001.h5')


import os
from PIL import Image

# Ruta de la carpeta que contiene tus imágenes
folder_path = 'C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\numeros_computadora'

# Obtener la lista de archivos en la carpeta
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Crear una figura con subgráficos dinámicamente según la cantidad de imágenes
num_images = len(image_files)
num_rows = 2
num_cols = num_images
fig, axes = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 4 * num_rows))

for i, image_file in enumerate(image_files):
    # Cargar la imagen
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path).convert('RGB')  # Convertir a RGB
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0  # Sin necesidad de expand_dims para RGB
    image_array = np.expand_dims(image_array, axis=0)  # Añadir dimensión de lote

    # Realizar la predicción
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)

    # Mostrar la imagen
    axes[0, i].imshow(image_array[0], cmap=plt.cm.binary)  
    axes[0, i].grid(False)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    axes[0, i].set_xlabel(f'Predicción: {class_names[predicted_class]}')

    # Mostrar el gráfico de barras de confianza
    x_ticks = np.arange(10)
    axes[1, i].bar(x_ticks, predictions[0], color="#888888")
    axes[1, i].set_xticks(x_ticks)
    axes[1, i].set_xticklabels(x_ticks)
    axes[1, i].set_ylim([0, 1])
    axes[1, i].set_xlabel('Clase')
    axes[1, i].set_ylabel('Confianza de la red')

# Ajustar el tamaño de la figura
fig.set_size_inches(2 * num_cols, 4 * num_rows)

plt.tight_layout()
plt.savefig('Prueba_numeros.jpg')
plt.show()


