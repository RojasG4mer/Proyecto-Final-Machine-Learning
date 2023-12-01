# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:42:21 2023

@author: PC
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from keras.models import load_model, Model
from keras.layers import Input, concatenate, Dense
import tensorflow as tf

# Cargar modelos existentes
model_vocales = load_model('modelo_vocales001.h5')
model_numeros = load_model('modelo_numeros001.h5')

# Elimina la capa de salida de cada modelo
modelo_vocales = Model(inputs=model_vocales.input, outputs=model_vocales.layers[-1].output)
modelo_numeros = Model(inputs=model_numeros.input, outputs=model_numeros.layers[-1].output)

# Crea un nuevo modelo que concatene las salidas de los dos modelos originales
concatenated_output = concatenate([modelo_vocales.output, modelo_numeros.output])

# Agrega una capa densa para la salida final que combine la información
# Puedes ajustar la cantidad de unidades según tu necesidad
combined_output = tf.keras.layers.Dense(10, activation='softmax')(concatenated_output)

# Crea el nuevo modelo combinado
modelo_combinado = Model(inputs=[modelo_vocales.input, modelo_numeros.input], outputs=combined_output)

# Compila el modelo con el optimizador, la función de pérdida y métricas adecuadas
modelo_combinado.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




############### Viendo resultados del modelo

class_names = ['a', 'e', 'i', 'o', 'u', '0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9'
]

# Ruta de la carpeta que contiene tus imágenes
folder_path = 'C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\pruebas'

# Obtener la lista de archivos en la carpeta
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Crear una figura con subgráficos dinámicamente según la cantidad de imágenes
num_images = len(image_files)
num_rows = 2
num_cols = num_images
fig, axes = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 4 * num_rows))

# Crear un conjunto de datos de prueba con etiquetas derivadas del nombre del archivo
X_test = []
y_test = []

for i, image_file in enumerate(image_files):
    # Cargar la imagen
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path).convert('RGB')  # Convertir a RGB
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0  # Sin necesidad de expand_dims para RGB
    image_array = np.expand_dims(image_array, axis=0)  # Añadir dimensión de lote

    # Obtener la etiqueta del nombre del archivo
    label = os.path.splitext(image_file)[0]  # Eliminar la extensión (.jpg)

    # Realizar la predicción
    predictions = modelo_combinado.predict([image_array, image_array])
    predicted_class = np.argmax(predictions)

    # Agregar la imagen y su etiqueta al conjunto de datos de prueba
    X_test.append(image_array)
    y_test.append(label)

    # Mostrar la imagen
    axes[0, i].imshow(image_array[0], cmap=plt.cm.binary)
    axes[0, i].grid(False)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    axes[0, i].set_xlabel(f'Predicción: {class_names[predicted_class]}')

# Convertir el conjunto de datos de prueba a formato numpy
X_test = np.concatenate(X_test, axis=0)

# Codificación one-hot de las etiquetas
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=5)

# Evaluar el modelo combinado en el conjunto de datos de prueba
evaluacion = modelo_combinado.evaluate([X_test, X_test], tf.keras.utils.to_categorical(y_test))

# Imprimir la métrica de interés (accuracy en este caso)
accuracy = evaluacion[1]
print(f'Accuracy del modelo combinado en el conjunto de prueba: {accuracy}')


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
    predictions = modelo_combinado.predict([image_array,image_array])
    predicted_class = np.argmax(predictions)

    # Mostrar la imagen
    axes[0, i].imshow(image_array[0], cmap=plt.cm.binary)  
    axes[0, i].grid(False)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    axes[0, i].set_xlabel(f'Predicción: {class_names[predicted_class]}')

    # Mostrar el gráfico de barras de confianza
    num_classes_combined = 5  # Ajusta esto según la cantidad de clases en tu problema combinado
    x_ticks_combined = np.arange(num_classes_combined)
    axes[1, i].bar(x_ticks_combined, predictions[0][:num_classes_combined], color="#888888")
    axes[1, i].set_xticks(x_ticks_combined)
    axes[1, i].set_xticklabels(class_names[:num_classes_combined])  # Utilizar las etiquetas de las clases directamente
    axes[1, i].set_ylim([0, 1])
    axes[1, i].set_xlabel('Clase')
    axes[1, i].set_ylabel('Confianza de la red')

# Ajustar el tamaño de la figura
fig.set_size_inches(2 * num_cols, 4 * num_rows)

plt.tight_layout()
plt.savefig('Prueba_combinados_general.jpg')
plt.show()





