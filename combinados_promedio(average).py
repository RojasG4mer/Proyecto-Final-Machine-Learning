# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:25:42 2023

@author: Rojas Martinez Jonathan Francisco
"""
# average ensemble model 

# import Average layer
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from tensorflow.keras.layers import Average, Dropout, Dense
from keras.models import load_model, Model
from keras.layers import Input

## Parametro
n = 15 ## Numero de clases de salida

# Cargar modelos existentes
model_vocales = load_model('modelo_vocales001.h5')
model_numeros = load_model('modelo_numeros001.h5')

# get list of models
models = [model_vocales, model_numeros] 

input = Input(shape=(28, 28, 3), name='input') # input layer

# get output for each model input
outputs = [model(input) for model in models]

# Add Dense layers to make sure outputs have the same shape
outputs = [Dense(n, activation='relu')(output) for output in outputs]

# take average of the outputs
x = Average()(outputs)

x = Dense(n, activation='relu')(x) 
x = Dropout(0.3)(x) 
output = Dense(n, activation='softmax', name='output')(x) # output layer

# create average ensembled model
avg_model = Model(input, output)

# Guardar el nuevo modelo
avg_model.save('modelo_combinado_promedios.h5')



############### Viendo resultados del modelo

# Ruta de la carpeta que contiene tus imágenes
folder_path = 'C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\numeros_computadora'

# Obtener la lista de archivos en la carpeta
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

class_names = ['a', 'e', 'i', 'o', 'u', 'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro',
               'Cinco', 'Seis', 'Siete', 'Ocho', 'Nueve'
]

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
    predictions = avg_model.predict(image_array)
    predicted_class = np.argmax(predictions)

    # Mostrar la imagen
    axes[0, i].imshow(image_array[0], cmap=plt.cm.binary)  
    axes[0, i].grid(False)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    axes[0, i].set_xlabel(f'Predicción: {class_names[predicted_class]}')

    # Mostrar el gráfico de barras de confianza
    x_ticks_combined = np.arange(15)  # Cambiar a 15 para reflejar las 15 clases
    axes[1, i].bar(x_ticks_combined, predictions[0], color="#888888")
    axes[1, i].set_xticks(x_ticks_combined)
    axes[1, i].set_xticklabels(class_names)  # Utilizar las etiquetas de las clases directamente
    axes[1, i].set_ylim([0, 1])
    axes[1, i].set_xlabel('Clase')
    axes[1, i].set_ylabel('Confianza de la red')

# Ajustar el tamaño de la figura
fig.set_size_inches(2 * num_cols, 4 * num_rows)

plt.tight_layout()
plt.show()





