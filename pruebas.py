

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from keras.models import load_model, Model
from keras.layers import Input, concatenate, Dense


# Cargar modelos existentes
model_vocales = load_model('modelo_vocales001.h5')
model_numeros = load_model('modelo_numeros001.h5')

#Congelar las capas de los modelos hechos

for layer in model_vocales.layers:
    layer.trainable = False

for layer in model_numeros.layers:
    layer.trainable = False


entrada = [28, 28, 3]
num_clases = 10 + 5


# Supongamos que tienes dos modelos CNN llamados model_vocales y model_numeros

# Definir la entrada compartida
input_layer = Input(entrada)

# Salida del modelo de vocales
output_vocales = model_vocales(input_layer)

# Salida del modelo de números
output_numeros = model_numeros(input_layer)

# Concatenar las salidas
concatenated_output = concatenate([output_vocales, output_numeros], axis=-1)

# Capa Dense para la salida final
final_output = Dense(num_clases, activation='softmax')(concatenated_output)

# Crear el modelo combinado
model_combined = Model(inputs=input_layer, outputs=final_output)

# Compilar el modelo
model_combined.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])






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
    predictions = model_combined.predict(image_array)
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





