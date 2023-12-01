# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:15:52 2023

@author: Rojas Martinez Jonathan Francisco
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:09:45 2023

@author: PC
"""

from PIL import Image
import os
import numpy as np
# Carga la imagen
path_iamgenesoriginales = "C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\imagenes_originales"

paths_carpetas = ['C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\letras_computadora\\a',
         'C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\letras_computadora\\e', 
         'C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\letras_computadora\\i',
         'C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\letras_computadora\\o',
         'C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\letras_computadora\\u', 
         "C:\\Users\\PC\\Desktop\\Universidad\\Machine Learning\\Proyecto_final\\imagenes_recortadas"]

# Obtener la lista de nombres de archivo en la carpeta
nombres_archivos = os.listdir(path_iamgenesoriginales)

# Filtrar solo los archivos de imagen (puedes ajustar las extensiones según tus necesidades)
archivos_imagen = [archivo for archivo in nombres_archivos if archivo.lower().endswith(('.png', '.jpg', '.jpeg'))]

i = 0 ## Para saber con que nombre guardar las imagenes

# Iterar sobre cada archivo de imagen y abrirlo
for archivo_imagen in archivos_imagen:
    ruta_imagen = os.path.join(path_iamgenesoriginales, archivo_imagen)
    imagen = Image.open(ruta_imagen)

    ### Numero de imagenes
    nombre = [f for f in range(50, 200)]


    # Tamaño de la imagen
    ancho, alto = imagen.size

    # Número de columnas y filas
    columnas = 2
    filas = 3

    # Tamaño de cada fragmento
    ancho_fragmento = (ancho) // (columnas )
    alto_fragmento = (alto) // (filas )


    n = 10  # Número de píxeles para quitar de los bordes
    
    j = 0 ## Para saber en que carpeta de las 5 guardarlas
    # Itera sobre las filas y columnas para recortar y guardar cada letra

    for fila in range(filas):
        for columna in range(columnas):
        # Coordenadas de inicio y fin del recorte, ajustando para quitar píxeles de los bordes
            x_inicio = columna * ancho_fragmento + n*6
            y_inicio = fila * alto_fragmento + n
            x_fin = x_inicio + ancho_fragmento - n*12
            y_fin = y_inicio + alto_fragmento - n*2
        
            # Recorta la imagen
            fragmento = imagen.crop((x_inicio, y_inicio, x_fin, y_fin))
        
            # Guarda el fragmento con un nombre único en la carpeta de salida
            nombre_archivo = f"{nombre[i]}.png"
            ruta_salida = os.path.join(paths_carpetas[j], nombre_archivo)
            fragmento.save(ruta_salida)
            print(f"{fila+1},{columna+1} --- {paths_carpetas[j]} -- {nombre[i]}.png ")
            j += 1
    i += 1 

    # Cierra la imagen después de procesarla
    imagen.close()

print("Proceso completado. Imágenes abiertas y procesadas.")

