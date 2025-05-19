import os
from PIL import Image
import numpy as np

def load_training_data(data_dir):
    """
    Carga las imágenes de entrenamiento desde el directorio especificado.

    Args:
        data_dir (str): Ruta al directorio 'training_images'.

    Returns:
        tuple: Una tupla que contiene dos listas:
               - faces (list): Lista de arrays numpy representando los rostros en escala de grises.
               - labels (list): Lista de enteros representando las etiquetas de cada rostro.
               - label_map (dict): Diccionario que mapea las etiquetas numéricas a los nombres de las personas.
    """
    faces = []
    labels = []
    label_map = {}
    label_counter = 0

    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[label_counter] = person_name

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            try:
                img = Image.open(image_path).convert('L')  # Convertir a escala de grises
                img_np = np.array(img, 'uint8')
                faces.append(img_np)
                labels.append(label_counter)
            except Exception as e:
                print(f"Error al cargar la imagen: {image_path} - {e}")

        label_counter += 1

    return faces, np.array(labels), label_map