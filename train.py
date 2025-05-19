import os
from PIL import Image
import numpy as np
import cv2

def train_model(data_dir):
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
                img = Image.open(image_path).convert('L') # Convertir a escala de grises
                img_np = np.array(img, 'uint8')
                faces.append(img_np)
                labels.append(label_counter)
            except Exception as e:
                print(f"Error al cargar la imagen: {image_path} - {e}")

        label_counter += 1

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    return recognizer, label_map

if __name__ == '__main__':
    # Directorio donde guardaste las imágenes de entrenamiento
    data_directory = './data/training_images'
    model, names = train_model(data_directory)
    model.save('./data/trained_model.xml') # Guardar el modelo entrenado
    print("Modelo entrenado exitosamente.")
    print("Mapeo de etiquetas:", names)
