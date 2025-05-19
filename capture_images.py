import cv2
import os

def capture_images(name, num_samples=50):
    """
    Captura imágenes de un rostro y las guarda en la carpeta correspondiente.

    Args:
        name (str): El nombre de la persona a la que se le tomarán las fotos (será el nombre de la carpeta).
        num_samples (int): El número de imágenes a capturar.
    """
    data_dir = './data/training_images'
    person_dir = os.path.join(data_dir, name)

    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)  # Intenta con 0, si no funciona prueba con 1, 2, etc.
    count = 0

    print(f"Capturando {num_samples} imágenes para {name}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame de la cámara.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = gray[y:y + h, x:x + w]
            save_path = os.path.join(person_dir, f'image_{count}.jpg')
            cv2.imwrite(save_path, face_roi)
            count += 1
            print(f"Imagen guardada: {save_path}")

        cv2.imshow('Captura de Imágenes', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Captura de imágenes completada.")

if __name__ == '__main__':
    person_name = input("Introduce tu nombre para la captura de imágenes: ")
    num_images = int(input("Introduce el número de imágenes a capturar: "))
    capture_images(person_name, num_images)