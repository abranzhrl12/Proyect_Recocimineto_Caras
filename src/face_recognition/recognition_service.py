import cv2
import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

class RecognitionService:
    def __init__(self, model_path, names):
        """
        Inicializa el servicio de reconocimiento facial.

        Args:
            model_path (str): Ruta al archivo del modelo entrenado (e.g., 'trained_model.xml').
            names (dict): Diccionario que mapea las etiquetas numéricas a los nombres de las personas.
                           Ejemplo: {0: 'Tu Nombre', 1: 'Otra Persona'}
        """
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            self.recognizer.read(model_path)
            self.names = names
            logger.info(f"Modelo de reconocimiento cargado desde: {model_path}")
            logger.info(f"Mapeo de nombres cargado: {names}")
        except Exception as e:
            logger.error(f"Error al cargar el modelo de reconocimiento desde {model_path}: {e}", exc_info=True)
            self.recognizer = None
            self.names = {}

    def recognize(self, face_image):
        """
        Realiza el reconocimiento facial en una imagen de rostro.

        Args:
            face_image (numpy.ndarray): Imagen del rostro en escala de grises.

        Returns:
            tuple: Una tupla que contiene la etiqueta predicha (nombre) y el nivel de confianza.
                   Si no se puede realizar la predicción, devuelve ("Desconocido", float('inf')).
        """
        if self.recognizer is not None:
            try:
                label, confidence = self.recognizer.predict(face_image)
                name = self.names.get(label, "Desconocido")
                return name, confidence
            except Exception as e:
                logger.error(f"Error durante la predicción: {e}", exc_info=True)
                return "Desconocido", float('inf')
        else:
            logger.warning("El modelo de reconocimiento no ha sido cargado.")
            return "Desconocido", float('inf')