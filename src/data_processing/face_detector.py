import cv2
from ..utils.logger import get_logger

logger = get_logger(__name__)

class FaceDetector:
    def __init__(self, haar_cascade_path):
        """
        Inicializa el detector de rostros utilizando un clasificador Haar cascade.

        Args:
            haar_cascade_path (str): Ruta al archivo XML del clasificador Haar cascade.
        """
        try:
            self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
            if self.face_cascade.empty():
                logger.error(f"No se pudo cargar el clasificador Haar cascade desde: {haar_cascade_path}")
                self.face_cascade = None
            else:
                logger.info(f"Clasificador Haar cascade cargado desde: {haar_cascade_path}")
        except Exception as e:
            logger.error(f"Error al cargar el clasificador Haar cascade desde {haar_cascade_path}: {e}", exc_info=True)
            self.face_cascade = None

    def detect_faces(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        """
        Detecta rostros en una imagen en escala de grises.

        Args:
            image (numpy.ndarray): Imagen en escala de grises donde buscar rostros.
            scaleFactor (float): Parámetro para el clasificador Haar.
            minNeighbors (int): Parámetro para el clasificador Haar.
            minSize (tuple): Tamaño mínimo que puede tener un rostro.

        Returns:
            list: Una lista de rectángulos (x, y, w, h) que delimitan los rostros detectados.
                   Devuelve una lista vacía si no se encuentran rostros o si el clasificador no se cargó.
        """
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(
                    image,
                    scaleFactor=scaleFactor,
                    minNeighbors=minNeighbors,
                    minSize=minSize
                )
                return faces
            except Exception as e:
                logger.error(f"Error durante la detección de rostros: {e}", exc_info=True)
                return []
        else:
            logger.warning("El clasificador Haar cascade no ha sido cargado. No se puede realizar la detección.")
            return []