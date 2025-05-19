
import cv2
from ..utils.logger import get_logger

logger = get_logger(__name__)

class VideoStream:
    def __init__(self, camera_index=0):
        """
        Inicializa el objeto VideoStream.
        """
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"No se pudo abrir la cámara {self.camera_index}.")
        else:
            logger.info(f"Cámara abierta en el índice {self.camera_index}.")
        self.frame = None

    def read(self):
        """
        Lee un frame de la cámara.
        """
        if self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret:
                logger.error("No se pudo leer el frame de la cámara.")
                return None
            return self.frame
        return None

    def stop(self):
        """
        Detiene la captura de video.
        """
        if self.cap.isOpened():
            self.cap.release()
            logger.info(f"Captura de video finalizada de la cámara {self.camera_index}.")

    def is_running(self):
        """
        Devuelve True si la cámara está abierta.
        """
        return self.cap.isOpened()