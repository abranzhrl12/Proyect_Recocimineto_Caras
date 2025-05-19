# import cv2
# import threading
# from ..utils.logger import get_logger

# logger = get_logger(__name__)

# class VideoStream:
#     def __init__(self, camera_index=0):
#         """
#         Inicializa el objeto VideoStream.

#         Args:
#             camera_index (int): El índice de la cámara a utilizar (por defecto es 0 para la cámara web principal).
#         """
#         self.camera_index = camera_index
#         self.cap = None
#         self.frame = None
#         self.running = False
#         self.thread = None

#     def start(self):
#         """
#         Inicia la captura de video en un hilo separado.
#         """
#         if not self.running:
#             self.running = True
#             self.thread = threading.Thread(target=self._update, args=())
#             self.thread.start()
#             logger.info(f"Captura de video iniciada desde la cámara {self.camera_index}.")

#     def _update(self):
#         """
#         Función interna para leer continuamente los frames de la cámara.
#         """
#         self.cap = cv2.VideoCapture(self.camera_index)
#         if not self.cap.isOpened():
#             logger.error(f"No se pudo abrir la cámara {self.camera_index}.")
#             self.running = False
#             return

#         while self.running:
#             ret, frame = self.cap.read()
#             if ret:
#                 self.frame = frame
#             else:
#                 logger.error(f"Error al leer el frame de la cámara {self.camera_index}.")
#                 self.running = False
#                 break
#         self.cap.release()
#         logger.info(f"Captura de video finalizada desde la cámara {self.camera_index}.")

#     def read(self):
#         """
#         Devuelve el último frame capturado.
#         """
#         return self.frame

#     def stop(self):
#         """
#         Detiene la captura de video.
#         """
#         self.running = False
#         if self.thread and self.thread.is_alive():
#             self.thread.join()
#         logger.info(f"Deteniendo la captura de video de la cámara {self.camera_index}.")

#     def is_running(self):
#         """
#         Devuelve True si la captura de video está en curso, False en caso contrario.
#         """
#         return self.running
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