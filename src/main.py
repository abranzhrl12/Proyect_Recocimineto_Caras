import cv2
from src.camera.video_stream import VideoStream
from src.data_processing.face_detector import FaceDetector
from src.face_recognition.recognition_service import RecognitionService
from src.utils.logger import get_logger
# from src.utils.config_loader import load_config # Si decides usar un archivo de configuración

logger = get_logger(__name__)

def main():
    """
    Función principal para ejecutar el reconocimiento facial en tiempo real.
    """
    logger.info("Iniciando la aplicación de reconocimiento facial.")

    # Cargar la configuración (si la usas)
    # config = load_config('config.yaml')

    # Inicializar los módulos
    video_stream = VideoStream(camera_index=0) # Puedes pasar la índice de la cámara si es diferente
    face_detector = FaceDetector(haar_cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognition_service = RecognitionService(model_path='data/trained_model.xml', names={0: 'Abraham'}) # Cargar el modelo y las etiquetas

    try:
        while True:
            frame = video_stream.read()
            if frame is None:
                logger.error("No se pudo leer el frame de la cámara.")
                break
            print("Frame leído correctamente.")

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detect_faces(gray_frame)
            print(f"Rostros detectados: {len(faces)}")

            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y + h, x:x + w]

                label, confidence = recognition_service.recognize(roi_gray)
                print(f"Label: {label}, Confidence: {confidence}")

                if confidence < 30: # Ajusta el umbral de confianza según sea necesario
                    name = label
                    confidence_text = f"  {100 - confidence:.2f}%"
                else:
                    name = "Desconocido"
                    confidence_text = ""

                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, confidence_text, (x + w + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('Reconocimiento Facial', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Ocurrió un error en la aplicación: {e}", exc_info=True)
    finally:
        video_stream.stop()
        cv2.destroyAllWindows()
        logger.info("Aplicación de reconocimiento facial finalizada.")

if __name__ == "__main__":
    main()