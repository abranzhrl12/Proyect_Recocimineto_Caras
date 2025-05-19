import cv2

def test_camera_realtime(camera_index=0):
    """Abre la cámara en el índice especificado y muestra el video en tiempo real."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"No se pudo abrir la cámara en el índice: {camera_index}")
        return

    print(f"Mostrando video en tiempo real desde la cámara {camera_index}. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame.")
            break

        cv2.imshow(f'Cámara {camera_index} - Tiempo Real', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_realtime(camera_index=0)