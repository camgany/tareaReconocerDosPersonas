import argparse
import cv2
import mediapipe as mp
import time

from app.detector import MediapipeStreamObjectDetector

def show_camera_and_detection(model, camera_id, width, height, has_gui, threshold):
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Error al abrir la cámara.")
        return

    # instantiate detector
    print("Calling object detector...")
    detector = MediapipeStreamObjectDetector(model, threshold)

    if has_gui:
        cv2.startWindowThread()
        cv2.namedWindow("Cámara", cv2.WINDOW_NORMAL)
        cv2.imshow("Cámara", cv2.cvtColor(cv2.imread("black_image.jpg"), cv2.COLOR_BGR2RGB))
        time.sleep(0.1)  # Agregar una pequeña pausa

    num_persons_prev = 0  # Número de personas en el fotograma anterior

    while True:
        ret, image = cap.read()

        if not ret:
            print("Error al capturar el frame.")
            break

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # Proporcionar la marca de tiempo en milisegundos
        timestamp_ms = int(time.time() * 1000)
        detector.detector.detect_async(mp_image, timestamp_ms)

        current_frame = mp_image.numpy_view()
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        if has_gui:
            cv2.imshow("Cámara", current_frame)

        if detector.result_list:
            num_persons_current = 0
            for i, detection in enumerate(detector.result_list):
                # Filtrar detecciones para mantener solo las personas
                person_detections = [
                    (label, score)
                    for label, score in zip(detection.labels, detection.confidences)
                    if "person" in label.lower()
                ]
                num_persons_current += len(person_detections)

                if person_detections:
                    detections_dict = {
                        f"Person_{i+1}_{label}": score
                        for label, score in person_detections
                    }
                   # print(f"Detections for Person {i+1}: {detections_dict}")

            if num_persons_current != num_persons_prev:
                print(f"Se encontraron {num_persons_current} persona(s).")
                num_persons_prev = num_persons_current

            detector.result_list.clear()

        key = cv2.waitKey(1)
        if has_gui and key == 27:  # 27 es el código ASCII para la tecla 'Esc'
            break

    if has_gui:
        time.sleep(0.1)  # Agregar una pequeña pausa
        cv2.destroyAllWindows()

    detector.detector.close()
    cap.release()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        help="Path of the object detection model.",
        required=False,
        default="efficientdet.tflite",
    )
    parser.add_argument(
        "--cameraId", help="Id of camera.", required=False, type=int, default=0
    )
    parser.add_argument(
        "--frameWidth",
        help="Width of frame to capture from camera.",
        required=False,
        type=int,
        default=1280,
    )
    parser.add_argument(
        "--frameHeight",
        help="Height of frame to capture from camera.",
        required=False,
        type=int,
        default=720,
    )
    parser.add_argument(
        "--threshold",
        help="Score threshold for detections.",
        required=False,
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--has_gui",
        help="If we should visualize the predictions on the screen",
        required=False,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()
    print("Starting...")

    show_camera_and_detection(
        args.model,
        int(args.cameraId),
        args.frameWidth,
        args.frameHeight,
        args.has_gui,
        args.threshold,
    )

if __name__ == "__main__":
    main()
