import argparse
import sys
import time

import cv2
import mediapipe as mp

from app.detector import MediapipeStreamObjectDetector
from app.utils import visualize


def run(model: str, camera_id: int, width: int, height: int, has_gui: bool) -> None:
    counter = 0
    fps = 0
    start_time = time.time()
    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # instantiate detector
    detector = MediapipeStreamObjectDetector(model)

    # main loop
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit("Could not read from webcam")

        # counter for detections
        counter += 1
        # preprocess image
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detector.detector.detect_async(mp_image, counter)
        current_frame = mp_image.numpy_view()
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        fps_text = f"FPS: {fps:.1f}"
        if has_gui:
            text_location = (left_margin, row_size)
            cv2.putText(
                current_frame,
                fps_text,
                text_location,
                cv2.FONT_HERSHEY_PLAIN,
                font_size,
                text_color,
                font_thickness,
            )

        if detector.result_list:
            # print(detector.result_list)
            if has_gui:
                vis_image = visualize(current_frame, detector.result_list[0])
                cv2.imshow("object_detector", vis_image)
            else:

                print(f"{fps_text} \t Detection: {detector.result_list[0].detections[0]}")
            detector.result_list.clear()

        elif has_gui:
            cv2.imshow("object_detector", current_frame)

        if cv2.waitKey(1) == 27:
            break

    detector.detector.close()
    cap.release()
    cv2.destroyAllWindows()


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
        "--has_gui",
        help="If we should visualize the predictions on the screen",
        required=False,
        type=bool,
        default=True,
    )
    args = parser.parse_args()

    run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight, args.has_gui)


if __name__ == '__main__':
    print("HELLO")
    main()