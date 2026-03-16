import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def run_webcam_hand_tracking():
    # Path to the MediaPipe Tasks hand landmark model.
    # Download from:
    # https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    # and place it at: models/hand_landmarker.task
    model_path = "models/hand_landmarker.task"

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Higher resolution can help recognition but may be heavier on CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Flip for natural selfie-view
        frame = cv2.flip(frame, 1)

        # Convert BGR (OpenCV) to RGB and wrap into MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run hand landmark detection (IMAGE mode = frame-by-frame)
        detection_result = hand_landmarker.detect(mp_image)

        # Draw landmarks (simple circles and lines) on the original BGR frame
        image_height, image_width, _ = frame.shape

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw connections by approximating with simple lines between consecutive landmarks
                prev_point = None
                for landmark in hand_landmarks:
                    x_px = int(landmark.x * image_width)
                    y_px = int(landmark.y * image_height)

                    cv2.circle(frame, (x_px, y_px), 3, (0, 255, 0), -1)

                    if prev_point is not None:
                        cv2.line(frame, prev_point, (x_px, y_px), (0, 0, 255), 1)

                    prev_point = (x_px, y_px)

        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Sign-to-Voice - Phase 1 (Hand Tracking)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_hand_tracking()

