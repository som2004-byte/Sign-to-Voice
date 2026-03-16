import json
import os
import time
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from gtts import gTTS
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def _load_label_map(path: str = "models/sign_label_map.json") -> Dict[int, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Label map '{path}' not found. Train the model first with "
            "'python src/train_sign_model.py'."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # JSON keys are strings; convert to int
    return {int(k): v for k, v in data.items()}


def _load_model(path: str = "models/sign_classifier.h5") -> tf.keras.Model:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained model '{path}' not found. Train the model first with "
            "'python src/train_sign_model.py'."
        )
    return tf.keras.models.load_model(path)


def _extract_hand_landmarks(detection_result, image_width: int, image_height: int) -> Tuple[np.ndarray, bool]:
    if not detection_result.hand_landmarks:
        return np.zeros(42, dtype=np.float32), False

    hand_landmarks = detection_result.hand_landmarks[0]
    coords: List[float] = []
    for lm in hand_landmarks:
        coords.append(lm.x)
        coords.append(lm.y)

    return np.array(coords, dtype=np.float32), True


def _normalize_landmarks_xy(landmarks_xy: np.ndarray) -> np.ndarray:
    landmarks_xy = landmarks_xy.astype(np.float32)
    v = landmarks_xy.reshape(21, 2)
    wrist = v[0].copy()
    v = v - wrist
    scale = float(np.max(np.linalg.norm(v, axis=1)))
    if scale < 1e-6:
        scale = 1.0
    v = v / scale
    return v.reshape(-1).astype(np.float32)


def _init_tts_engine() -> pyttsx3.Engine:
    engine = pyttsx3.init()
    # Slightly slower speech for clarity
    rate = engine.getProperty("rate")
    engine.setProperty("rate", int(rate * 0.9))
    return engine


def _speak(text: str, engine: pyttsx3.Engine, prefer: str = "pyttsx3"):
    """
    Try pyttsx3 first (offline). If it fails, fallback to gTTS + os.startfile (Windows).
    """
    text = text.strip()
    if not text:
        return

    if prefer == "pyttsx3":
        try:
            engine.say(text)
            engine.runAndWait()
            return
        except Exception:
            pass

    try:
        os.makedirs("tmp", exist_ok=True)
        mp3_path = os.path.join("tmp", "tts.mp3")
        gTTS(text=text, lang="en").save(mp3_path)
        os.startfile(mp3_path)
    except Exception as e:
        print(f"TTS error: {e}")


def run_sign_to_voice():
    """
    Real-time sign-to-voice pipeline:
      - Captures webcam frames.
      - Uses MediaPipe to get hand landmarks.
      - Uses trained classifier to predict the sign.
      - Speaks out the predicted sign when it is stable.
    """
    model_path = "models/sign_classifier.h5"
    label_map_path = "models/sign_label_map.json"
    hand_model_path = "models/hand_landmarker.task"

    if not os.path.exists(hand_model_path):
        print(
            "Error: MediaPipe hand_landmarker.task model not found.\n"
            "Download it from:\n"
            "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n"
            "and place it at: models/hand_landmarker.task"
        )
        return

    label_map = _load_label_map(label_map_path)
    model = _load_model(model_path)
    tts_engine = _init_tts_engine()

    base_options = python.BaseOptions(model_asset_path=hand_model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # For stability: require same prediction N times before speaking
    last_label = None
    stable_count = 0
    required_stable_frames = 5
    min_confidence = 0.45

    # Speak gating (prevents "only once" and prevents spamming)
    last_spoken_label = None
    last_spoken_time = 0.0
    cooldown_seconds = 1.0

    print("Running real-time sign-to-voice.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = hand_landmarker.detect(mp_image)

        image_height, image_width, _ = frame.shape

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                prev_point = None
                for lm in hand_landmarks:
                    x_px = int(lm.x * image_width)
                    y_px = int(lm.y * image_height)
                    cv2.circle(frame, (x_px, y_px), 3, (0, 255, 0), -1)
                    if prev_point is not None:
                        cv2.line(frame, prev_point, (x_px, y_px), (0, 0, 255), 1)
                    prev_point = (x_px, y_px)

        landmarks_vector, found = _extract_hand_landmarks(
            detection_result, image_width, image_height
        )

        predicted_label_str = ""
        predicted_conf = 0.0
        if found:
            input_vec = _normalize_landmarks_xy(landmarks_vector).reshape(1, -1)
            probs = model.predict(input_vec, verbose=0)[0]
            pred_id = int(np.argmax(probs))
            predicted_conf = float(probs[pred_id])
            predicted_label_str = label_map.get(pred_id, "")

            if predicted_label_str and predicted_conf >= min_confidence:
                if predicted_label_str == last_label:
                    stable_count += 1
                else:
                    last_label = predicted_label_str
                    stable_count = 1

                now = time.time()
                if (
                    stable_count >= required_stable_frames
                    and predicted_label_str != last_spoken_label
                    and (now - last_spoken_time) >= cooldown_seconds
                ):
                    print(f"Speaking: {predicted_label_str} ({predicted_conf:.2f})")
                    _speak(predicted_label_str.replace("_", " "), tts_engine, prefer="pyttsx3")
                    last_spoken_label = predicted_label_str
                    last_spoken_time = now
        else:
            last_label = None
            stable_count = 0

        if predicted_label_str:
            status_text = f"Predicted: {predicted_label_str} ({predicted_conf:.2f})"
        else:
            status_text = "Predicted: ---"
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Sign-to-Voice - Real-time", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_sign_to_voice()

