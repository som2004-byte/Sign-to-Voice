import argparse
import os
import string
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def _ensure_directories():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def _build_label_map() -> Dict[str, str]:
    """
    Map keyboard keys to human-readable sign labels.

    You can customize this mapping to match the signs you want to collect.
    For example:
      "1": "hello"
      "2": "yes"
      "3": "no"
      "4": "thank_you"
      "5": "please"
    """
    return {
        "1": "hello",
        "2": "yes",
        "3": "no",
        "4": "thank_you",
        "5": "please",
    }


def _build_alnum_labels() -> List[str]:
    return [str(i) for i in range(10)] + list(string.ascii_lowercase)


def _draw_instructions(frame, label_map: Dict[str, str], saved_count: Dict[str, int]):
    """Overlay instructions and sample counts on the frame."""
    y = 30
    cv2.putText(
        frame,
        "Sign-to-Voice - Data Collection",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    y += 30
    cv2.putText(
        frame,
        "Press key for label, 's' to save current frame, 'q' to quit.",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    y += 25
    for key, label in label_map.items():
        count = saved_count.get(label, 0)
        text = f"{key} -> {label} (saved: {count})"
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        y += 20


def _extract_hand_landmarks(detection_result, image_width: int, image_height: int) -> Tuple[np.ndarray, bool]:
    """
    Extract landmarks for the first detected hand (single-hand prototype).

    Returns:
        (landmarks_vector, found)
        landmarks_vector shape: (42,) for 21 (x, y) pairs.
    """
    if not detection_result.hand_landmarks:
        return np.zeros(42, dtype=np.float32), False

    hand_landmarks = detection_result.hand_landmarks[0]
    coords: List[float] = []
    for lm in hand_landmarks:
        coords.append(lm.x)
        coords.append(lm.y)

    return np.array(coords, dtype=np.float32), True


def _normalize_landmarks_xy(landmarks_xy: np.ndarray) -> np.ndarray:
    """
    Make landmarks more invariant to position/scale:
    - translate so wrist (landmark 0) is at origin
    - scale by max distance from wrist (avoid division by 0)

    Input/Output shape: (42,) where pairs are (x, y) for 21 landmarks.
    """
    landmarks_xy = landmarks_xy.astype(np.float32)
    v = landmarks_xy.reshape(21, 2)
    wrist = v[0].copy()
    v = v - wrist
    scale = float(np.max(np.linalg.norm(v, axis=1)))
    if scale < 1e-6:
        scale = 1.0
    v = v / scale
    return v.reshape(-1).astype(np.float32)


def collect_sign_data(
    output_path: str = "data/sign_landmarks.npz",
    label_mode: str = "basic",
):
    """
    Run webcam, detect hands with MediaPipe, and save labeled landmark samples.

    - Press a numeric key (e.g. '1') to select a label.
    - Perform the corresponding sign in front of the camera.
    - Press 's' to save the current frame's hand landmarks for that label.
    - Press 'q' to quit and write all collected samples to an .npz file.
    """
    _ensure_directories()
    if label_mode == "basic":
        label_map = _build_label_map()
        labels_list = list(label_map.values())
        current_label_key = "1"  # Default selected label key
        current_label_idx = labels_list.index(label_map[current_label_key])
        use_cycle_mode = False
    elif label_mode == "alnum":
        label_map = {}
        labels_list = _build_alnum_labels()
        current_label_key = ""
        current_label_idx = 0
        use_cycle_mode = True
    else:
        raise ValueError("label_mode must be 'basic' or 'alnum'")

    # Prepare MediaPipe hand landmarker
    model_path = "models/hand_landmarker.task"
    if not os.path.exists(model_path):
        print(
            "Error: MediaPipe hand_landmarker.task model not found.\n"
            "Download it from:\n"
            "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n"
            "and place it at: models/hand_landmarker.task"
        )
        return

    base_options = python.BaseOptions(model_asset_path=model_path)
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

    samples: List[np.ndarray] = []
    labels: List[int] = []

    # Map label string to numeric id
    label_to_id = {label: idx for idx, label in enumerate(labels_list)}
    saved_count: Dict[str, int] = {label: 0 for label in labels_list}

    print("Starting data collection.")
    print("Controls:")
    if use_cycle_mode:
        print("  Use 'a' / 'd' to change label (previous/next).")
        print("  Press 's' to save a sample for the current label.")
    else:
        for key, label in label_map.items():
            print(f"  Press '{key}' to select label: {label}")
        print("  Press 's' to save a sample for the current label.")
    print("  Press 'q' to quit and save dataset.")

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

        if use_cycle_mode:
            current_label = labels_list[current_label_idx]
            cv2.putText(
                frame,
                "Use 'a'/'d' to change label, 's' to save, 'q' to quit.",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Saved for '{current_label}': {saved_count.get(current_label, 0)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            _draw_instructions(frame, label_map, saved_count)
            current_label = label_map.get(current_label_key, "")
        cv2.putText(
            frame,
            f"Current label: {current_label}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Sign-to-Voice - Collect Sign Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if use_cycle_mode:
            if key == ord("a"):
                current_label_idx = (current_label_idx - 1) % len(labels_list)
            elif key == ord("d"):
                current_label_idx = (current_label_idx + 1) % len(labels_list)
        else:
            # Switch current label
            key_char = chr(key) if key != 255 and chr(key) in string.printable else ""
            if key_char in label_map:
                current_label_key = key_char
                current_label_idx = labels_list.index(label_map[current_label_key])

        # Save sample for current label
        if key == ord("s"):
            landmarks_vector, found = _extract_hand_landmarks(
                detection_result, image_width, image_height
            )
            if not found:
                print("No hand detected; sample not saved.")
            else:
                label_str = labels_list[current_label_idx]
                label_id = label_to_id[label_str]
                samples.append(_normalize_landmarks_xy(landmarks_vector))
                labels.append(label_id)
                saved_count[label_str] += 1
                print(f"Saved sample for label '{label_str}'. Total: {saved_count[label_str]}")

    cap.release()
    cv2.destroyAllWindows()

    if not samples:
        print("No samples collected. Dataset not saved.")
        return

    X = np.stack(samples, axis=0)
    y = np.array(labels, dtype=np.int64)

    id_to_label = {int(i): str(lbl) for i, lbl in enumerate(labels_list)}
    np.savez(output_path, X=X, y=y, id_to_label=id_to_label)
    print(f"Saved dataset to {output_path}")
    print(f"Shape X: {X.shape}, y: {y.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sign_landmarks.npz")
    parser.add_argument(
        "--labels",
        default="basic",
        choices=["basic", "alnum"],
        help="basic=5 words, alnum=0-9+a-z",
    )
    args = parser.parse_args()
    collect_sign_data(output_path=args.output, label_mode=args.labels)

