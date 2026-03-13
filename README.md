# Sign-to-Voice

An AI-powered accessibility project to convert sign language gestures to spoken voice in multiple languages.

## Getting Started (Phase 1: Hand Tracking Prototype)

### 1. Create and activate virtual environment (Windows PowerShell)

```bash
cd Sign-to-Voice
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Phase 1 prototype

```bash
python src/phase1_webcam_mediapipe.py
```

You should see:

- A webcam window.
- Your hand(s) with green keypoints and red connections drawn.
- A small message: “Press 'q' to quit”.

This confirms:

- Webcam capture works.
- MediaPipe hands is correctly installed.
- You are ready to move to gesture recognition and, later, sign-to-voice.
