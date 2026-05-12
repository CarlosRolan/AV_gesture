"""
gradio_demo.py — Demo interactiva para HuggingFace Spaces.

Permite subir una imagen de mano y obtener la predicción del gesto.
Para lanzar localmente:
    pip install gradio mediapipe tensorflow
    python gradio_demo.py
"""

import json
from pathlib import Path

import gradio as gr
import mediapipe as mp
import numpy as np
import tensorflow as tf

MODEL_PATH = Path("model/gesture_model.keras")
LABEL_MAP  = Path("model/label_map.json")

GESTURE_INFO = {
    "v_sign":      {"emoji": "✌️",  "av": "Activates Volume Control Mode"},
    "thumbs_up":   {"emoji": "👍",  "av": "Volume Up"},
    "thumbs_down": {"emoji": "👎",  "av": "Volume Down"},
}

# ── Load model ────────────────────────────────────────────────────────────
model     = tf.keras.models.load_model(MODEL_PATH)
label_map = json.loads(LABEL_MAP.read_text())   # {"0": "thumbs_down", ...}

mp_hands  = mp.solutions.hands


def normalize_landmarks(landmarks) -> list[float]:
    points = [(lm.x, lm.y, lm.z) for lm in landmarks]
    ox, oy, oz = points[0]
    points = [(x - ox, y - oy, z - oz) for x, y, z in points]
    max_dist = max((x**2 + y**2 + z**2) ** 0.5 for x, y, z in points[1:]) or 1.0
    points = [(x / max_dist, y / max_dist, z / max_dist) for x, y, z in points]
    return [c for p in points for c in p]


def predict(image: np.ndarray):
    """Receives an RGB numpy array from Gradio, returns prediction text."""
    if image is None:
        return "No image provided.", "", ""

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        results = hands.process(image)

    if not results.multi_hand_landmarks:
        return "No hand detected in the image.", "", ""

    features = normalize_landmarks(results.multi_hand_landmarks[0].landmark)
    tensor   = np.array([features], dtype=np.float32)
    preds    = model.predict(tensor, verbose=0)[0]

    idx       = int(np.argmax(preds))
    confidence = float(preds[idx])
    label     = label_map.get(str(idx), "unknown")
    info      = GESTURE_INFO.get(label, {"emoji": "❓", "av": "Unknown"})

    result = (
        f"{info['emoji']}  **{label}**  ({confidence*100:.1f}% confidence)"
    )
    av_action = f"AV Action → {info['av']}"
    breakdown = "  |  ".join(
        f"{label_map.get(str(i), '?')}: {p*100:.1f}%"
        for i, p in enumerate(preds)
    )
    return result, av_action, breakdown


# ── Interface ─────────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Hand gesture photo", type="numpy"),
    outputs=[
        gr.Markdown(label="Detected gesture"),
        gr.Textbox(label="AV action"),
        gr.Textbox(label="Class probabilities"),
    ],
    title="Gesture AV Controller",
    description=(
        "Upload a photo showing one of the three AV control gestures: "
        "✌️ V-sign · 👍 Thumbs up · 👎 Thumbs down. "
        "Inference runs server-side using the trained Keras model."
    ),
    examples=[],
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
