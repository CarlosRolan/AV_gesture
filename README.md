# Gesture AV Controller

Real-time hand gesture recognition mapped to AV control actions — built as a portfolio project by an AV Technician at the European Patent Office (Munich).

The web demo runs **100% client-side**: MediaPipe Hands extracts 21 hand landmarks in the browser, a lightweight TF.js MLP classifies the gesture, and the result maps to AV actions (volume control, etc.).

---

## Gestures (Phase 1)

| Gesture | Label | AV Action |
|---|---|---|
| ✌️ V sign | `v_sign` | Activate volume control mode |
| 👍 Thumbs up | `thumbs_up` | Volume up |
| 👎 Thumbs down | `thumbs_down` | Volume down |

---

## Project Structure

```
gesture-av-controller/
├── data/                  # captured landmarks CSV (gitignored)
├── model/
│   ├── gesture_model.keras
│   ├── label_map.json
│   └── tfjs_model/        # TF.js exported model
├── scripts/
│   ├── capture_dataset.py # webcam capture → CSV
│   ├── train.py           # Keras MLP training
│   └── export_tfjs.py     # export to TF.js
├── web/                   # browser demo
│   ├── index.html
│   ├── main.js
│   └── style.css
├── gradio_demo.py         # HuggingFace Spaces demo
└── requirements.txt
```

---

## Quickstart

### 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2 — Capture training data

```bash
python scripts/capture_dataset.py
```

Controls during capture:

| Key | Label |
|-----|-------|
| `v` | v_sign |
| `u` | thumbs_up |
| `d` | thumbs_down |
| `SPACE` | pause / resume |
| `q` | quit & save CSV |

Aim for **≥ 200 samples per gesture** with varied hand positions.

### 3 — Train the model

```bash
python scripts/train.py
```

Produces `model/gesture_model.keras` and `model/label_map.json`.

### 4 — Export to TF.js

```bash
python scripts/export_tfjs.py
```

Produces `model/tfjs_model/` (target < 20 MB).

### 5 — Run the web demo

Serve the repo root (required for ES module / fetch paths):

```bash
python -m http.server 8080
# open http://localhost:8080/web/
```

### 6 — Gradio demo (optional)

```bash
python gradio_demo.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Landmark extraction | MediaPipe Hands (Python + JS) |
| Classifier | TensorFlow / Keras MLP (63 inputs → 128 → 64 → 3) |
| Browser inference | TensorFlow.js |
| Alternative demo | Gradio (HuggingFace Spaces) |
| Deploy | GitHub Pages / Vercel |

---

## Model Architecture

Input: 63 features (21 landmarks × [x, y, z], origin-normalised and scale-normalised)

```
Dense(128, relu) → BatchNorm → Dropout(0.3)
Dense(64,  relu) → BatchNorm → Dropout(0.3)
Dense(3, softmax)
```

Model size after float16 quantisation: **< 1 MB**.

---

## Background

I work as an AV Technician controlling video switchers, Allen & Heath SQ-6 audio mixers, and Crestron systems. This project explores whether hand gestures can complement physical AV control in meeting rooms — a natural fit for touchless operation during presentations.
