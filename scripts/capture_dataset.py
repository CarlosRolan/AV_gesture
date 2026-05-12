"""
capture_dataset.py — Captura landmarks de mano con MediaPipe y los guarda en CSV.

Uso:
    python scripts/capture_dataset.py

Controles durante la captura:
    v  → label = v_sign       (✌️  dos dedos)
    u  → label = thumbs_up    (👍  pulgar arriba)
    d  → label = thumbs_down  (👎  pulgar abajo)
    q  → salir y guardar CSV
    SPACE → pausar/reanudar captura

El CSV resultante se guarda en data/landmarks.csv con columnas:
    x0,y0,z0, x1,y1,z1, ..., x20,y20,z20, label
"""

import csv
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "landmarks.csv"
NUM_LANDMARKS = 21
GESTURES = {
    ord("v"): "v_sign",
    ord("u"): "thumbs_up",
    ord("d"): "thumbs_down",
}
CAPTURE_COOLDOWN_MS = 100   # ms mínimos entre muestras para evitar duplicados
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Colores BGR
COLOR_ACTIVE   = (0, 255, 120)   # verde — capturando
COLOR_PAUSED   = (0, 165, 255)   # naranja — pausado
COLOR_IDLE     = (200, 200, 200) # gris — sin gesto activo
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_landmarks(hand_landmarks) -> list[float]:
    """
    Normaliza los 21 landmarks respecto a la muñeca (landmark 0).
    Devuelve una lista plana [x0,y0,z0, x1,y1,z1, ...] con coordenadas relativas
    escaladas por la distancia máxima al punto de la muñeca, para ser invariante
    a posición y escala de la mano en frame.
    """
    lm = hand_landmarks.landmark
    # Coordenadas crudas
    points = [(p.x, p.y, p.z) for p in lm]

    # Trasladar al origen (muñeca = landmark 0)
    ox, oy, oz = points[0]
    points = [(x - ox, y - oy, z - oz) for x, y, z in points]

    # Escalar por la distancia máxima para invarianza de escala
    max_dist = max((x**2 + y**2 + z**2) ** 0.5 for x, y, z in points[1:]) or 1.0
    points = [(x / max_dist, y / max_dist, z / max_dist) for x, y, z in points]

    return [coord for point in points for coord in point]


def draw_overlay(frame, current_label: str | None, sample_count: int,
                 paused: bool, counts_per_label: dict) -> None:
    h, w = frame.shape[:2]

    # Barra de estado superior
    bar_color = COLOR_PAUSED if paused else (COLOR_ACTIVE if current_label else COLOR_IDLE)
    cv2.rectangle(frame, (0, 0), (w, 50), bar_color, -1)

    status = "PAUSADO" if paused else (f"Capturando: {current_label}" if current_label else "Sin gesto activo")
    cv2.putText(frame, status, (12, 34), FONT, 0.9, (0, 0, 0), 2)

    # Contador total
    cv2.putText(frame, f"Total: {sample_count}", (w - 160, 34), FONT, 0.8, (0, 0, 0), 2)

    # Leyenda de teclas
    legend = [
        "v → v_sign",
        "u → thumbs_up",
        "d → thumbs_down",
        "SPACE → pausar",
        "q → guardar y salir",
    ]
    y_start = 80
    for i, line in enumerate(legend):
        cv2.putText(frame, line, (12, y_start + i * 22), FONT, 0.55, (255, 255, 255), 1)

    # Contadores por gesto
    y_start = 80
    for i, (label, count) in enumerate(counts_per_label.items()):
        text = f"{label}: {count}"
        cv2.putText(frame, text, (w - 200, y_start + i * 22), FONT, 0.55, (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = OUTPUT_CSV.exists()

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Verifica que está disponible.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_label: str | None = None
    paused = False
    sample_count = 0
    counts_per_label: dict[str, int] = {g: 0 for g in GESTURES.values()}
    last_capture_time = 0.0

    # Abre el CSV en modo append para poder retomar capturas previas
    csv_mode = "a" if file_exists else "w"
    csv_file = open(OUTPUT_CSV, csv_mode, newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        header = [f"{c}{i}" for i in range(NUM_LANDMARKS) for c in ("x", "y", "z")]
        header.append("label")
        writer.writerow(header)

    # Si el archivo ya existía, cuenta las muestras previas por label
    if file_exists:
        with open(OUTPUT_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label", "")
                if label in counts_per_label:
                    counts_per_label[label] += 1
                    sample_count += 1

    print(f"Guardando en: {OUTPUT_CSV}")
    print("Controles: v=v_sign  u=thumbs_up  d=thumbs_down  SPACE=pausa  q=salir")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error leyendo frame de cámara.")
                break

            frame = cv2.flip(frame, 1)  # espejo para UX más natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            hand_detected = results.multi_hand_landmarks is not None

            # Dibuja landmarks si se detecta mano
            if hand_detected:
                for hand_lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    # Captura muestra si hay label activo, no está pausado y
                    # ha pasado el cooldown
                    now = time.time() * 1000
                    if (
                        current_label
                        and not paused
                        and (now - last_capture_time) >= CAPTURE_COOLDOWN_MS
                    ):
                        row_data = normalize_landmarks(hand_lm)
                        row_data.append(current_label)
                        writer.writerow(row_data)
                        csv_file.flush()
                        sample_count += 1
                        counts_per_label[current_label] += 1
                        last_capture_time = now

            draw_overlay(frame, current_label if hand_detected else None,
                         sample_count, paused, counts_per_label)

            cv2.imshow("Gesture AV Controller — Dataset Capture", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
                current_label = None if paused else current_label
            elif key in GESTURES:
                current_label = None if current_label == GESTURES[key] else GESTURES[key]
                paused = False

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    print(f"\nCaptura finalizada. Muestras guardadas: {sample_count}")
    for label, count in counts_per_label.items():
        print(f"  {label}: {count}")
    print(f"CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
