"""
export_tfjs.py — Exporta el modelo Keras entrenado a formato TensorFlow.js.

Requisito previo:
    pip install tensorflowjs

Uso:
    python scripts/export_tfjs.py

Produce:
    model/tfjs_model/model.json
    model/tfjs_model/group1-shard1of1.bin   (u otros shards si supera 4 MB)
"""

import subprocess
import sys
from pathlib import Path

MODEL_KERAS = Path(__file__).parent.parent / "model" / "gesture_model.keras"
TFJS_DIR    = Path(__file__).parent.parent / "model" / "tfjs_model"


def main() -> None:
    if not MODEL_KERAS.exists():
        print(f"ERROR: No se encontró el modelo en {MODEL_KERAS}")
        print("Ejecuta primero: python scripts/train.py")
        sys.exit(1)

    TFJS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "tensorflowjs_converter",
        "--input_format", "keras",
        "--output_format", "tfjs_layers_model",
        "--quantization_dtype_map", "float16:*",   # reduce tamaño ~50%
        str(MODEL_KERAS),
        str(TFJS_DIR),
    ]

    print(f"Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("ERROR durante la conversión.")
        sys.exit(result.returncode)

    # Muestra tamaño total del modelo exportado
    total_bytes = sum(f.stat().st_size for f in TFJS_DIR.rglob("*") if f.is_file())
    print(f"\nModelo TF.js exportado en: {TFJS_DIR}")
    print(f"Tamaño total: {total_bytes / 1024:.1f} KB")

    if total_bytes > 20 * 1024 * 1024:
        print("ADVERTENCIA: El modelo supera 20 MB. Considera reducir la arquitectura.")


if __name__ == "__main__":
    main()
