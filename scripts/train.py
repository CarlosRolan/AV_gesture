"""
train.py — Entrena un clasificador MLP ligero sobre los landmarks normalizados.

Uso:
    python scripts/train.py

Produce:
    model/gesture_model.keras
    model/label_map.json      (mapeo índice → nombre de gesto)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_CSV    = Path(__file__).parent.parent / "data" / "landmarks.csv"
MODEL_DIR   = Path(__file__).parent.parent / "model"
MODEL_PATH  = MODEL_DIR / "gesture_model.keras"
LABEL_MAP   = MODEL_DIR / "label_map.json"

EPOCHS      = 60
BATCH_SIZE  = 32
DROPOUT     = 0.3
SEED        = 42


def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="gesture_mlp")


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    print(f"Clases: {list(le.classes_)}")
    print(f"Muestras: {len(X)}  |  Features: {X.shape[1]}  |  Clases: {num_classes}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    model = build_model(X.shape[1], num_classes)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modelo guardado: {MODEL_PATH}")

    label_map = {str(i): label for i, label in enumerate(le.classes_)}
    LABEL_MAP.write_text(json.dumps(label_map, indent=2))
    print(f"Label map guardado: {LABEL_MAP}")

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidación final — loss: {loss:.4f}  acc: {acc:.4f}")


if __name__ == "__main__":
    main()
