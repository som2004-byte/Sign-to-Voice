import json
import os
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_dataset(path: str = "data/sign_landmarks.npz") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file '{path}' not found. "
            "Run 'python src/collect_sign_data.py' first to collect samples."
        )

    data = np.load(path, allow_pickle=True)
    X = data["X"].astype("float32")
    y = data["y"].astype("int64")

    id_to_label = None
    if "id_to_label" in data.files:
        raw = data["id_to_label"].item()
        id_to_label = {int(k): str(v) for k, v in dict(raw).items()}

    return {"X": X, "y": y, "id_to_label": id_to_label}


def build_model(input_dim: int, num_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(
    dataset_path: str = "data/sign_landmarks.npz",
    model_path: str = "models/sign_classifier.h5",
    label_map_path: str = "models/sign_label_map.json",
    epochs: int = 30,
    batch_size: int = 32,
):
    os.makedirs("models", exist_ok=True)

    data = load_dataset(dataset_path)
    X, y, id_to_label = data["X"], data["y"], data["id_to_label"]

    num_samples = X.shape[0]
    num_classes = int(y.max()) + 1

    print(f"Loaded dataset from {dataset_path}")
    print(f"Samples: {num_samples}, Feature dim: {X.shape[1]}, Classes: {num_classes}")

    # Shuffle dataset
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Simple train/validation split
    split = int(0.8 * num_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_model(input_dim=X.shape[1], num_classes=num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=_compute_class_weight(y_train),
        verbose=1,
    )

    val_acc = history.history.get("val_accuracy", [0])[-1]
    print(f"Final validation accuracy: {val_acc:.3f}")

    model.save(model_path)
    print(f"Saved trained model to {model_path}")

    if id_to_label is None:
        # Fallback: store ids only
        id_to_label = {int(i): str(i) for i in range(num_classes)}

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(id_to_label, f, indent=2, ensure_ascii=False)
    print(f"Saved label map to {label_map_path}")


def _compute_class_weight(y_train: np.ndarray) -> Dict[int, float]:
    # Inverse-frequency weighting to prevent collapse to one label.
    y_list = y_train.astype("int64").tolist()
    counts: Dict[int, int] = {}
    for v in y_list:
        counts[v] = counts.get(v, 0) + 1
    total = float(len(y_list))
    num_classes = float(len(counts))
    return {cls: total / (num_classes * float(cnt)) for cls, cnt in counts.items()}


if __name__ == "__main__":
    train()

