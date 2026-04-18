"""Utilities for training and evaluating Keras 1D CNN multiclass classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import keras
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class CNNTrainingArtifacts:
    """Container for trained model and preprocessing state."""

    model: keras.Model
    label_encoder: LabelEncoder
    scaler: StandardScaler
    history: dict[str, list[float]]


def _resolve_class_weight(
    class_weight: str | dict[int, float] | None,
    y_encoded: np.ndarray,
) -> dict[int, float] | None:
    """Translate class-weight configuration into a Keras-compatible mapping."""
    if class_weight is None:
        return None

    if isinstance(class_weight, dict):
        return class_weight

    if class_weight != "balanced":
        raise ValueError("class_weight must be None, 'balanced', or a dict[int, float].")

    classes, counts = np.unique(y_encoded, return_counts=True)
    n_samples = y_encoded.shape[0]
    n_classes = classes.shape[0]
    return {
        int(class_id): float(n_samples / (n_classes * class_count))
        for class_id, class_count in zip(classes, counts, strict=True)
    }


def _reshape_for_cnn(X: np.ndarray) -> np.ndarray:
    """Add a channel dimension expected by Conv1D: (n_samples, n_features, 1)."""
    return np.expand_dims(X, axis=-1)

METRICS = [
      keras.metrics.SparseCategoricalCrossentropy(name='cross entropy'),  # same as model's loss
      keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
]

def build_cnn_classifier(
    *,
    input_dim: int,
    num_classes: int,
    random_state: int = 42,
    conv_filters: tuple[int, ...] = (64, 128),
    kernel_size: int = 3,
    dense_units: int = 128,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
    **compile_kwargs: Any,
) -> keras.Model:
    """Create a compiled 1D CNN classifier for multiclass tabular data."""
    keras.utils.set_random_seed(random_state)

    layers: list[keras.layers.Layer] = [keras.layers.Input(shape=(input_dim, 1))]

    for i, filters in enumerate(conv_filters):
        layers.append(
            keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
            )
        )
        layers.append(keras.layers.BatchNormalization())
        if i < len(conv_filters) - 1:
            layers.append(keras.layers.MaxPooling1D(pool_size=2))
        if dropout_rate > 0:
            layers.append(keras.layers.Dropout(dropout_rate))

    layers.append(keras.layers.GlobalAveragePooling1D())
    if dense_units > 0:
        layers.append(keras.layers.Dense(dense_units, activation="relu"))
        if dropout_rate > 0:
            layers.append(keras.layers.Dropout(dropout_rate))

    layers.append(keras.layers.Dense(num_classes, activation="softmax"))

    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=METRICS,
        **compile_kwargs,
    )
    return model


def train_cnn_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    early_stopping_patience: int | None = 5,
    random_state: int = 42,
    class_weight: str | dict[int, float] | None = None,
    verbose: int = 0,
    **model_kwargs: Any,
) -> CNNTrainingArtifacts:
    """Train a 1D CNN classifier with optional validation-based early stopping."""
    keras.utils.set_random_seed(random_state)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_train_cnn = _reshape_for_cnn(X_train_scaled)

    model = build_cnn_classifier(
        input_dim=X_train_scaled.shape[1],
        num_classes=len(label_encoder.classes_),
        random_state=random_state,
        **model_kwargs,
    )

    fit_kwargs: dict[str, Any] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "verbose": verbose,
        "class_weight": _resolve_class_weight(class_weight, y_train_encoded),
    }

    callbacks: list[keras.callbacks.Callback] = []
    if X_valid is not None and y_valid is not None:
        y_valid_encoded = label_encoder.transform(y_valid)
        X_valid_scaled = scaler.transform(X_valid).astype(np.float32)
        X_valid_cnn = _reshape_for_cnn(X_valid_scaled)
        fit_kwargs["validation_data"] = (X_valid_cnn, y_valid_encoded)

        if early_stopping_patience is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                )
            )

    if callbacks:
        fit_kwargs["callbacks"] = callbacks

    history = model.fit(X_train_cnn, y_train_encoded, **fit_kwargs)

    return CNNTrainingArtifacts(
        model=model,
        label_encoder=label_encoder,
        scaler=scaler,
        history=history.history,
    )


def predict_labels(artifacts: CNNTrainingArtifacts, X: pd.DataFrame) -> pd.Series:
    """Predict human-readable class labels for input features."""
    X_scaled = artifacts.scaler.transform(X).astype(np.float32)
    X_cnn = _reshape_for_cnn(X_scaled)
    y_pred_proba = artifacts.model.predict(X_cnn, verbose=0)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    y_pred = artifacts.label_encoder.inverse_transform(y_pred_encoded)
    return pd.Series(y_pred, index=X.index, name="prediction")


def classification_report_for_model(
    artifacts: CNNTrainingArtifacts,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    digits: int = 3,
) -> str:
    """Generate a sklearn classification report for a trained 1D CNN model."""
    y_pred = predict_labels(artifacts, X_test)
    return classification_report(y_test, y_pred, digits=digits)
