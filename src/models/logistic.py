"""Utilities for training and evaluating Logistic Regression multiclass classifiers."""

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class LogisticTrainingArtifacts:
    """Container for trained model, label encoding state, and feature scaler."""

    model: LogisticRegression
    label_encoder: LabelEncoder
    scaler: StandardScaler


def build_logistic_classifier(
    *,
    random_state: int = 42,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    class_weight: str | dict[int, float] | None = "balanced",
    **kwargs,
) -> LogisticRegression:
    """Create a Logistic Regression classifier for multiclass classification."""
    return LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        solver=solver,
        class_weight=class_weight,
        **kwargs,
    )


def train_logistic_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 42,
    class_weight: str | dict[int, float] | None = "balanced",
    **model_kwargs,
) -> LogisticTrainingArtifacts:
    """Train a Logistic Regression on a multiclass problem."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    model = build_logistic_classifier(
        random_state=random_state,
        class_weight=class_weight,
        **model_kwargs,
    )

    model.fit(X_train_scaled, y_train_encoded)

    return LogisticTrainingArtifacts(
        model=model, label_encoder=label_encoder, scaler=scaler
    )


def predict_labels(artifacts: LogisticTrainingArtifacts, X: pd.DataFrame) -> pd.Series:
    """Predict human-readable class labels for input features."""
    X_scaled = artifacts.scaler.transform(X)
    y_pred_encoded = artifacts.model.predict(X_scaled)
    y_pred = artifacts.label_encoder.inverse_transform(y_pred_encoded)
    return pd.Series(y_pred, index=X.index, name="prediction")


def classification_report_for_model(
    artifacts: LogisticTrainingArtifacts,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    digits: int = 3,
) -> str:
    """Generate a sklearn classification report for a trained Logistic model."""
    y_pred = predict_labels(artifacts, X_test)
    return classification_report(y_test, y_pred, digits=digits)
