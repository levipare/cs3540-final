"""Utilities for training and evaluating Random Forest multiclass classifiers."""

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


@dataclass
class RFTrainingArtifacts:
    """Container for trained model and label encoding state."""

    model: RandomForestClassifier
    label_encoder: LabelEncoder


def build_rf_classifier(
    *,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int | None = None,
    **kwargs,
) -> RandomForestClassifier:
    """Create a Random Forest classifier for multiclass classification."""
    return RandomForestClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        **kwargs,
    )


def train_rf_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int | None = None,
    **model_kwargs,
) -> RFTrainingArtifacts:
    """Train a Random Forest on a multiclass problem."""
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    model = build_rf_classifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        **model_kwargs,
    )

    model.fit(X_train, y_train_encoded)

    return RFTrainingArtifacts(model=model, label_encoder=label_encoder)


def predict_labels(artifacts: RFTrainingArtifacts, X: pd.DataFrame) -> pd.Series:
    """Predict human-readable class labels for input features."""
    y_pred_encoded = artifacts.model.predict(X)
    y_pred = artifacts.label_encoder.inverse_transform(y_pred_encoded)
    return pd.Series(y_pred, index=X.index, name="prediction")


def classification_report_for_model(
    artifacts: RFTrainingArtifacts,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    digits: int = 3,
) -> str:
    """Generate a sklearn classification report for a trained Random Forest model."""
    y_pred = predict_labels(artifacts, X_test)
    return classification_report(y_test, y_pred, digits=digits)
