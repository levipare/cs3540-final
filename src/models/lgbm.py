"""Utilities for training and evaluating LightGBM multiclass classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


@dataclass
class LGBMTrainingArtifacts:
    """Container for trained model and label encoding state."""

    model: Any
    label_encoder: LabelEncoder



def build_lightgbm_classifier(
    *,
    random_state: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    class_weight: str | dict[int, float] | None = None,
    n_jobs: int = -1,
    **kwargs: Any,
) -> Any:
    """Create a LightGBM classifier with sensible defaults for tabular multiclass data."""
    return lgb.LGBMClassifier(
        objective="multiclass",
        random_state=random_state,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        class_weight=class_weight,
        n_jobs=n_jobs,
        **kwargs,
    )


def train_lightgbm_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    early_stopping_rounds: int | None = 50,
    random_state: int = 42,
    class_weight: str | dict[int, float] | None = None,
    verbose_eval: bool = False,
    **model_kwargs: Any,
) -> LGBMTrainingArtifacts:
    """Train LightGBM on a multiclass problem with optional validation early stopping."""
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    model = build_lightgbm_classifier(
        random_state=random_state,
        class_weight=class_weight,
        **model_kwargs,
    )

    fit_kwargs: dict[str, Any] = {}
    if X_valid is not None and y_valid is not None:
        y_valid_encoded = label_encoder.transform(y_valid)
        fit_kwargs["eval_set"] = [(X_valid, y_valid_encoded)]

        if early_stopping_rounds is not None:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose_eval),
            ]

    model.fit(X_train, y_train_encoded, **fit_kwargs)

    return LGBMTrainingArtifacts(model=model, label_encoder=label_encoder)


def predict_labels(artifacts: LGBMTrainingArtifacts, X: pd.DataFrame) -> pd.Series:
    """Predict human-readable class labels for input features."""
    y_pred_encoded = artifacts.model.predict(X)
    y_pred = artifacts.label_encoder.inverse_transform(y_pred_encoded)
    return pd.Series(y_pred, index=X.index, name="prediction")


def classification_report_for_model(
    artifacts: LGBMTrainingArtifacts,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    digits: int = 3,
) -> str:
    """Generate a sklearn classification report for a trained LightGBM model."""
    y_pred = predict_labels(artifacts, X_test)
    return classification_report(y_test, y_pred, digits=digits)
