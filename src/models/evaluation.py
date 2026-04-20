"""Utilities for computing and visualizing classification metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

NormalizeMode = Literal["true", "pred", "all"]


@dataclass
class ModelEvaluationResult:
    """Container for computed metrics and confusion matrix values."""

    model_name: str
    aggregate_metrics: dict[str, float]
    classification_report_text: str
    classification_report_df: pd.DataFrame
    confusion_matrix_df: pd.DataFrame


def _to_series(values: pd.Series | Sequence) -> pd.Series:
    """Convert values to a Series while preserving an existing Series index."""
    if isinstance(values, pd.Series):
        return values
    return pd.Series(values)


def _resolve_labels(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: Sequence | None,
) -> list:
    """Resolve deterministic class label order for reports and confusion matrix."""
    if labels is not None:
        return list(labels)
    combined = pd.concat((y_true, y_pred), ignore_index=True)
    return list(pd.Index(combined).drop_duplicates())


def evaluate_model_predictions(
    model_name: str,
    y_true: pd.Series | Sequence,
    y_pred: pd.Series | Sequence,
    *,
    labels: Sequence | None = None,
    digits: int = 3,
    zero_division: int | float | str = 0,
) -> ModelEvaluationResult:
    """Compute metrics, a classification report, and confusion matrix."""
    y_true_series = _to_series(y_true)
    y_pred_series = _to_series(y_pred)

    if y_true_series.shape[0] != y_pred_series.shape[0]:
        raise ValueError(
            "y_true and y_pred must have the same number of rows. "
            f"Got {y_true_series.shape[0]} and {y_pred_series.shape[0]}."
        )

    label_order = _resolve_labels(y_true_series, y_pred_series, labels)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_series,
        y_pred_series,
        labels=label_order,
        average="macro",
        zero_division=zero_division,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true_series,
            y_pred_series,
            labels=label_order,
            average="weighted",
            zero_division=zero_division,
        )
    )

    aggregate_metrics = {
        "accuracy": float(accuracy_score(y_true_series, y_pred_series)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }

    report_text = classification_report(
        y_true_series,
        y_pred_series,
        labels=label_order,
        digits=digits,
        zero_division=zero_division,
    )

    report_dict = classification_report(
        y_true_series,
        y_pred_series,
        labels=label_order,
        digits=digits,
        zero_division=zero_division,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    cm = confusion_matrix(y_true_series, y_pred_series, labels=label_order)
    cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)

    return ModelEvaluationResult(
        model_name=model_name,
        aggregate_metrics=aggregate_metrics,
        classification_report_text=report_text,
        classification_report_df=report_df,
        confusion_matrix_df=cm_df,
    )


def print_evaluation_summary(result: ModelEvaluationResult, *, decimals: int = 4) -> None:
    """Print metrics and per-class report for a model."""
    print(f"== {result.model_name} ==")
    for metric_name, metric_value in result.aggregate_metrics.items():
        print(f"{metric_name:>18}: {metric_value:.{decimals}f}")
    print()
    print(result.classification_report_text)


def save_classification_report(
    result: ModelEvaluationResult,
    output_path: str | Path,
) -> Path:
    """Write a text classification report to disk."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(result.classification_report_text, encoding="utf-8")
    return output_file


def _normalize_confusion_matrix(cm: np.ndarray, normalize: NormalizeMode) -> np.ndarray:
    """Normalize a confusion matrix across rows, columns, or globally."""
    cm_float = cm.astype(float)
    if normalize == "true":
        row_totals = cm_float.sum(axis=1, keepdims=True)
        return np.divide(
            cm_float,
            row_totals,
            out=np.zeros_like(cm_float),
            where=row_totals != 0,
        )

    if normalize == "pred":
        col_totals = cm_float.sum(axis=0, keepdims=True)
        return np.divide(
            cm_float,
            col_totals,
            out=np.zeros_like(cm_float),
            where=col_totals != 0,
        )

    total = cm_float.sum()
    if total == 0:
        return np.zeros_like(cm_float)
    return cm_float / total


def plot_confusion_matrix(
    result: ModelEvaluationResult,
    *,
    normalize: NormalizeMode | None = None,
    cmap: str = "Blues",
    figsize: tuple[float, float] = (12.0, 10.0),
    annotate: bool = True,
    cbar: bool = True,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """Plot confusion matrix for an evaluated model, optionally normalized."""
    matrix = result.confusion_matrix_df.to_numpy()

    if normalize is None:
        plot_values = matrix
        fmt = "d"
        suffix = ""
    else:
        plot_values = _normalize_confusion_matrix(matrix, normalize)
        fmt = ".2f"
        suffix = f" ({normalize}-normalized)"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_values,
        annot=annotate,
        fmt=fmt,
        cmap=cmap,
        cbar=cbar,
        xticklabels=result.confusion_matrix_df.columns,
        yticklabels=result.confusion_matrix_df.index,
        ax=ax,
    )
    ax.set_title(f"{result.model_name} Confusion Matrix{suffix}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    if save_path is not None:
        output_file = Path(save_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return ax


def evaluate_and_visualize(
    model_name: str,
    y_true: pd.Series | Sequence,
    y_pred: pd.Series | Sequence,
    *,
    labels: Sequence | None = None,
    digits: int = 3,
    zero_division: int | float | str = 0,
    normalize: NormalizeMode | None = None,
    confusion_matrix_path: str | Path | None = None,
    report_path: str | Path | None = None,
    show_plot: bool = True,
) -> None:
    """evaluate, print summary, and show confusion matrix."""
    result = evaluate_model_predictions(
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        digits=digits,
        zero_division=zero_division,
    )

    print_evaluation_summary(result)

    if report_path is not None:
        save_classification_report(result, report_path)

    plot_confusion_matrix(
        result,
        normalize=normalize,
        save_path=confusion_matrix_path,
        show=show_plot,
    )


def build_metrics_leaderboard(
    results: Sequence[ModelEvaluationResult],
    *,
    sort_by: str = "weighted_f1",
    ascending: bool = False,
) -> pd.DataFrame:
    """sortable DataFrame to compare metrics across models."""
    rows = [
        {"model_name": result.model_name, **result.aggregate_metrics}
        for result in results
    ]
    leaderboard = pd.DataFrame(rows)

    if leaderboard.empty:
        return leaderboard

    if sort_by not in leaderboard.columns:
        available = ", ".join(sorted(leaderboard.columns))
        raise ValueError(
            f"sort_by '{sort_by}' is not available. Choose one of: {available}."
        )

    return leaderboard.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
