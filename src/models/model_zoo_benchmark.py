"""Benchmark a variety of classical ML models on the intrusion dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.data.loader import load_dataset
from src.data.resampler import resample


@dataclass
class ModelZooResult:
    """Container for one model's benchmark metrics."""

    model_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    fit_seconds: float
    predict_seconds: float
    status: str
    error: str | None = None


def build_model_zoo(random_state: int = 42) -> dict[str, Any]:
    """Create a diverse set of baseline and strong tabular classifiers."""
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        max_iter=300,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "sgd_log_loss": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SGDClassifier(
                        loss="log_loss",
                        class_weight="balanced",
                        max_iter=200,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "linear_svc": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gaussian_nb": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=15, n_jobs=-1)),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gbdt": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            random_state=random_state,
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=random_state,
        ),
        "mlp_sklearn": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128, 64),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=1e-3,
                        batch_size=512,
                        max_iter=50,
                        random_state=random_state,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }


def run_model_zoo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model_names: list[str] | None = None,
    random_state: int = 42,
    continue_on_error: bool = True,
    log_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fit/evaluate multiple models and return a sorted benchmark table."""
    model_zoo = build_model_zoo(random_state=random_state)
    selected_names = model_names or list(model_zoo.keys())

    unknown_names = sorted(set(selected_names) - set(model_zoo.keys()))
    if unknown_names:
        known = ", ".join(sorted(model_zoo.keys()))
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown_names)}. "
            f"Available: {known}"
        )

    rows: list[dict[str, Any]] = []
    reports: dict[str, str] = {}

    for model_name in selected_names:
        model = model_zoo[model_name]
        if log_progress:
            print(f"Training {model_name}...")

        try:
            fit_start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            y_pred = model.predict(X_test)
            predict_seconds = time.perf_counter() - predict_start

            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")

            reports[model_name] = classification_report(y_test, y_pred, digits=3, zero_division=0)
            rows.append(
                ModelZooResult(
                    model_name=model_name,
                    accuracy=float(acc),
                    macro_f1=float(macro_f1),
                    weighted_f1=float(weighted_f1),
                    fit_seconds=float(fit_seconds),
                    predict_seconds=float(predict_seconds),
                    status="ok",
                ).__dict__
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if not continue_on_error:
                raise
            if log_progress:
                print(f"{model_name} failed: {type(exc).__name__}: {exc}")
            rows.append(
                ModelZooResult(
                    model_name=model_name,
                    accuracy=float("nan"),
                    macro_f1=float("nan"),
                    weighted_f1=float("nan"),
                    fit_seconds=float("nan"),
                    predict_seconds=float("nan"),
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                ).__dict__
            )

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False, na_position="last")
    return results, reports


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a variety of classical models.")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model names, or 'all'.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splits/models.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional fraction of rows sampled before splitting (0 < f <= 1).",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="Label",
        help="Name of target label column.",
    )
    parser.add_argument(
        "--resample-train",
        action="store_true",
        help="Apply src.data.resampler.resample to training split.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force reload/reclean dataset cache.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional path to save benchmark table as CSV.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="",
        help="Optional directory to save per-model classification reports.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many rows to print.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available model names and exit.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on first model error.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress logs.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    available_names = list(build_model_zoo(random_state=args.random_state).keys())

    if args.list_models:
        print("Available models:")
        for name in available_names:
            print(f"- {name}")
        return 0

    if not (0 < args.sample_frac <= 1.0):
        raise ValueError("--sample-frac must be in (0, 1].")
    if not (0 < args.test_size < 1.0):
        raise ValueError("--test-size must be in (0, 1).")

    if args.models.strip().lower() == "all":
        selected_names = available_names
    else:
        selected_names = [name.strip() for name in args.models.split(",") if name.strip()]

    df = load_dataset(force_refresh=args.force_refresh)
    if args.label_col not in df.columns:
        raise KeyError(f"Label column '{args.label_col}' not found in dataset.")

    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=args.random_state)
        if not args.no_progress:
            print(f"Sampled dataset frac={args.sample_frac:.3f}, shape={df.shape}")

    X = df.drop(columns=[args.label_col])
    y = df[args.label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    if args.resample_train:
        X_train, y_train = resample(X_train, y_train)

    results, reports = run_model_zoo(
        X_train,
        y_train,
        X_test,
        y_test,
        model_names=selected_names,
        random_state=args.random_state,
        continue_on_error=not args.strict,
        log_progress=not args.no_progress,
    )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Saved benchmark table to {output_path}")

    if args.reports_dir:
        reports_path = Path(args.reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        for model_name, report_text in reports.items():
            (reports_path / f"{model_name}.txt").write_text(report_text, encoding="utf-8")
        print(f"Saved classification reports to {reports_path}")

    top_k = min(args.top_k, len(results))
    print(f"\nTop {top_k} models by macro_f1:")
    print(results.head(top_k).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
