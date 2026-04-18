"""Random search utilities and CLI for tuning the CNN model."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import random
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data.loader import load_dataset
from src.data.resampler import resample
from src.models.cnn import predict_labels, train_cnn_classifier


def _make_class_weight(
    y_train: pd.Series,
    *,
    w_benign: float,
    w_bot: float,
    w_brute: float,
    w_xss: float,
) -> dict[int, float]:
    """Build encoded class weights from label-space weights."""
    label_encoder = LabelEncoder().fit(y_train)
    by_label = {label: 1.0 for label in label_encoder.classes_}

    if "BENIGN" in by_label:
        by_label["BENIGN"] = w_benign
    if "Bot" in by_label:
        by_label["Bot"] = w_bot
    if "Web Attack - Brute Force" in by_label:
        by_label["Web Attack - Brute Force"] = w_brute
    if "Web Attack - XSS" in by_label:
        by_label["Web Attack - XSS"] = w_xss

    return {
        int(label_encoder.transform([label])[0]): float(weight)
        for label, weight in by_label.items()
    }


def _sample_cfg(rng: random.Random) -> dict[str, Any]:
    """Sample one hyperparameter configuration."""
    return {
        "conv_filters": rng.choice([(32, 64), (64, 128), (64, 128, 256)]),
        "kernel_size": rng.choice([1, 3, 5]),
        "dense_units": rng.choice([64, 128, 256]),
        "dropout_rate": rng.uniform(0.10, 0.45),
        "learning_rate": 10 ** rng.uniform(np.log10(2e-4), np.log10(2e-3)),
        "batch_size": rng.choice([128, 256, 512]),
        "early_stopping_patience": rng.choice([3, 5, 8]),
        "w_benign": rng.uniform(1.0, 2.0),
        "w_bot": rng.uniform(0.3, 1.0),
        "w_brute": rng.uniform(1.0, 3.5),
        "w_xss": rng.uniform(1.0, 4.0),
    }


def run_cnn_random_search(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    n_trials: int = 25,
    epochs: int = 30,
    random_state: int = 42,
    verbose: int = 0,
    log_progress: bool = True,
    trial_offset: int = 0,
    progress_prefix: str = "",
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Run random search over ``train_cnn_classifier`` and score by validation macro-F1.

    Returns:
        A tuple of:
        1) DataFrame sorted by ``macro_f1`` descending.
        2) Best successful trial row as a dict (or ``None`` if all trials failed).
    """
    rng = random.Random(random_state)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for trial in range(1, n_trials + 1):
        global_trial = trial + trial_offset
        cfg = _sample_cfg(rng)
        class_weight = _make_class_weight(
            y_fit,
            w_benign=cfg["w_benign"],
            w_bot=cfg["w_bot"],
            w_brute=cfg["w_brute"],
            w_xss=cfg["w_xss"],
        )

        keras.backend.clear_session()

        try:
            artifacts = train_cnn_classifier(
                X_fit,
                y_fit,
                X_valid=X_valid,
                y_valid=y_valid,
                epochs=epochs,
                batch_size=cfg["batch_size"],
                early_stopping_patience=cfg["early_stopping_patience"],
                class_weight=class_weight,
                verbose=verbose,
                conv_filters=cfg["conv_filters"],
                kernel_size=cfg["kernel_size"],
                dense_units=cfg["dense_units"],
                dropout_rate=cfg["dropout_rate"],
                learning_rate=cfg["learning_rate"],
                jit_compile=False,
            )

            y_pred = predict_labels(artifacts, X_valid)
            macro_f1 = f1_score(y_valid, y_pred, average="macro")
            weighted_f1 = f1_score(y_valid, y_pred, average="weighted")

            row: dict[str, Any] = {
                "trial": global_trial,
                "status": "ok",
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                **cfg,
            }
            rows.append(row)

            if best is None or row["macro_f1"] > best["macro_f1"]:
                best = row

            if log_progress:
                print(
                    f"{progress_prefix}[{global_trial:02d}] "
                    f"macro_f1={row['macro_f1']:.4f} "
                    f"weighted_f1={row['weighted_f1']:.4f}"
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            row = {
                "trial": global_trial,
                "status": "failed",
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "error": f"{type(exc).__name__}: {exc}",
                **cfg,
            }
            rows.append(row)
            if log_progress:
                print(f"{progress_prefix}[{global_trial:02d}] failed: {row['error']}")

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False, na_position="last")
    return results, best


def _required_min_per_class_for_splits(test_size: float) -> int:
    """Minimum count per class needed to keep both stratified splits viable."""
    train_fraction = 1.0 - test_size
    return max(2, int(np.ceil(2.0 / train_fraction)))


def _sample_with_min_per_class(
    df: pd.DataFrame,
    *,
    label_col: str,
    sample_frac: float,
    min_per_class: int,
    random_state: int,
) -> pd.DataFrame:
    """Sample rows while keeping at least ``min_per_class`` rows per class when possible."""
    if sample_frac >= 1.0:
        return df

    rng = np.random.default_rng(random_state)
    target_size = max(1, int(round(len(df) * sample_frac)))

    guaranteed_indices: list[int] = []
    remaining_indices: list[int] = []
    unavailable_classes: list[tuple[str, int, int]] = []

    for label, group in df.groupby(label_col):
        label_indices = group.index.to_numpy()
        available = len(label_indices)
        take = min(available, min_per_class)
        if available < min_per_class:
            unavailable_classes.append((str(label), available, min_per_class))

        if take > 0:
            if take == available:
                selected = label_indices
                leftover = np.array([], dtype=label_indices.dtype)
            else:
                selected = rng.choice(label_indices, size=take, replace=False)
                leftover = np.setdiff1d(label_indices, selected, assume_unique=False)
            guaranteed_indices.extend(selected.tolist())
            remaining_indices.extend(leftover.tolist())
        else:
            remaining_indices.extend(label_indices.tolist())

    if unavailable_classes:
        details = ", ".join([f"{label}:{available}/{needed}" for label, available, needed in unavailable_classes])
        print(
            "Warning: Could not meet min-per-class for all classes in sampled subset. "
            f"(available/required) => {details}"
        )

    guaranteed_size = len(guaranteed_indices)
    if guaranteed_size >= target_size:
        sampled_indices = np.array(guaranteed_indices)
        if guaranteed_size > target_size:
            print(
                "Warning: sample target size is smaller than guaranteed min-per-class rows. "
                f"Using {guaranteed_size} rows to preserve class minimums."
            )
    else:
        extra_needed = target_size - guaranteed_size
        if extra_needed > 0 and remaining_indices:
            extra_needed = min(extra_needed, len(remaining_indices))
            extra = rng.choice(np.array(remaining_indices), size=extra_needed, replace=False)
            sampled_indices = np.concatenate([np.array(guaranteed_indices), extra])
        else:
            sampled_indices = np.array(guaranteed_indices)

    sampled_df = df.loc[sampled_indices]
    return sampled_df.sample(frac=1.0, random_state=random_state)


def _prepare_data(
    *,
    force_refresh: bool,
    label_col: str,
    sample_frac: float,
    min_per_class: int,
    random_state: int,
    test_size: float,
    valid_size: float,
    resample_train: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load dataset and create fit/validation splits for random search."""
    df = load_dataset(force_refresh=force_refresh)
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in dataset.")

    split_required_min = _required_min_per_class_for_splits(test_size=test_size)
    effective_min_per_class = max(min_per_class, split_required_min)
    if sample_frac < 1.0:
        df = _sample_with_min_per_class(
            df,
            label_col=label_col,
            sample_frac=sample_frac,
            min_per_class=effective_min_per_class,
            random_state=random_state,
        )
        print(
            "Sampled dataset with min-per-class guarantee. "
            f"frac={sample_frac:.3f}, min_per_class={effective_min_per_class}, shape={df.shape}"
        )

    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=valid_size,
        random_state=random_state,
        stratify=y_train,
    )

    if resample_train:
        X_fit, y_fit = resample(X_fit, y_fit)

    return X_fit, X_valid, y_fit, y_valid


def _run_search_shard(shard_args: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Worker entrypoint for one random-search shard."""
    X_fit, X_valid, y_fit, y_valid = _prepare_data(
        force_refresh=shard_args["force_refresh"],
        label_col=shard_args["label_col"],
        sample_frac=shard_args["sample_frac"],
        min_per_class=shard_args["min_per_class"],
        random_state=shard_args["split_random_state"],
        test_size=shard_args["test_size"],
        valid_size=shard_args["valid_size"],
        resample_train=shard_args["resample_train"],
    )
    return run_cnn_random_search(
        X_fit,
        y_fit,
        X_valid,
        y_valid,
        n_trials=shard_args["n_trials"],
        epochs=shard_args["epochs"],
        random_state=shard_args["search_random_state"],
        verbose=shard_args["verbose"],
        log_progress=shard_args["log_progress"],
        trial_offset=shard_args["trial_offset"],
        progress_prefix=shard_args["progress_prefix"],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run random search for src.models.cnn.")
    parser.add_argument("--n-trials", type=int, default=25, help="Number of random search trials.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes to run (1 = sequential).",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs per trial.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splits/search.")
    parser.add_argument("--verbose", type=int, default=0, help="Keras fit verbosity per trial.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction for held-out test split before fit/valid split.",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.2,
        help="Fraction of train split to use for validation.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="Label",
        help="Name of target label column.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional fraction of rows sampled before splitting (0 < f <= 1).",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=1,
        help=(
            "Minimum rows per class in sampled subset before splitting. "
            "Automatically raised as needed for stratified split viability."
        ),
    )
    parser.add_argument(
        "--resample-train",
        action="store_true",
        help="Apply src.data.resampler.resample to the fit split.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force reload/reclean dataset cache in src.data.loader.load_dataset.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional CSV path to save all trial results.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top trial rows to print.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-trial logging output.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if not (0 < args.sample_frac <= 1.0):
        raise ValueError("--sample-frac must be in (0, 1].")
    if not (0 < args.test_size < 1.0):
        raise ValueError("--test-size must be in (0, 1).")
    if not (0 < args.valid_size < 1.0):
        raise ValueError("--valid-size must be in (0, 1).")
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1.")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1.")
    if args.min_per_class < 1:
        raise ValueError("--min-per-class must be >= 1.")

    if args.jobs == 1:
        X_fit, X_valid, y_fit, y_valid = _prepare_data(
            force_refresh=args.force_refresh,
            label_col=args.label_col,
            sample_frac=args.sample_frac,
            min_per_class=args.min_per_class,
            random_state=args.random_state,
            test_size=args.test_size,
            valid_size=args.valid_size,
            resample_train=args.resample_train,
        )
        results, best = run_cnn_random_search(
            X_fit,
            y_fit,
            X_valid,
            y_valid,
            n_trials=args.n_trials,
            epochs=args.epochs,
            random_state=args.random_state,
            verbose=args.verbose,
            log_progress=not args.no_progress,
        )
    else:
        active_jobs = min(args.jobs, args.n_trials)
        if not args.no_progress:
            print(
                f"Running {args.n_trials} trials across {active_jobs} parallel jobs. "
                "On single-GPU systems, this may reduce throughput."
            )

        base_trials = args.n_trials // active_jobs
        remainder = args.n_trials % active_jobs
        shard_sizes = [
            base_trials + (1 if shard_idx < remainder else 0)
            for shard_idx in range(active_jobs)
        ]

        shard_args_list: list[dict[str, Any]] = []
        trial_offset = 0
        for shard_idx, shard_trials in enumerate(shard_sizes):
            shard_args_list.append(
                {
                    "n_trials": shard_trials,
                    "epochs": args.epochs,
                    "verbose": args.verbose,
                    "log_progress": not args.no_progress,
                    "trial_offset": trial_offset,
                    "progress_prefix": f"[job {shard_idx + 1}/{active_jobs}] ",
                    "search_random_state": args.random_state + 10_000 * shard_idx,
                    "split_random_state": args.random_state,
                    "force_refresh": args.force_refresh,
                    "label_col": args.label_col,
                    "sample_frac": args.sample_frac,
                    "min_per_class": args.min_per_class,
                    "test_size": args.test_size,
                    "valid_size": args.valid_size,
                    "resample_train": args.resample_train,
                }
            )
            trial_offset += shard_trials

        shard_results: list[pd.DataFrame] = []
        shard_bests: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=active_jobs) as executor:
            futures = [executor.submit(_run_search_shard, shard_args) for shard_args in shard_args_list]
            for future in as_completed(futures):
                results_part, best_part = future.result()
                shard_results.append(results_part)
                if best_part is not None:
                    shard_bests.append(best_part)

        results = pd.concat(shard_results, ignore_index=True)
        results = results.sort_values("macro_f1", ascending=False, na_position="last")
        best = results.loc[results["status"] == "ok"].head(1).to_dict(orient="records")
        best = best[0] if best else (shard_bests[0] if shard_bests else None)

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")

    top_k = min(args.top_k, len(results))
    print(f"\nTop {top_k} trials by macro_f1:")
    print(results.head(top_k).to_string(index=False))

    print("\nBest trial:")
    print(json.dumps(best, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
