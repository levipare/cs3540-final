import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_label_distribution(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    label_counts = df['Label'].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(label_counts.index, label_counts.values)
    ax.set_yscale('log')
    ax.set_xlabel('Label')
    ax.set_ylabel('Count (log scale)')
    ax.set_title('Label Distribution')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.bar_label(bars, labels=[f'{c:,}' for c in label_counts.values], fontsize=8, padding=3)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    corr_matrix = df.select_dtypes(include='number').corr()

    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, linewidth=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_correlated_pairs(df, threshold: float = 0.97):
    corr_matrix = df.select_dtypes(include='number').corr()
    high_corr = np.where(np.abs(corr_matrix) > threshold)
    pairs = [
        (corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
        for x, y in zip(*high_corr)
        if x != y and x < y
    ]

    if not pairs:
        print(f"No feature pairs with correlation > {threshold}")
        return

    cols = 5
    rows = math.ceil(len(pairs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    for ax, (f1, f2, corr) in zip(axes.flatten(), pairs):
        ax.scatter(df[f1], df[f2], s=10)
        ax.set_xlabel(f1, fontsize=7)
        ax.set_ylabel(f2, fontsize=7)
        ax.set_title(f"{f1} vs {f2}\ncorr={corr:.4f}", fontsize=8)

    plt.tight_layout()
    plt.show()


def save_f1_chart(report_dict: dict, title: str, save_path: str) -> None:
    classes = [k for k in report_dict if k not in ('accuracy', 'macro avg', 'weighted avg')]
    f1_scores = [report_dict[c]['f1-score'] for c in classes]
    sorted_pairs = sorted(zip(classes, f1_scores), key=lambda x: x[1])
    classes_sorted, f1_sorted = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(list(classes_sorted), list(f1_sorted), color='steelblue')
    ax.set_xlabel('F1 Score')
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.bar_label(bars, fmt='%.3f', padding=3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {save_path}")


def save_f1_comparison_chart(report_dicts: dict, save_path: str) -> None:
    first = next(iter(report_dicts.values()))
    all_classes = [k for k in first if k not in ('accuracy', 'macro avg', 'weighted avg')]
    model_names = list(report_dicts.keys())
    n_models = len(model_names)
    x = np.arange(len(all_classes))
    width = 0.15

    fig, ax = plt.subplots(figsize=(18, 8))
    for i, name in enumerate(model_names):
        f1s = [report_dicts[name].get(c, {}).get('f1-score', 0) for c in all_classes]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, f1s, width, label=name)

    ax.set_xlabel('Attack Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison Across All Models')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {save_path}")


def save_training_curves(history: dict, model_name: str, save_path: str) -> None:
    epochs = range(1, len(history['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history['loss'], label='Train')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} — Loss')
    ax1.legend()

    ax2.plot(epochs, history['accuracy'], label='Train')
    if 'val_accuracy' in history:
        ax2.plot(epochs, history['val_accuracy'], label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} — Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {save_path}")
