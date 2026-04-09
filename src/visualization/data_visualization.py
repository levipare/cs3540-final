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
