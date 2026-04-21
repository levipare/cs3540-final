import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def resample(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Balance the training set using undersampling on BENIGN followed by SMOTE on minority classes.

    The BENIGN class is undersampled to match the size of the largest attack class,
    then SMOTE generates synthetic samples for all minority classes to match that size.

    Args:
            X_train: Training features.
            y_train: Training labels.

    Returns:
            Tuple of (X_resampled, y_resampled) with balanced class distribution.
    """

    sampling_strategy = {
        "SSH-Patator": 5000,
        "Bot": 5000,
        "Web Attack - Brute Force": 5000,
        "Web Attack - XSS": 5000,
        "Infiltration": 2000,
        "Web Attack - Sql Injection": 2000,
        "Heartbleed": 1000,
    }
    rus = RandomUnderSampler(sampling_strategy={"BENIGN": 500000}, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)

    print("Resampled class distribution:")
    print(pd.Series(y_resampled).value_counts().to_string())

    return X_resampled, y_resampled
