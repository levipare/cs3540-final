import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def resample(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
	"""Balance the training set using undersampling on BENIGN followed by SMOTE on minority classes.

	The BENIGN class is undersampled to match the size of the largest attack class,
	then SMOTE generates synthetic samples for all minority classes to match that size.

	Args:
		X_train: Training features.
		y_train: Training labels.

	Returns:
		Tuple of (X_resampled, y_resampled) with balanced class distribution.
	"""
	# find the size of the largest minority (non-BENIGN) class
	class_counts = y_train.value_counts()
	largest_minority = class_counts.drop('BENIGN').max()

	# undersample BENIGN down to the largest minority class size
	rus = RandomUnderSampler(sampling_strategy={'BENIGN': largest_minority}, random_state=42)
	X_temp, y_temp = rus.fit_resample(X_train, y_train)

	# oversample all minority classes up to match BENIGN
	smote = SMOTE(random_state=42)
	X_resampled, y_resampled = smote.fit_resample(X_temp, y_temp)

	print("Resampled class distribution:")
	print(pd.Series(y_resampled).value_counts().to_string())

	return X_resampled, y_resampled
