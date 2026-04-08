import numpy as np
import pandas as pd

# columns with correlation  > 0.97
CORRELATED_COLS_TO_DROP = [
	'Avg Bwd Segment Size',
	'Avg Fwd Segment Size',
	'Subflow Bwd Bytes',
	'Total Backward Packets',
	'Total Length of Bwd Packets'
]

def clean(df: pd.DataFrame) -> pd.DataFrame:
	"""Function for cleaning dataframe and dropping problematic columns

	Args:
		df (pd.DataFrame): Input dataframe to clean

	Returns:
		pd.DataFrame: Cleaned dataframe
	"""
	df = df.copy()

	# clean up column names
	df.columns = df.columns.str.strip()

	# clean up label names
	df['Label'] = df['Label'].str.strip().str.replace('�', '-')

	# replace na and inf with 0
	df = df.fillna(0)
	df = df.replace([np.inf, -np.inf], 0)

	# drop duplicate rows
	df = df.drop_duplicates(keep='first')

	# remove duplicate columns
	duplicate_columns = set()
	for i in range(df.shape[1]):
		col_i = df.iloc[:, i]
		for j in range(i + 1, df.shape[1]):
			col_j = df.iloc[:, j]
			if col_i.equals(col_j):
				duplicate_columns.add(df.columns[j])

	print(f"Duplicate columns: {duplicate_columns}")
	df = df.drop(columns=list(duplicate_columns))


	# drop highly correlated columns
	cols_to_drop = [c for c in CORRELATED_COLS_TO_DROP if c in df.columns]
	df = df.drop(columns=cols_to_drop)

	return df