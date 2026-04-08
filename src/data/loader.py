import glob
import os
from pathlib import Path
import pandas as pd
import kagglehub
from dotenv import load_dotenv

from src.data.cleaner import clean

CACHE_PATH = Path('cache/cleaned.parquet')

def load_dataset(force_refresh: bool = False) -> pd.DataFrame:
	"""Function for loading dataset from Kaggle with cleaning and caching

		Args:
			force_refresh (bool, optional): Whether to force refresh the cleaned dataset or to return the cached version. Defaults to False.

	Returns:
		pd.DataFrame: Cleaned dataframe loaded from Kaggle or cache.
	"""

	load_dotenv()

	# check if cleaned dataset is already cached or if force_refresh is False
	if not force_refresh and CACHE_PATH.exists():
		print(f"Loading cleaned dataset from cache at {CACHE_PATH}")
		return pd.read_parquet(CACHE_PATH)

	# get the raw csv files from Kaggle, concatenate into a single df
	print("Loading raw dataset from Kaggle...")
	path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")
	csv_files = glob.glob(os.path.join(path, "*.csv"))
	df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
	print(f"Raw shape: {df.shape[0]:,} rows, {df.shape[1]:,} columns")

	# call cleaner
	print("Cleaning...")
	df = clean(df)
	print(f"Cleaned shape: {df.shape[0]:,} rows, {df.shape[1]:,} columns")

	# save cleaned df as a parquet file to the cache directory
	CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
	df.to_parquet(CACHE_PATH, index=False)
	print(f"Saved cleaned dataset to cache at {CACHE_PATH}")

	return df