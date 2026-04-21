import pickle
import sys
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.models.cnn import CNNTrainingArtifacts, predict_labels as cnn_predict
from src.models.lgbm import LGBMTrainingArtifacts, predict_labels as lgbm_predict
from src.models.logistic import LogisticTrainingArtifacts, predict_labels as lr_predict
from src.models.mlp import MLPTrainingArtifacts, predict_labels as mlp_predict
from src.models.rf import RFTrainingArtifacts, predict_labels as rf_predict

RESULTS = Path(__file__).resolve().parents[3] / "results"

st.set_page_config(layout="wide", page_title="Model Test Comparison")
st.title("Model Test Comparison")

# load all model artifacts, cache results
@st.cache_resource
def load_artifacts():
	# unpickle artifacts
    with open(RESULTS / "logistic_artifacts.pkl", "rb") as f:
        lr = pickle.load(f)
    with open(RESULTS / "rf_artifacts.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(RESULTS / "lgbm_artifacts.pkl", "rb") as f:
        lgbm = pickle.load(f)

	# load keras models with their scalars and encoders
    mlp_model = keras.models.load_model(RESULTS / "mlp_model.keras")
    with open(RESULTS / "mlp_scaler.pkl", "rb") as f:
        mlp_scaler = pickle.load(f)
    with open(RESULTS / "mlp_encoder.pkl", "rb") as f:
        mlp_encoder = pickle.load(f)
    mlp = MLPTrainingArtifacts(
        model=mlp_model, label_encoder=mlp_encoder, scaler=mlp_scaler, history={}
    )

    cnn_model = keras.models.load_model(RESULTS / "cnn_model.keras")
    with open(RESULTS / "cnn_scaler.pkl", "rb") as f:
        cnn_scaler = pickle.load(f)
    with open(RESULTS / "cnn_encoder.pkl", "rb") as f:
        cnn_encoder = pickle.load(f)
    cnn = CNNTrainingArtifacts(
        model=cnn_model, label_encoder=cnn_encoder, scaler=cnn_scaler, history={}
    )

    return lr, rf, lgbm, mlp, cnn

# load demo dataset from parquet, cache results
@st.cache_data
def load_demo_data() -> pd.DataFrame:
    return pd.read_parquet(RESULTS / "demo_set.parquet")


lr_art, rf_art, lgbm_art, mlp_art, cnn_art = load_artifacts()
demo_df = load_demo_data()

# define model tuples for iteration
MODELS = [
    ("Logistic Regression", lr_art,   lr_predict),
    ("Random Forest",       rf_art,   rf_predict),
    ("LightGBM",            lgbm_art, lgbm_predict),
    ("MLP",                 mlp_art,  mlp_predict),
    ("CNN",                 cnn_art,  cnn_predict),
]
