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
    return pd.read_parquet(RESULTS / "test_set.parquet")


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

# helper function to get confidence for a single sample from each model
def get_confidence(name: str, artifacts, X_sample: pd.DataFrame) -> float:
    if name == "Logistic Regression":
        X_scaled = artifacts.scaler.transform(X_sample)
        return float(artifacts.model.predict_proba(X_scaled)[0].max())
    if name in ("Random Forest", "LightGBM"):
        return float(artifacts.model.predict_proba(X_sample)[0].max())
    if name == "MLP":
        X_scaled = artifacts.scaler.transform(X_sample).astype(np.float32)
        return float(artifacts.model.predict(X_scaled, verbose=0)[0].max())
    # CNN
    X_scaled = artifacts.scaler.transform(X_sample).astype(np.float32)
    X_cnn = np.expand_dims(X_scaled, axis=-1)
    return float(artifacts.model.predict(X_cnn, verbose=0)[0].max())

# button to choose single sample or batch
mode = st.radio("Mode", ["Single Sample", "Batch"], horizontal=True)

if mode == "Single Sample":
	# set the sample index in session state
    if st.button(" New Random Sample") or "sample_idx" not in st.session_state:
        st.session_state["sample_idx"] = int(np.random.randint(0, len(demo_df)))

	# save the current sample index
    idx = st.session_state["sample_idx"]
	# get the sample and true label
    X_sample = demo_df.drop("Label", axis=1).iloc[[idx]]
    true_label = demo_df["Label"].iloc[idx]

    st.markdown(f"**True label:** `{true_label}`")
    st.caption(
        " Confidence values for MLP and CNN are raw softmax outputs not confidence scores"
	)
	# build table, row for each model
    rows = []
    for name, artifacts, pred_fn in MODELS:
        predicted = pred_fn(artifacts, X_sample).iloc[0]
        confidence = get_confidence(name, artifacts, X_sample)
        rows.append({
            "Model": name,
            "Predicted": predicted,
            "Confidence": f"{confidence:.1%}",
            "Correct": "Yes" if predicted == true_label else "Wrong",
        })

	# helper function to hightlight predictions
    def highlight(row):
        color = "#d4edda" if row["Correct"] == "Yes" else "#f8d7da"
        return [f"background-color: {color}"] * len(row)

	# display the table with highlighted rows
    st.dataframe(
        pd.DataFrame(rows).style.apply(highlight, axis=1),
        use_container_width=True,
        hide_index=True,
    )

else:  # Batch mode
    n = st.slider("Number of samples", min_value=10, max_value=200, value=50, step=10)

    if st.button("Run Batch"):
		# randomly sample n indices from the demo dataset
        indices = np.random.choice(len(demo_df), size=n, replace=False)
        X_batch = demo_df.drop("Label", axis=1).iloc[indices]
        y_batch = demo_df["Label"].iloc[indices].values

		# build table, row for each model
        rows = []
        for name, artifacts, pred_fn in MODELS:
            y_pred = pred_fn(artifacts, X_batch).values
            correct = int((y_pred == y_batch).sum())
            rows.append({
                "Model": name,
                "Correct": correct,
                "Total": n,
                "Accuracy": f"{correct / n:.1%}",
            })

		# create dataframe to display
        batch_df = pd.DataFrame(rows).sort_values("Correct", ascending=False)

		# helper function to highlight the best performing model(s)
        def highlight_batch(row):
            best = batch_df["Correct"].max()
            color = "#d4edda" if row["Correct"] == best else ""
            return [f"background-color: {color}"] * len(row)

		# display the table with hightlights
        st.dataframe(
            batch_df.style.apply(highlight_batch, axis=1),
            use_container_width=True,
            hide_index=True,
        )
