import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data.loader import load_dataset

st.set_page_config(layout="wide")
st.title("PCA Analysis")

with st.spinner("Loading dataset..."):
    df = load_dataset()

with st.spinner("Performing PCA..."):
    X = df.drop(columns=["Label"])
    y = df["Label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_encoded = LabelEncoder().fit_transform(y)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(X_pca.shape[1])])
    pca_df["label"] = y

    counts = pca_df["label"].value_counts()
    weights = 1 / counts
    sample_weights = pca_df["label"].map(weights)
    pca_sample = pca_df.sample(
        n=50000, weights=sample_weights, random_state=42, replace=True
    )

st.subheader("Explained Variance")
for i, e in enumerate(pca.explained_variance_ratio_):
    st.write(f"PC{i + 1}: {e:.2%}")
st.write(f"Total: {sum(pca.explained_variance_ratio_):.2%}")

st.subheader("3D PCA Scatter Plot")
fig = px.scatter_3d(
    pca_sample,
    x="PC1",
    y="PC2",
    z="PC3",
    color="label",
    opacity=0.5,
)
fig.update_traces(marker=dict(size=5))
st.plotly_chart(fig, height=700)
