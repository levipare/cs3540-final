import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

pages = [
    st.Page("pages/pca.py", title="PCA", icon=":material/scatter_plot:"),
    st.Page("pages/models.py", title="Model Training Comparison", icon=":material/model_training:"),
]
pg = st.navigation(pages)
pg.run()
