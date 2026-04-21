import streamlit as st

pages = [
    st.Page("pages/pca.py", title="PCA", icon=":material/scatter_plot:"),
    st.Page("pages/models.py", title="Models", icon=":material/model_training:"),
]
pg = st.navigation(pages)
pg.run()
