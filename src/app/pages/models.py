import json
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# add the parent directory so we can import from src
sys.path.append(str(Path(__file__).resolve().parents[3]))

# set up constants for loading results 
RESULTS = Path(__file__).resolve().parents[3] / "results"
SKIP_KEYS = {"accuracy", "macro avg", "weighted avg"}
MODEL_LABELS = ["Logistic Regression", "Random Forest", "LightGBM", "MLP", "CNN"]
MODEL_KEYS   = ["logistic","rf", "lgbm", "mlp", "cnn"]
F1_CHARTS = {
    "logistic": "logistic_f1_chart.png",
    "rf":       "rf_f1_chart.png",
    "lgbm":     "lgbm_f1.png",
    "mlp":      "mlp_f1.png",
    "cnn":      "cnn_f1.png",
}

st.set_page_config(layout="wide", page_title="Model Comparison")
st.title("Model Training Comparison")


# load report for each model, cache results
@st.cache_data
def load_reports() -> dict[str, dict]:
    return {
        label: json.loads((RESULTS / f"{key}_report.json").read_text())
        for label, key in zip(MODEL_LABELS, MODEL_KEYS)
    }

# build a leaderboard dataframe from the report dicts
def build_leaderboard(report_dicts: dict) -> pd.DataFrame:
    rows = []
    for model_name, report in report_dicts.items():
        rows.append({
            "Model": model_name,
            "Accuracy": round(report["accuracy"]["f1-score"], 4),
            "Macro F1": round(report["macro avg"]["f1-score"], 4),
            "Weighted F1": round(report["weighted avg"]["f1-score"], 4),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("Weighted F1", ascending=False)
        .reset_index(drop=True)
    )

report_dicts = load_reports()
attack_classes = [k for k in next(iter(report_dicts.values())) if k not in SKIP_KEYS]

st.subheader("Model Training Leaderboard")
st.dataframe(build_leaderboard(report_dicts), use_container_width=True, hide_index=True)

st.subheader("F1 Score Comparison by Attack Class")

fig = go.Figure()
for label, report in report_dicts.items():
	f1_scores = [report.get(cls, {}).get("f1-score", 0.0) for cls in attack_classes]
	fig.add_trace(go.Bar(name=label, x=attack_classes, y=f1_scores))

fig.update_layout(
    barmode="group",
    xaxis_tickangle=-45,
    yaxis=dict(range=[0, 1.1], title="F1 Score"),
    xaxis_title="Attack Class",
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

