"""Streamlit website:  streamlit run app/app.py"""
import os, sys
import pandas as pd
import streamlit as st

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from predict import load_artifacts, predict_risk

st.set_page_config(page_title="Biomedical AI Health Risk", page_icon="🩺", layout="wide")
st.title("🩺 Biomedical AI Health Risk Assessment")
st.warning("Educational prototype - NOT medical advice. Consult a qualified doctor for any health decision.")

model, meta = load_artifacts()
# (label, min, max, default, step) - clinical input specs
SPEC = {
    "age": ("Age (years)", 18, 100, 50, 1), "sex": ("Sex (0 = female, 1 = male)", 0, 1, 1, 1),
    "cp": ("Chest pain type (1-4)", 1, 4, 3, 1), "trestbps": ("Resting blood pressure (mmHg)", 80, 220, 130, 1),
    "chol": ("Cholesterol (mg/dl)", 100, 600, 220, 1), "fbs": ("Fasting sugar > 120 (0/1)", 0, 1, 0, 1),
    "restecg": ("Resting ECG (0-2)", 0, 2, 0, 1), "thalach": ("Max heart rate achieved", 60, 220, 150, 1),
    "exang": ("Exercise-induced angina (0/1)", 0, 1, 0, 1), "oldpeak": ("ST depression", 0.0, 7.0, 1.0, 0.1),
    "slope": ("ST slope (1-3)", 1, 3, 2, 1), "ca": ("Major vessels coloured (0-3)", 0, 3, 0, 1),
    "thal": ("Thal (3, 6, 7)", 3, 7, 3, 1),
}
needed = list(dict.fromkeys(meta["selected"] + ["age", "trestbps", "chol", "thalach"]))  # chol/HR feed the fuzzy module

tab1, tab2 = st.tabs(["Risk assessment", "Find hospitals / blood banks / transplant services"])

with tab1:
    st.caption(f"Model uses GA-selected features: {', '.join(meta['selected'])}  |  Dataset: {meta['dataset']}")
    vals, cols = {}, st.columns(3)
    for i, f in enumerate(needed):
        label, lo, hi, dflt, step = SPEC[f]
        with cols[i % 3]:
            vals[f] = st.number_input(label, min_value=type(step)(lo), max_value=type(step)(hi),
                                      value=type(step)(dflt), step=step)
    if st.button("Assess risk", type="primary"):
        r = predict_risk(model, meta, vals)
        st.session_state["cat"] = r["category"]
        icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}[r["category"]]
        st.subheader(f"{icon} {r['category']} risk  ({r['final']:.0f}/100)")
        c1, c2 = st.columns(2)
        c1.metric("ML model probability", f"{r['ml']:.0f}%")
        c2.metric("Fuzzy risk score", f"{r['fuzzy']:.0f}/100")
        with st.expander("Which fuzzy rules fired?"):
            for label, s in r["fired"]:
                st.write(f"Rule → **{label}** risk (strength {s})")
        if r["category"] != "Low":
            st.info("Consider consulting a doctor. Use the second tab to find nearby services.")

with tab2:
    df = pd.read_csv(os.path.join(ROOT, "data", "resources.csv"))
    c1, c2 = st.columns(2)
    types = c1.multiselect("Service type", sorted(df["type"].unique()), default=list(df["type"].unique()))
    cities = c2.multiselect("City", sorted(df["city"].unique()), default=list(df["city"].unique()))
    out = df[df["type"].isin(types) & df["city"].isin(cities)]
    for _, row in out.iterrows():
        st.markdown(f"**[{row['name']}]({row['website']})** - {row['type'].replace('_', ' ')}, {row['city']}  \n{row['notes']}")
    st.caption("Curated sample list - verify details on the official website before visiting.")
