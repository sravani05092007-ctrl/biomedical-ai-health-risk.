"""Inference: ML probability (GA-selected features) + fuzzy score -> hybrid risk."""
import os, json
import numpy as np, joblib
from fuzzy_risk import fuzzy_risk, category

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_artifacts():
    model = joblib.load(os.path.join(ROOT, "models", "risk_model.joblib"))
    meta = json.load(open(os.path.join(ROOT, "models", "model_meta.json")))
    return model, meta


def predict_risk(model, meta, values):
    """values: dict feature_name -> number. Missing features fall back to training medians."""
    v = {k: values.get(k, meta["medians"][k]) for k in meta["all_features"]}
    x = np.array([[v[f] for f in meta["selected"]]], dtype=float)
    ml = float(model.predict_proba(x)[0, 1]) * 100
    fz, fired, _ = fuzzy_risk(v["age"], v["trestbps"], v["chol"], v["thalach"], explain=True)
    final = meta["w_ml"] * ml + meta["w_fuzzy"] * fz
    return dict(ml=ml, fuzzy=fz, final=final, category=category(final),
                fired=[f for f in fired if f[1] > 0])
