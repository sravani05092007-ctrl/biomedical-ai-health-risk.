"""Data loading + preprocessing.

Uses the real UCI Heart Disease (Cleveland) file if you place it in data/:
  data/processed.cleveland.data   (UCI original, no header)   OR
  data/heart.csv                  (Kaggle/UCI copy with header, target column)
If neither exists, a SYNTHETIC dataset with the same 13 columns is generated so the
pipeline can run end-to-end. Results on synthetic data are NOT valid for submission.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SEED = 42


def _synthetic(n=303, seed=SEED):
    r = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age": r.normal(54, 9, n).clip(29, 77).round(),
        "sex": r.binomial(1, 0.68, n),
        "cp": r.choice([1, 2, 3, 4], n, p=[.08, .17, .28, .47]),
        "trestbps": r.normal(131, 17, n).clip(94, 200).round(),
        "chol": r.normal(246, 51, n).clip(126, 400).round(),
        "fbs": r.binomial(1, 0.15, n),
        "restecg": r.choice([0, 1, 2], n, p=[.5, .01, .49]),
        "thalach": r.normal(150, 22, n).clip(71, 202).round(),
        "exang": r.binomial(1, 0.33, n),
        "oldpeak": r.exponential(1.0, n).clip(0, 6).round(1),
        "slope": r.choice([1, 2, 3], n, p=[.46, .46, .08]),
        "ca": r.choice([0, 1, 2, 3], n, p=[.58, .22, .13, .07]),
        "thal": r.choice([3, 6, 7], n, p=[.55, .06, .39]),
    })
    z = (0.05 * (df.age - 54) + 0.02 * (df.trestbps - 131) + 0.006 * (df.chol - 246)
         - 0.035 * (df.thalach - 150) + 0.6 * df.oldpeak + 0.8 * df.ca
         + 0.5 * (df.cp == 4) + 0.9 * df.exang + 0.6 * (df.thal == 7) + 0.4 * df.sex - 0.6)
    p = 1 / (1 + np.exp(-z))
    df["target"] = (r.rand(n) < p).astype(int)
    return df


def load_raw():
    p1 = os.path.join(DATA_DIR, "processed.cleveland.data")
    p2 = os.path.join(DATA_DIR, "heart.csv")
    if os.path.exists(p1):
        df = pd.read_csv(p1, header=None, names=COLS + ["target"], na_values="?")
        return df, "UCI Heart Disease (Cleveland) - real"
    if os.path.exists(p2):
        df = pd.read_csv(p2, na_values="?")
        df = df.rename(columns={"num": "target", "thalch": "thalach"})
        return df[COLS + ["target"]], "heart.csv - real"
    return _synthetic(), "SYNTHETIC (placeholder - replace with real UCI data)"


def load_data(test_size=0.2):
    """Returns X_train, X_test, y_train, y_test, feature_names, dataset_label.
    Median imputation + binary target. Features are left unscaled (tree models,
    and the fuzzy module needs raw clinical units)."""
    df, label = load_raw()
    df["target"] = (df["target"] > 0).astype(int)
    df[COLS] = df[COLS].fillna(df[COLS].median())
    X, y = df[COLS].values.astype(float), df["target"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          stratify=y, random_state=SEED)
    return Xtr, Xte, ytr, yte, COLS, label
