"""Train the final ML model on the GA-selected features and evaluate on the untouched
hold-out test set. Also evaluates the fuzzy score and the hybrid (ML + fuzzy) score.
Usage: python src/train_model.py
"""
import os, sys, json, glob
import numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from data import load_data
from fuzzy_risk import fuzzy_risk

ROOT = os.path.join(os.path.dirname(__file__), "..")
W_ML, W_FUZZY = 0.6, 0.4   # hybrid weights: final = W_ML*ML% + W_FUZZY*fuzzy


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred),
                recall=recall_score(y, pred), f1=f1_score(y, pred), auc=roc_auc_score(y, p))


def main():
    Xtr, Xte, ytr, yte, names, label = load_data()
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results", "attempt_*.json"))]
    best = max(runs, key=lambda r: r["fitness"])
    idx = [i for i, g in enumerate(best["chromosome"]) if g == 1]
    sel = [names[i] for i in idx]
    print(f"Using attempt {best['attempt']} (fitness {best['fitness']}); features: {sel}")

    mk = lambda: RandomForestClassifier(n_estimators=300, random_state=42)
    m_all = mk().fit(Xtr, ytr)
    m_ga = mk().fit(Xtr[:, idx], ytr)
    p_all = m_all.predict_proba(Xte)[:, 1]
    p_ga = m_ga.predict_proba(Xte[:, idx])[:, 1]

    ci = {n: names.index(n) for n in ["age", "trestbps", "chol", "thalach"]}
    fz = np.array([fuzzy_risk(r[ci["age"]], r[ci["trestbps"]], r[ci["chol"]], r[ci["thalach"]]) for r in Xte])
    hybrid = (W_ML * p_ga * 100 + W_FUZZY * fz) / 100

    table = {
        f"RandomForest - all {len(names)} features": metrics(yte, p_all),
        f"RandomForest - GA-selected {len(idx)} features": metrics(yte, p_ga),
        "Fuzzy system only": metrics(yte, fz / 100),
        f"Hybrid ({W_ML} ML + {W_FUZZY} fuzzy)": metrics(yte, hybrid),
    }
    lines = ["# Hold-out test results", "", f"Dataset: {label}", f"Test rows: {len(yte)}",
             f"GA attempt used: {best['attempt']}  |  Selected features: {', '.join(sel)}", "",
             "| Model | Accuracy | Precision | Recall | F1 | AUC |", "|---|---|---|---|---|---|"]
    for k, v in table.items():
        lines.append(f"| {k} | {v['accuracy']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['auc']:.3f} |")
    open(os.path.join(ROOT, "results", "final_test_results.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump(m_ga, os.path.join(ROOT, "models", "risk_model.joblib"))
    json.dump(dict(selected=sel, all_features=names, w_ml=W_ML, w_fuzzy=W_FUZZY, dataset=label,
                   medians={n: float(np.median(Xtr[:, i])) for i, n in enumerate(names)},
                   ranges={n: [float(Xtr[:, i].min()), float(Xtr[:, i].max())] for i, n in enumerate(names)}),
              open(os.path.join(ROOT, "models", "model_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
