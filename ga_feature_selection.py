"""GA-based feature selection.

Representation : binary chromosome, one gene per feature (1 = feature kept)
Fitness        : mean 5-fold CV ROC-AUC (Random Forest) - ALPHA * (n_selected / n_total)
Operators      : tournament selection, single-point crossover, bit-flip mutation, elitism
Usage          : python src/ga_feature_selection.py 1      # run attempt 1 (see ATTEMPTS)
                 python src/ga_feature_selection.py all    # run every attempt
"""
import sys, os, json, time, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from data import load_data

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# ------------------------------------------------------------------
# PARAMETER CONFIGURATION - one entry per attempt.
# 'note' is the mandatory one-line "what changed and why" (attempts 2+).
# ------------------------------------------------------------------
BASE = dict(pop=20, gens=15, cx=0.8, mut=0.02, tourn=3, elite=2,
            alpha=0.01, cv=5, trees=30, seed=42)
ATTEMPTS = {
    1: dict(BASE, note="Baseline configuration."),
    2: dict(BASE, mut=0.05, note="Mutation 0.02->0.05 because best fitness stagnated for 9 generations in attempt 1 (too little exploration)."),
    3: dict(BASE, mut=0.05, alpha=0.03, note="Kept mutation 0.05; feature penalty alpha 0.01->0.03 to test whether a smaller subset keeps the same AUC."),
}


def make_fitness(X, y, cfg):
    cache = {}
    skf = StratifiedKFold(cfg["cv"], shuffle=True, random_state=cfg["seed"])

    def fitness(ch):
        key = tuple(ch)
        if key in cache:
            return cache[key]
        if ch.sum() == 0:
            cache[key] = 0.0
            return 0.0
        m = RandomForestClassifier(n_estimators=cfg["trees"], random_state=cfg["seed"])
        auc = cross_val_score(m, X[:, ch == 1], y, cv=skf, scoring="roc_auc").mean()
        cache[key] = auc - cfg["alpha"] * ch.sum() / len(ch)
        return cache[key]
    return fitness


def run_ga(X, y, cfg, verbose=True):
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    n = X.shape[1]
    fitness = make_fitness(X, y, cfg)
    pop = [np.random.randint(0, 2, n) for _ in range(cfg["pop"])]
    hist = []

    def tournament(fits):
        idx = random.sample(range(len(pop)), cfg["tourn"])
        return pop[max(idx, key=lambda i: fits[i])].copy()

    for g in range(cfg["gens"]):
        fits = [fitness(c) for c in pop]
        hist.append(dict(gen=g, best=max(fits), avg=float(np.mean(fits))))
        if verbose:
            print(f"Gen {g:02d} | best={max(fits):.4f} | avg={np.mean(fits):.4f}")
        order = np.argsort(fits)[::-1]
        new = [pop[i].copy() for i in order[:cfg["elite"]]]
        while len(new) < cfg["pop"]:
            p1, p2 = tournament(fits), tournament(fits)
            if random.random() < cfg["cx"]:
                pt = random.randint(1, n - 1)
                p1[pt:], p2[pt:] = p2[pt:].copy(), p1[pt:].copy()
            for c in (p1, p2):
                flip = np.random.rand(n) < cfg["mut"]
                c[flip] = 1 - c[flip]
                new.append(c)
        pop = new[:cfg["pop"]]
    fits = [fitness(c) for c in pop]
    best = pop[int(np.argmax(fits))]
    return best, max(fits), hist


def run_attempt(k):
    cfg = ATTEMPTS[k]
    Xtr, Xte, ytr, yte, names, label = load_data()
    print(f"\n=== Attempt {k} | dataset: {label} ===\n{cfg}")
    # baseline: all features, same CV protocol
    allfit = make_fitness(Xtr, ytr, dict(cfg, alpha=0.0))(np.ones(Xtr.shape[1], dtype=int))
    t = time.time()
    best, fit, hist = run_ga(Xtr, ytr, cfg)
    sel = [names[i] for i in range(len(names)) if best[i] == 1]
    cv_auc = fit + cfg["alpha"] * best.sum() / len(best)
    res = dict(attempt=k, dataset=label, config={x: cfg[x] for x in cfg if x != "note"},
               note=cfg["note"], fitness=round(fit, 4), cv_auc=round(cv_auc, 4),
               all_features_cv_auc=round(allfit, 4), n_selected=int(best.sum()),
               selected=sel, chromosome=best.tolist(), history=hist,
               seconds=round(time.time() - t, 1))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"attempt_{k}.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(RESULTS, f"attempt_{k}_log.txt"), "w") as f:
        for h in hist:
            f.write(f"Gen {h['gen']:02d} | best={h['best']:.4f} | avg={h['avg']:.4f}\n")
    plt.figure(figsize=(6, 3.6))
    plt.plot([h["best"] for h in hist], label="best fitness")
    plt.plot([h["avg"] for h in hist], label="average fitness")
    plt.xlabel("Generation"); plt.ylabel("Fitness"); plt.title(f"GA convergence - attempt {k}")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"convergence_attempt_{k}.png"), dpi=130); plt.close()
    print(f"Selected ({res['n_selected']}/{len(names)}): {sel}")
    print(f"Final fitness={res['fitness']} | CV-AUC={res['cv_auc']} | all-features CV-AUC={res['all_features_cv_auc']}")
    return res


def write_log():
    """Rebuild results/attempts_log.md from the saved attempt_*.json files."""
    rows = []
    for f in sorted(os.listdir(RESULTS)):
        if f.startswith("attempt_") and f.endswith(".json"):
            rows.append(json.load(open(os.path.join(RESULTS, f))))
    rows.sort(key=lambda r: r["attempt"])
    out = ["# Attempts log", "", f"Dataset: {rows[0]['dataset']}", "",
           "| # | pop | gens | cx | mut | alpha | #feat | CV-AUC | fitness | all-feat CV-AUC | what changed and why |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["config"]
        out.append(f"| {r['attempt']} | {c['pop']} | {c['gens']} | {c['cx']} | {c['mut']} | {c['alpha']} | "
                   f"{r['n_selected']} | {r['cv_auc']} | {r['fitness']} | {r['all_features_cv_auc']} | {r['note']} |")
    out += ["", "## Selected features per attempt", ""]
    for r in rows:
        out.append(f"- Attempt {r['attempt']}: {', '.join(r['selected'])}")
    open(os.path.join(RESULTS, "attempts_log.md"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k in (ATTEMPTS if arg == "all" else [int(arg)]):
        run_attempt(k)
    write_log()
