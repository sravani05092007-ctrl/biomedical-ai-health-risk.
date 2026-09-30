# Attempts log

Dataset: SYNTHETIC (placeholder - replace with real UCI data)

| # | pop | gens | cx | mut | alpha | #feat | CV-AUC | fitness | all-feat CV-AUC | what changed and why |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 15 | 0.8 | 0.02 | 0.01 | 9 | 0.7688 | 0.7619 | 0.7195 | Baseline configuration. |
| 2 | 20 | 15 | 0.8 | 0.05 | 0.01 | 6 | 0.779 | 0.7744 | 0.7195 | Mutation 0.02->0.05 because best fitness stagnated for 9 generations in attempt 1 (too little exploration). |
| 3 | 20 | 15 | 0.8 | 0.05 | 0.03 | 6 | 0.779 | 0.7652 | 0.7195 | Kept mutation 0.05; feature penalty alpha 0.01->0.03 to test whether a smaller subset keeps the same AUC. |

## Selected features per attempt

- Attempt 1: sex, trestbps, chol, fbs, thalach, exang, slope, ca, thal
- Attempt 2: age, sex, trestbps, thalach, exang, ca
- Attempt 3: age, sex, trestbps, thalach, exang, ca
