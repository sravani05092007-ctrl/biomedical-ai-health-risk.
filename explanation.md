# Written explanation

## Problem and approach
Predict a user's heart-health risk from clinical inputs and guide them to relevant services. The system combines a
**Genetic Algorithm (GA)** for feature selection, a **Random Forest** classifier, and a **fuzzy inference system**, then
merges the ML probability and fuzzy score into one interpretable risk category.

## Representation
Each GA chromosome is a **binary vector of length 13** (one gene per clinical feature). Gene = 1 means the feature is used
by the classifier, 0 means it is dropped. A binary encoding maps directly onto the yes/no decision of feature inclusion
and keeps crossover and mutation simple and always valid.

## Fitness function
`fitness = mean 5-fold CV ROC-AUC (Random Forest) - alpha * (n_selected / n_total)`
The first term rewards predictive power; the penalty (alpha = 0.01 to 0.03) rewards smaller subsets, which are cheaper for
users to fill in and less prone to overfitting. Cross-validation runs only on the training split; the 20% test set is never
seen by the GA. Fitness values are cached per chromosome to avoid recomputation.

## Operators and rules
- Initialisation: random binary population (size 20).
- Selection: tournament (size 3).
- Crossover: single-point, probability 0.8.
- Mutation: bit-flip per gene (0.02 in attempt 1, 0.05 afterwards).
- Elitism: the best 2 chromosomes pass unchanged to the next generation.
- Termination: fixed 15 generations.

## Fuzzy risk assessment
Inputs: age, resting blood pressure, cholesterol, maximum heart rate. Each has 3 triangular/shoulder membership functions
(e.g. BP: normal / elevated / high). 12 IF-THEN rules (min for AND, max for OR) are aggregated and defuzzified by centroid
to a 0-100 score, e.g. *IF age is old AND blood pressure is high THEN risk is high*. Fuzzy sets suit clinical thresholds
because risk changes gradually rather than at a hard cut-off, and the fired rules can be shown to the user.

## Hybrid risk
`final = 0.6 x ML probability + 0.4 x fuzzy score`, mapped to Low (<35), Medium (35-65), High (>65). The ML part gives
data-driven accuracy; the fuzzy part gives transparency and a sanity check from clinical knowledge.

## Computational-intelligence rationale
Feature selection is a combinatorial search (2^13 subsets here, far more on larger datasets). A GA explores this space with
a population instead of exhaustive search or greedy selection, and can find interacting features. Fuzzy logic complements it by
handling imprecise clinical categories and providing explanations.

## Attempts summary
See `results/attempts_log.md` for the full table with the one-line "what changed and why" for every attempt after the first.
Attempt 1 (baseline) stalled for 9 generations; attempt 2 raised mutation to 0.05 and reached higher fitness with fewer
features (6 vs 9); attempt 3 raised the feature penalty to 0.03 and found the same 6-feature subset, which suggests that subset
is stable. Attempt 2 had the best fitness and was used for the final model.

## Limitations
- Educational prototype, not a medical device.
- The included results come from a synthetic placeholder dataset; re-run on the real UCI data before submitting.
- The Cleveland dataset is small (~300 rows), so the hold-out set is small and metrics are noisy.
- The resources list is a small curated sample and must be verified.
