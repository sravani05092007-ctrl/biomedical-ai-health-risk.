# Biomedical AI Health Risk Website

A healthcare web app that predicts health risk from patient data using **GA-based feature selection**, a **fuzzy risk system**,
and **machine learning**, and helps users find hospitals, blood banks, and official organ transplantation services.

> Educational prototype - not medical advice.

## Team
| Name | Role |
|---|---|
| sravani | Data & ML |
| saritha | Genetic Algorithm |
| sravya | Fuzzy system |
| all members | Website & resources |

## Problem Statement

Early awareness of respiratory health risks and access to appropriate healthcare information can be difficult for many people. Users may need help understanding their risk indicators and finding reliable healthcare resources.

## Objectives

1. Select the most useful clinical features with a Genetic Algorithm.
2. Predict risk with an ML model and a fuzzy inference system, then combine them.
3. Provide a website with a risk form and a hospital / blood bank / transplant service finder.

##Target Users

People seeking preliminary respiratory health-risk awareness.
Individuals looking for understandable information about their health indicators.
Users who need links to reliable healthcare resources

##Proposed Solution

This project uses a machine-learning model to provide an interpretable respiratory health-risk estimate based on the supported input features. It presents the result in an understandable format and directs users toward relevant official healthcare resources.

##Key Features

Health-related input collection and validation.
Machine-learning-based risk estimation.
Understandable presentation of prediction results.
Guidance and links to official healthcare resources.
Clear explanation of the limitations of AI predictions.

##Expected Impact

The application aims to improve preliminary risk awareness and help users identify appropriate next steps for seeking healthcare information.

##Limitations

This application is an awareness tool, not a medical diagnostic system. Predictions may be inaccurate and should not replace professional medical advice.

##Technology Stack
Python
Streamlit
Machine Learning
Relevant data-processing libraries
How It Addresses the Problem

The application connects health-related inputs to an understandable risk estimate and provides healthcare resource links, addressing both preliminary risk awareness and access to reliable information.

## Architecture
```
Patient data -> Preprocessing -> GA feature selection -> Random Forest -> ML probability -+
                                                                                          +-> Hybrid risk (Low/Medium/High) -> Website -> Resource finder
User inputs (age, BP, chol, HR) -> Fuzzy rules -> Fuzzy score ---------------------------+
```

## Project structure
```
src/data.py                  data loading + preprocessing
src/ga_feature_selection.py  GA, attempts, logs, convergence plots
src/fuzzy_risk.py            fuzzy inference (12 rules)
src/train_model.py           final model + hold-out evaluation
src/predict.py               inference used by the website
app/app.py                   Streamlit website
data/resources.csv           curated hospital / blood bank / transplant list
results/                     attempt logs, convergence plots, final test results
docs/explanation.md          written explanation (representation, operators, rationale)
```

## How to run
```
pip install -r requirements.txt
# put the real UCI Heart Disease file in data/ (see data/README.md)
python src/ga_feature_selection.py all     # runs all attempts, writes results/
python src/train_model.py                  # trains + evaluates final model
streamlit run app/app.py                   # launches the website
```

## Parameter configuration
Defined in `src/ga_feature_selection.py` (`BASE` and `ATTEMPTS`): population 20, generations 15, crossover 0.8, mutation 0.02 / 0.05,
tournament 3, elitism 2, alpha 0.01 / 0.03, 5-fold CV, seed 42.

## Results
- `results/attempts_log.md` - every attempt with parameters, final fitness, and "what changed and why"
- `results/convergence_attempt_*.png` and `attempt_*_log.txt` - convergence evidence
- `results/final_test_results.md` - hold-out comparison: all features vs GA features vs fuzzy vs hybrid

## Status of results
The committed results were produced with a **synthetic placeholder dataset**. Re-run the commands above with the real UCI data before final submission.

## Disclaimer
Educational use only. Not a diagnostic tool.
