# Crop Yield Prediction Using Machine Learning

**Student:** Vansh Nain  
**Project type:** Supervised machine-learning regression  
**Target:** `Yield`  
**Dataset file:** `crop_yield.csv`

## 1. Problem framing

This project investigates how well historical crop records can estimate crop yield in later years. The target, `Yield`, is continuous, so this is a regression problem. The analysis is predictive and descriptive; it does not establish that changing rainfall, fertiliser, pesticide, or cultivated area causes a change in yield.

`Production` is excluded from the model features because it is directly related to the target yield and could create target leakage. `Crop_Year` is used to create chronological training, validation, and final test periods, not as a predictor.

## 2. Dataset and cleaning

The committed CSV contains 19,689 rows and 10 columns. The observed year range is 1997–2020. The cleaning pipeline removes exact duplicate rows, records without a usable year or target, and negative target values. Negative numeric input values are treated as missing and imputed within the training-fitted preprocessing pipeline.

The current file has no missing values or exact duplicate rows before cleaning. After cleaning, all 19,689 records remain.

**Dataset source and attribution:** the project owner confirmed that the supplied `crop_yield.csv` was downloaded from Kaggle. The dataset is titled *Agricultural Crop Yield in Indian States Dataset* and is published by `akshatgupta7`: https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset. The supplied copy contains 19,689 rows and 10 columns, covering 1997–2020. The CSV does not contain embedded source or licence metadata; check the licence and usage terms shown on the Kaggle page before redistribution. The public listing's schema and coverage are consistent with the supplied file.

## 3. Exploratory data analysis

The notebook examines:

- Raw and log-transformed yield distributions
- Median yield by year
- Yield summaries by season and crop
- Numeric-variable correlation matrix
- Fertiliser and pesticide intensity per unit area
- Descriptive state-level summaries

These summaries are descriptive. They are not interpreted as causal effects.

## 4. Features and preprocessing

The model uses `Area`, `Annual_Rainfall`, `Fertilizer`, `Pesticide`, `Crop`, `Season`, and `State`. Numeric features are median-imputed, clipped using 1st/99th percentile bounds learned from the training fold, and standardised. Categorical features are mode-imputed and one-hot encoded. All preprocessing is inside scikit-learn pipelines so it is fitted separately within each training fold.

The polynomial regression pipeline adds degree-2 polynomial terms to numeric features only.

## 5. Validation design

A chronological split is used:

- Training: 1997–2015 (15,404 rows)
- Final hold-out: 2016–2020 (4,285 rows)

Expanding-window cross-validation is performed only within the training period:

1. Train through 2007; validate 2008–2010
2. Train through 2010; validate 2011–2013
3. Train through 2013; validate 2014–2015

The final hold-out is not used for hyperparameter tuning or model selection. Decision Tree hyperparameters are selected with `GridSearchCV` using the same temporal folds. The selection score is mean absolute error (MAE) after converting predictions back to the original yield scale.

## 6. Models compared

- Median Dummy Regressor (baseline)
- Multiple Linear Regression
- Polynomial Regression (degree 2)
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- k-Nearest Neighbours Regressor
- Tuned Decision Tree Regressor

## 7. Temporal cross-validation results

| Model | Mean CV MAE | CV MAE standard deviation |
|---|---:|---:|
| Tuned Decision Tree | 15.3283 | 4.2822 |
| Decision Tree | 15.6826 | 4.2820 |
| k-NN Regressor | 20.5598 | 4.5463 |
| Polynomial Regression | 36.2329 | 7.1882 |
| Multiple Linear Regression | 37.2703 | 6.7437 |
| Lasso Regression | 41.1877 | 7.4009 |
| Ridge Regression | 60.4951 | 3.1315 |
| Median Dummy Baseline | 86.4406 | 5.5005 |

The Tuned Decision Tree has the lowest mean temporal-CV MAE among the evaluated models and is selected for the final hold-out report.

## 8. Final hold-out results (2016–2020)

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Tuned Decision Tree | 182.8920 | 11.9181 | 0.9539 |
| Decision Tree | 182.8949 | 12.1568 | 0.9539 |
| k-NN Regressor | 191.7960 | 13.0424 | 0.9493 |
| Polynomial Regression | 410.5824 | 30.4686 | 0.7674 |
| Multiple Linear Regression | 418.5399 | 31.7759 | 0.7583 |
| Lasso Regression | 461.2655 | 36.4333 | 0.7065 |
| Ridge Regression | 576.2993 | 47.3198 | 0.5418 |
| Median Dummy Baseline | 854.6144 | 74.2854 | -0.0075 |

The metrics are calculated on the original `Yield` scale. The baseline comparison indicates that the selected model improves substantially over a median-only prediction on this particular chronological hold-out. The near-equal performance of the tuned and untuned trees should be reported honestly; tuning produced only a small hold-out MAE improvement.

## 9. Diagnostics and interpretation

The notebook includes actual-versus-predicted and residual plots, input-intensity summaries, state-level descriptive summaries, and permutation feature importance on the final hold-out. Permutation importance measures predictive reliance in this dataset; it is not evidence of causal importance.

## 10. Reproducibility

The project records a fixed random seed (`42`), dependency versions, dataset SHA-256, data dimensions, split sizes, selected model, and generated result files. The notebook and standalone Python script follow the same workflow. GitHub Actions executes the notebook and script on each relevant push or manual workflow run.

Generated files:

- `crop_yield_model_comparison.csv`
- `crop_yield_temporal_cv_results.csv`
- `crop_yield_test_predictions.csv`
- `crop_yield_run_metadata.json`

## 11. Limitations

- The final test period covers only 2016–2020; performance may differ in other periods, regions, or datasets.
- The dataset is aggregated observational data and may omit soil, irrigation, local weather, and management variables.
- Input/yield associations are not causal effects.
- The very strong hold-out R² should be interpreted in the context of the dataset's construction, target definition, and feature availability at prediction time.
- Dataset provenance and licence must be verified against the original download source before redistribution or submission.

## 12. How to reproduce

From the repository root:

```bash
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute vanshnain-24570087.ipynb --output executed-notebook.ipynb --ExecutePreprocessor.timeout=2400
python vanshnain_ml.py
```

Both workflows read `crop_yield.csv` and regenerate the evaluation outputs listed above.
