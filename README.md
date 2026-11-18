# Crop Yield Prediction Using Machine Learning

A supervised regression project that compares multiple machine-learning models for estimating crop yield from historical crop, season, state, area, rainfall, fertiliser, and pesticide data.

## Project contents

- `vanshnain-24570087.ipynb` — full notebook with executed outputs
- `vanshnain_ml.py` — standalone Python implementation matching the notebook workflow
- `crop_yield.csv` — project dataset
- `REPORT.md` — methodology, validation design, measured results, limitations, and reproduction instructions
- `requirements.txt` — pinned Python dependencies
- `.github/workflows/validate-notebook.yml` — automated notebook and script execution

## Research question

How accurately can historical crop-yield records be used to predict yield in later years, and which regression approach performs best under time-ordered validation?

The target is `Yield`. `Production` is excluded from model features to avoid using a variable directly related to the target. `Crop_Year` defines chronological training, validation, and hold-out periods and is not used as a predictor.

## Validation design

- Training period: 1997–2015 (15,404 rows)
- Final hold-out: 2016–2020 (4,285 rows)
- Expanding-window temporal cross-validation inside the training period
- Decision Tree hyperparameter tuning with `GridSearchCV` on the temporal folds only
- Model selection based on temporal-CV MAE in original yield units

The selected model is the Tuned Decision Tree. On the 2016–2020 hold-out, its recorded results are RMSE 182.8920, MAE 11.9181, and R² 0.9539. See [REPORT.md](REPORT.md) for the complete model comparisons and methodology.

## Reproduce the project

```bash
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute vanshnain-24570087.ipynb --output executed-notebook.ipynb --ExecutePreprocessor.timeout=2400
python vanshnain_ml.py
```

The notebook and script regenerate:

- `crop_yield_model_comparison.csv`
- `crop_yield_temporal_cv_results.csv`
- `crop_yield_test_predictions.csv`
- `crop_yield_run_metadata.json`

## Dataset source and attribution

The project owner has identified Kaggle as the source from which `crop_yield.csv` was obtained. The dataset is titled **Agricultural Crop Yield in Indian States Dataset**, published by **akshatgupta7**:

- Kaggle dataset: https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset
- Dataset coverage stated in the public listing: Indian states and union territories, 1997–2020.
- File used: `crop_yield.csv` (19,689 rows and 10 columns in the supplied copy).

Please retain the Kaggle attribution and check the licence/usage terms displayed on the Kaggle dataset page before redistribution. The CSV file itself does not embed a source URL or licence, so this attribution is based on the project owner's statement about the download source, not file metadata.

## Limitations

The analysis is predictive and descriptive, not causal. Results apply to this dataset and its 2016–2020 hold-out period; they do not guarantee performance in other years or regions. The strong R² should be interpreted in the context of the dataset's target construction and the availability of predictors at prediction time.

## Author

Vansh Nain
