# Real Estate Investment Advisor

Predicts whether a property is a **Good Investment** (classification) and estimates its **price after 5 years** (regression). Built with Python, scikit-learn, XGBoost (optional), MLflow (optional) and Streamlit.

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

Place the dataset at `data/india_housing_prices.csv`.

## 2. Run the pipeline

```bash
python -m src.eda      # EDA figures -> reports/figures/
python -m src.train    # trains models, saves models/*.joblib and reports/metrics.json
mlflow ui              # optional: open http://localhost:5000 to see experiment runs
streamlit run app.py   # launch the app at http://localhost:8501
```

## 3. Project structure

```
app.py                 Streamlit app (Predict, Explore, Model Performance)
src/config.py          Paths, feature lists and business assumptions
src/data_utils.py      Load, clean, feature engineering, target creation
src/eda.py             EDA figures
src/train.py           Training, evaluation, MLflow logging, saving models
src/predict.py         Single-property prediction used by the app
models/                Saved models (commit these for Streamlit Cloud)
reports/               metrics.json and EDA figures
```

## 4. Method

**Cleaning:** remove duplicates and the ID column, drop rows without price, size or city, and cap price and size outliers at the 1st and 99th percentiles.

**Features:** `Price_per_SqFt` (price in rupees divided by area), `Age_of_Property`, `Amenities_Count`, plus the categorical fields. Missing values are imputed inside the model pipeline, so the same transforms run at prediction time.

**Good Investment label:** a property is labelled 1 when its price per sq ft is below the median for its own city. This is a simple, explainable domain rule. The price columns are excluded from the classifier inputs, otherwise the model would just learn the rule.

**Future price (5 years):** `Future = Current × (1 + r)^5`. The city rates in `config.py` are assumptions, not data. The regressor learns the growth ratio, and predictions are multiplied by the current price. Learning the ratio stops tree models from failing on prices outside the training range.

**Models compared:** Logistic Regression, Random Forest and XGBoost for classification (best by F1). Linear Regression, Random Forest and XGBoost for regression (best by RMSE), against a fixed 8% growth baseline.

**Metrics:** Accuracy, F1, ROC-AUC and confusion matrix for classification. RMSE, MAE and R² for regression.

## 5. Important notes

- The dataset has no real 5-year future prices, so the regression target is a rule-based proxy. R² close to 1.0 is expected for that reason. Document this limitation in the report. If you have historical price data, replace `add_targets` with real future prices.
- The brief mentions crime rate and an infrastructure score. These columns are not in the dataset, so use the available proxies (Security, Public_Transport_Accessibility, Nearby_Schools, Nearby_Hospitals) instead.
- Check your CSV column names with `df.columns`. If a column is missing, the code skips it automatically.

## 6. Deploy on Streamlit Cloud

1. Push this folder to GitHub, including `models/` and `reports/metrics.json`. Keep the dataset in the repo only if its license allows it.
2. Go to share.streamlit.io, select the repo, set the main file to `app.py`, and click Deploy.

## 7. Dataset notes (india_housing_prices.csv, 250,000 rows)

- `Parking_Space`, `Security` and `Availability_Status` are text (Yes/No or labels like `Ready_to_Move`), so they are treated as categorical features.
- The dataset's own `Price_per_SqFt` is on a different scale (around 0.1 to 90,000). The code recalculates it as `Price_in_Lakhs × 1e5 / Size_in_SqFt`.
- No missing values were found in this dataset, but the pipeline still imputes them, so it works on dirtier data too.
- Random Forest trains on a 60,000-row sample (`RF_TRAIN_SAMPLE` in `config.py`) for speed. Set it higher on a faster machine.
- `Size_in_SqFt` is the strongest signal for Good Investment in this data, because price per sq ft falls as size rises. Mention this in your findings.
