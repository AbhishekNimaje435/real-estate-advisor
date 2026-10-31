# Real Estate Investment Advisor

A machine learning project that answers two questions about a residential property:

1. **Is it a Good Investment?** (classification, based on price per sq ft compared with its city median)
2. **What could it be worth after 5 years?** (regression, using city-level growth assumptions)

The project includes an end-to-end pipeline (cleaning, EDA, training, MLflow tracking, model registry) and an interactive Streamlit app.

**Live app:** https://real-estate-advisor-god4sfy7oodg7y7sinxbe2.streamlit.app

> **Important:** This project is an academic demonstration. The models reflect patterns in the dataset, not real market outcomes. Read [Limitations](#limitations) before drawing any conclusions. Nothing here is financial advice.

---

## Key results

| Task | Model | Test-set result |
|---|---|---|
| Good Investment (classification) | Random Forest (selected by F1) | Accuracy 0.708, F1 0.709, ROC-AUC 0.773 |
| 5-year price (regression) | Random Forest (selected by RMSE) | R² 1.000 (target is a fixed formula, so this is not evidence of forecasting skill) |
| Fixed 8% growth baseline | none | RMSE 9.55 lakh, R² 0.998 |

**Most important finding:** The classifier's accuracy comes almost entirely from `Size_in_SqFt`. In an ablation test with size removed, accuracy falls to about 0.50, which is random guessing on this balanced dataset. Price per sq ft is roughly independent of size in this data, so size alone predicts the median-based label.

Other findings from EDA:
- Price, size and BHK are almost uniformly distributed, which is unusual for real estate data.
- City, state, locality, facing, furnishing, parking, amenities, schools and hospitals show almost no relationship with price per sq ft.
- The dataset has no sale dates, so price trends over time cannot be measured.

Full details are in [`docs/`](docs/) (Methodology, Findings, Model Performance, Limitations and Use Cases).

---

## Project structure

```
real_estate_advisor/
├── app.py                  # Streamlit app: Predict, Explore, Model Performance
├── requirements.txt        # Pinned library versions (match the training environment)
├── README.md
├── .gitignore
├── src/
│   ├── config.py           # Paths, feature lists, growth-rate assumptions
│   ├── data_utils.py       # Loading, cleaning, feature engineering, targets
│   ├── eda.py              # EDA charts (saved to reports/figures/)
│   ├── train.py            # Training, evaluation, MLflow logging, saving models
│   └── predict.py          # Single-property prediction used by the app
├── data/
│   └── app_sample.csv      # 30,000-row sample used by the app
├── models/
│   ├── classifier.joblib   # Trained Good Investment classifier
│   └── regressor.joblib    # Trained 5-year price regressor
└── reports/
    ├── metrics.json        # All metrics, confusion matrix, feature importance
    └── figures/            # EDA charts
```

---

## Dataset

- **File:** `india_housing_prices.csv`
- **Rows:** 250,000 properties, 23 original columns
- **Not included in this repository** because of its size (41 MB). Place it in `data/` to re-run training.

The app uses `data/app_sample.csv` (30,000 random rows) so that it deploys quickly. The models were trained on all 250,000 rows.

---

## Method at a glance

**Cleaning:** remove duplicates and the ID column, drop rows without price, size or city, and cap price and size at the 1st and 99th percentile.

**Features:**
- `Price_per_SqFt` = Price (lakh) × 1,00,000 ÷ Size (recalculated, because the dataset's own column has a different scale)
- `Amenities_Count` = number of listed amenities
- Categorical features are one-hot encoded. Missing values are imputed inside the pipeline.

**Targets:**
- `Good_Investment` = 1 if the property's price per sq ft is below the median for its city. This label is a domain rule, not a measured outcome.
- `Future_Price_5yr_Lakhs` = Price × (1 + city growth rate)^5. City growth rates are assumptions set in `src/config.py`.

**Models:**
- Classification: Logistic Regression, Random Forest, XGBoost. Best model chosen by F1.
- Regression: Linear Regression, Random Forest, XGBoost. The model learns the growth ratio, then multiplies by the current price. Best model chosen by RMSE. Compared against a fixed 8% baseline.

**Leakage control:** the classifier does not receive price columns, because the label is derived from price.

**Tracking:** all runs are logged to MLflow (`real_estate_advisor` experiment). The best classifier and regressor are registered in the MLflow Model Registry.

---

## Quick start

Requires Python 3.10 or newer (tested on Python 3.13 in Google Colab).

```powershell
# 1. Clone the repository
git clone https://github.com/AbhishekNimaje435/real-estate-advisor.git
cd real-estate-advisor

# 2. Create and activate a virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Run the app (uses the saved models in models/)
streamlit run app.py
```

Open http://localhost:8501 in your browser.

If PowerShell says scripts are disabled, run this once and try activation again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Re-training the models

1. Place `india_housing_prices.csv` in `data/`.
2. Run:
```powershell
python -m src.eda      # EDA charts -> reports/figures/
python -m src.train    # trains models, saves models/ and reports/metrics.json
mlflow ui --backend-store-uri sqlite:///mlflow.db   # optional: view experiment runs
```

The full notebook used for the project (Google Colab) runs every step end to end, including the EDA questions and the MLflow registry.

---

## App features

| Tab | What it does |
|---|---|
| **Predict** | Analyse one property. Shows the Good Investment result with a confidence score, the estimated 5-year price, the expected gain, and a comparison of price per sq ft with the median for the same city and BHK. |
| **Explore** | Filter by state, BHK and price. Shows average price per sq ft by city, Good Investment rate by property type, and a city × BHK heatmap. |
| **Model Performance** | Model comparison tables, confusion matrix, feature importance, and EDA charts. |

---

## Limitations

Read these before using the results:

- **The dataset looks synthetic.** Price, size and BHK are almost uniformly distributed, larger homes do not cost more, and city prices are nearly identical.
- **The Good Investment label is a rule, not an outcome.** It does not measure rental yield, resale value or actual returns.
- **The 5-year price is a formula, not a forecast.** Growth rates are assumptions, and the dataset has no real future prices. R² = 1.0 reflects this, not real accuracy.
- **The classifier mainly learns size.** Without size, accuracy drops to about 50%.
- **Missing features from the brief:** crime rate, infrastructure score, school density, sale dates, and real future prices are not in the dataset. Proxy features were used where possible.
- **Model selection used the test set.** There was no separate validation set, so test scores are slightly optimistic.
- **Category mismatches:** the dataset uses Owner/Builder/Broker and Ready_to_Move/Under_Construction, and Parking_Space is Yes/No only.

See [`docs/`](docs/) for the full discussion.

---

## Deployment

The app is deployed on Streamlit Community Cloud from this repository.

1. Push the repository to GitHub (include `models/`, `reports/` and `data/app_sample.csv`).
2. Go to share.streamlit.io, select the repository, set the main file to `app.py`, and deploy.
3. Python version: 3.13. Library versions are pinned in `requirements.txt`.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · MLflow · Streamlit · Matplotlib · Seaborn · joblib

---

## Acknowledgements

Built as a data science capstone project. Dataset: `india_housing_prices.csv` (250,000 rows).
