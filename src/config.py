"""Central configuration: paths, feature groups and business assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "india_housing_prices.csv"
APP_SAMPLE_PATH = ROOT / "data" / "app_sample.csv"  # 30k-row sample used by the app when deploying
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
# Random Forest limits: keep training fast and saved model files small (GitHub 100 MB limit).
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
RF_MIN_SAMPLES_LEAF = 5
RF_TRAIN_SAMPLE = 60000  # Random Forest trains on at most this many rows (speed); others use all rows
MLFLOW_EXPERIMENT = "real_estate_advisor"

# ---- Investment assumptions (tune these) ----
FIVE_YEAR_HORIZON = 5
BASE_ANNUAL_GROWTH = 0.08  # fixed-rate baseline: 8% per year
# ASSUMED city growth rates (not from the dataset). Change them as needed.
CITY_GROWTH_RATES = {
    "Mumbai": 0.09, "Bangalore": 0.10, "Hyderabad": 0.09, "Pune": 0.08,
    "Chennai": 0.07, "Delhi": 0.07, "Kolkata": 0.06, "Ahmedabad": 0.08,
}

TARGET_CLS = "Good_Investment"
TARGET_REG = "Future_Price_5yr_Lakhs"

# Price columns are excluded from the classifier because the label is derived from price.
PRICE_COLS = ["Price_in_Lakhs", "Price_per_SqFt"]

NUMERIC_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Age_of_Property",
    "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
    "Amenities_Count",
]
CATEGORICAL_FEATURES = [
    "State", "City", "Property_Type", "Furnished_Status",
    "Public_Transport_Accessibility", "Facing", "Owner_Type",
    "Availability_Status", "Security", "Parking_Space",
]
