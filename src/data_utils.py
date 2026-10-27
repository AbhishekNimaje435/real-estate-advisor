"""Loading, cleaning, feature engineering and target creation."""
from datetime import datetime

import pandas as pd

from src import config as C


def load_data(path=C.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    if "ID" in df.columns:
        df = df.drop(columns="ID")
    for col in C.NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Rows without price, size or city cannot be used at all.
    df = df.dropna(subset=["Price_in_Lakhs", "Size_in_SqFt", "City"])
    df = df[(df["Price_in_Lakhs"] > 0) & (df["Size_in_SqFt"] > 0)]
    # Cap outliers at the 1st/99th percentile instead of dropping rows.
    for col in ["Price_in_Lakhs", "Size_in_SqFt"]:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Price in Lakhs -> rupees per sq ft
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
    if "Age_of_Property" not in df.columns and "Year_Built" in df.columns:
        df["Age_of_Property"] = datetime.now().year - df["Year_Built"]
    if "Amenities_Count" not in df.columns and "Amenities" in df.columns:
        df["Amenities_Count"] = df["Amenities"].fillna("").apply(
            lambda s: len([a for a in str(s).split(",") if a.strip()])
        )
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Good investment: priced below the median price per sq ft of its own city.
    city_median = df.groupby("City")["Price_per_SqFt"].transform("median")
    df[C.TARGET_CLS] = (df["Price_per_SqFt"] < city_median).astype(int)
    # 5-year future price: current price compounded at the city growth rate.
    rates = df["City"].map(C.CITY_GROWTH_RATES).fillna(C.BASE_ANNUAL_GROWTH)
    df[C.TARGET_REG] = df["Price_in_Lakhs"] * (1 + rates) ** C.FIVE_YEAR_HORIZON
    return df


def get_feature_columns(df: pd.DataFrame, include_price: bool):
    num = [c for c in C.NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in C.CATEGORICAL_FEATURES if c in df.columns]
    if not include_price:
        num = [c for c in num if c not in C.PRICE_COLS]
    return num, cat
