import pandas as pd
from catboost import CatBoostRegressor, Pool
import numpy as np
import os
import pickle

os.makedirs("models", exist_ok=True)

# ------------------------------
# Load test data
df_test = pd.read_csv("data/test.csv", sep=";")

# ------------------------------
# Reduce to one row per product (in case there are multiple)
df_test = df_test.groupby("ID").first().reset_index()

# ------------------------------
# Preprocess to match training in cat_boosting.py
# Dates to datetime and derive simple temporal features
for col in ["phase_in", "phase_out"]:
    if col in df_test.columns:
        df_test[col] = pd.to_datetime(df_test[col], errors="coerce", dayfirst=False)

if "phase_in" in df_test.columns:
    df_test["phase_in_month"] = df_test["phase_in"].dt.month
    df_test["phase_in_week"] = df_test["phase_in"].dt.isocalendar().week.astype("Int64")

if "phase_out" in df_test.columns:
    df_test["phase_out_month"] = df_test["phase_out"].dt.month
    df_test["phase_out_week"] = df_test["phase_out"].dt.isocalendar().week.astype("Int64")

if {"phase_in", "phase_out"} <= set(df_test.columns):
    df_test["season_length_weeks"] = (
        (df_test["phase_out"] - df_test["phase_in"]).dt.days / 7.0
    )

# Drop the same columns as training (keep only intersecting)
drop_cols = [
    "Production",
    "ID",
    "phase_in",
    "phase_out",
    "color_rgb",
    "image_embedding",
    "life_cycle_length",
    "num_stores",
    "num_sizes",
    "has_plus_sizes",
    "price",
    "year",
    "num_week_iso",
    "weekly_sales",
    "weekly_demand",
    "weekly_production_per_store",
]
drop_cols = [c for c in drop_cols if c in df_test.columns]

X_test = df_test.drop(columns=drop_cols)
# Drop any unnamed columns that may appear from CSV artifacts
X_test = X_test.loc[:, ~X_test.columns.str.startswith("Unnamed")]

# Categorical columns as defined in training
cat_cols = [
    "id_season",
    "aggregated_family",
    "family",
    "category",
    "fabric",
    "color_name",
    "length_type",
    "silhouette_type",
    "waist_type",
    "sleeve_length_type",
    "heel_shape_type",
    "toecap_type",
    "woven_structure",
    "knit_structure",
    "print_type",
    "archetype",
    "moment",
    "ocassion",
]

# ------------------------------
# Derive training feature order and categorical set by mirroring training preprocessing
train_feature_order = None
train_cols_to_stringify = None

# Preferred: load feature names used at training time
try:
    with open("models/feature_config.pkl", "rb") as f:
        cfg = pickle.load(f)
    if isinstance(cfg.get("catboost_feature_names"), list):
        train_feature_order = cfg["catboost_feature_names"]
    if isinstance(cfg.get("catboost_cat_features"), list):
        train_cols_to_stringify = cfg["catboost_cat_features"]
except Exception:
    train_feature_order = None

if train_feature_order is None:
    # Fallback: reconstruct from train CSV using same preprocessing
    df_train_like = pd.read_csv("data/processed_data.csv")
    # Dates and derived features like training
    for col in ["phase_in", "phase_out"]:
        if col in df_train_like.columns:
            df_train_like[col] = pd.to_datetime(df_train_like[col], errors="coerce", dayfirst=False)
    if "phase_in" in df_train_like.columns:
        df_train_like["phase_in_month"] = df_train_like["phase_in"].dt.month
        df_train_like["phase_in_week"] = df_train_like["phase_in"].dt.isocalendar().week.astype("Int64")
    if "phase_out" in df_train_like.columns:
        df_train_like["phase_out_month"] = df_train_like["phase_out"].dt.month
        df_train_like["phase_out_week"] = df_train_like["phase_out"].dt.isocalendar().week.astype("Int64")
    if {"phase_in", "phase_out"} <= set(df_train_like.columns):
        df_train_like["season_length_weeks"] = (
            (df_train_like["phase_out"] - df_train_like["phase_in"]).dt.days / 7.0
        )
    drop_cols_train = [
        "Production",
        "ID",
        "phase_in",
        "phase_out",
        "target_demand",
        "color_rgb",
        "image_embedding",
        "life_cycle_length",
        "num_stores",
        "num_sizes",
        "has_plus_sizes",
        "price",
        "year",
        "num_week_iso",
        "weekly_sales",
        "weekly_demand",
        "weekly_production_per_store",
    ]
    drop_cols_train = [c for c in drop_cols_train if c in df_train_like.columns]
    X_train_like = df_train_like.drop(columns=drop_cols_train)
    X_train_like = X_train_like.loc[:, ~X_train_like.columns.str.startswith("Unnamed")]
    train_feature_order = list(X_train_like.columns)
    # If we didn't load cat features from config, infer them from train snapshot
    if train_cols_to_stringify is None:
        obj_cols_train = X_train_like.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        base_cat_cols_train = [c for c in cat_cols if c in X_train_like.columns]
        train_cols_to_stringify = sorted(set(obj_cols_train + base_cat_cols_train))

# If still not available, final fallback considers test dtypes + known list
if train_cols_to_stringify is None:
    obj_cols_test = X_test.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    base_cat_cols_train = [c for c in cat_cols if c in train_feature_order]
    train_cols_to_stringify = sorted(set(obj_cols_test + base_cat_cols_train))

# First pass: ensure candidate categorical columns in test are strings without NaNs
for c in train_cols_to_stringify:
    if c in X_test.columns:
        X_test[c] = X_test[c].fillna("NA").astype(str)

# Reindex test columns to EXACT training order; missing columns will be added
X_test = X_test.reindex(columns=train_feature_order)

# Second pass: enforce dtype/NA on categorical columns after reindex (handles newly added columns)
for c in train_cols_to_stringify:
    if c in X_test.columns:
        X_test[c] = X_test[c].astype(object).where(~X_test[c].isna(), "NA").astype(str)

cat_features_present = [c for c in train_cols_to_stringify if c in X_test.columns]

# ------------------------------
# Load CatBoost model
model = CatBoostRegressor()
model.load_model("models/cat_boosting.json")

# ------------------------------
# Predict
test_pool = Pool(X_test, cat_features=cat_features_present)
preds = model.predict(test_pool)

# ------------------------------
# Build submission
df_pred = pd.DataFrame({
    "ID": df_test["ID"].values,
    "Production": preds
})

df_pred.to_csv("predictions_catboost.csv", index=False)
print("Predictions saved to predictions_catboost.csv")


