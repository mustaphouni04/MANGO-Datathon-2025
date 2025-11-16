import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb  # <-- CHANGED
import os

os.makedirs("models", exist_ok=True)

# ------------------------------
# Load train data
df = pd.read_csv("../data/processed/train.csv", sep=";")

# ------------------------------
# Drop clearly unusable / empty columns
drop_cols = [
    "waist_type", "woven_structure", "knit_structure",
    "archetype", "heel_shape_type", "toecap_type",
    "weekly_sales", "num_week_iso", "year"  # keep weekly_demand for target
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ------------------------------
# Build product-level target: total demand per ID
# (sum of weekly_demand over the lifecycle)
if "weekly_demand" not in df.columns:
    raise ValueError("weekly_demand not found in train data; make sure processed_data.csv includes it")

target_per_id = (
    df.groupby("ID")["weekly_demand"]
      .sum()
      .rename("target_demand")
      .reset_index()
)

# Reduce to one row per product (take first row as representative of static features)
df_first = df.groupby("ID").first().reset_index()

# Merge target
df = df_first.merge(target_per_id, on="ID")

# Drop leftover per-week demand column from features
df = df.drop(columns=["weekly_demand"])

# ------------------------------
# Parse image embeddings
df["image_embedding"] = df["image_embedding"].apply(
    lambda x: np.fromstring(x, sep=",").astype(np.float32)
)

embedding_dim = len(df["image_embedding"].iloc[0])

# ------------------------------
# Split RGB
def parse_rgb(s):
    if isinstance(s, str):
        return list(map(int, s.split(",")))
    return [0, 0, 0]

# df[["r", "g", "b"]] = df["color_rgb"].apply(parse_rgb).tolist()
df = df.drop(columns=["color_rgb"])

# ------------------------------
# Dates → numeric (store mins so test uses same reference)
date_cols = ["moment", "phase_in", "phase_out"]
date_mins = {}

for col in date_cols:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    col_min = df[col].min()
    date_mins[col] = col_min
    df[col] = (df[col] - col_min).dt.days.fillna(0).astype(np.float32)

# ------------------------------
# Label encode categorical columns
cat_cols = [
    "aggregated_family", "family", "category", "fabric", "color_name",
    "length_type", "silhouette_type", "neck_lapel_type", "sleeve_length_type", "print_type"
]

encoders = {}
for col in cat_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ------------------------------
# Boolean → int
df["has_plus_sizes"] = df["has_plus_sizes"].astype(int)

# ------------------------------
# Define feature columns (tabular, not including image_embedding)
feature_cols = [
    "id_season",
    "aggregated_family", "family", "category", "fabric", "color_name",
    "length_type", "silhouette_type", "neck_lapel_type", "sleeve_length_type", "print_type",
    "moment", "phase_in", "phase_out",
    "life_cycle_length", "num_stores", "num_sizes",
    "has_plus_sizes", "price"
]

# Safety check all feature columns exist
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns in train data: {missing}")

# ------------------------------
# Build X, y
#  - X = [image_embedding || tabular_features]
#  - y = total demand per product
emb_matrix = np.stack(df["image_embedding"].values, axis=0)  # shape (N, D)
tab_matrix = df[feature_cols].to_numpy(dtype=np.float32)

X = np.hstack([emb_matrix, tab_matrix])  # shape (N, D + num_tab_features)
y = df["target_demand"].to_numpy(dtype=np.float32)

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

# ------------------------------
# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# LightGBM: quantile regression
quantile_alpha = 0.8  # >0.5 means preferring over-production vs under-production

# Create LightGBM Datasets
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

# Translate XGBoost params to LightGBM params
params = {
    'objective': 'quantile',
    'alpha': quantile_alpha,
    'metric': 'mae',            # 'mae' is 'l1'
    'boosting_type': 'gbdt',
    'n_jobs': -1,               # Use all cores
    'verbose': -1,
    'learning_rate': 0.03,      # 'eta' in XGBoost
    'max_depth': 12,
    'max_bin': 512,
    'bagging_fraction': 0.9,    # 'subsample' in XGBoost
    'bagging_freq': 1,          # Enable bagging
    'feature_fraction': 0.9,    # 'colsample_bytree' in XGBoost
    'min_sum_hessian_in_leaf': 3, # 'min_child_weight' in XGBoost
}

# Setup callbacks for early stopping and logging
callbacks = [
    lgb.log_evaluation(period=50),
    lgb.early_stopping(stopping_rounds=500)
]

print("Starting LightGBM training...")
bst = lgb.train(
    params,
    dtrain,
    num_boost_round=12000,
    valid_sets=[dtrain, dval],
    valid_names=['train', 'val'], # Corresponds to the evals list
    callbacks=callbacks
)

# ------------------------------
# Save model + preprocessors
bst.save_model("models/lgb_mango.txt")  # <-- CHANGED

with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("models/date_mins.pkl", "wb") as f:
    pickle.dump(date_mins, f)

with open("models/feature_config.pkl", "wb") as f:
    pickle.dump(
        {
            "feature_cols": feature_cols,
            "embedding_dim": embedding_dim,
            "quantile_alpha": quantile_alpha,
        },
        f,
    )

print("Training finished. Model and preprocessors saved.")
