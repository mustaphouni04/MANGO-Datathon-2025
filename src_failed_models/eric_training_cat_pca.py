import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
from sklearn.decomposition import PCA

os.makedirs("models", exist_ok=True)

# ------------------------------
# Load train data
df = pd.read_csv("data/processed_data.csv")

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

# ------------------------------
# PCA on embeddings → 12D
pca_n_components = 12
pca = PCA(n_components=pca_n_components, random_state=42)
emb_pca = pca.fit_transform(emb_matrix).astype(np.float32)

# Combine PCA-compressed embeddings with tabular features
X = np.hstack([emb_pca, tab_matrix])  # shape (N, pca_n_components + num_tab_features)
y = df["target_demand"].to_numpy(dtype=np.float32)

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

# ------------------------------
# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# XGBoost: quantile regression to penalize under-prediction more
# NOTE: requires xgboost >= 1.7 for "reg:quantileerror".
quantile_alpha = 0.6  # >0.5 means preferring over-production vs under-production

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    "objective": "reg:quantileerror",   # if this fails, fall back to "reg:squarederror"
    "quantile_alpha": quantile_alpha,
    "eval_metric": "mae",
    "tree_method": "hist",              # or "gpu_hist" if you have GPU
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "max_bin": 512,
}

evals = [(dtrain, "train"), (dval, "val")]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=4000,
    evals=evals,
    early_stopping_rounds=200,
    verbose_eval=50
)

# ------------------------------
# Save model + preprocessors
bst.save_model("models/model.json")

# Save PCA object
with open("models/pca.pkl", "wb") as f:
    pickle.dump(pca, f)

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
            "pca_n_components": pca_n_components,
        },
        f,
    )

print("Training finished. Model and preprocessors saved.")
