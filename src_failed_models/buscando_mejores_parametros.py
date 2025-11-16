import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

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

# X = np.hstack([emb_matrix, tab_matrix])  # shape (N, D + num_tab_features)
X = tab_matrix
y = df["target_demand"].to_numpy(dtype=np.float32)

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

# ------------------------------
# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def train_xgb_with_early_stopping(
    X_train, y_train,
    X_val, y_val,
    eta: float,
    max_depth: int,
    verbose: int = 50
):
    """
    Entrena un XGBRegressor con early stopping y devuelve:
    - modelo entrenado
    - mejor nº de árboles (best_iteration)
    - MAE en valid
    """
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error

    model = XGBRegressor(
        n_estimators=5000,       # grande a propósito
        eta=eta,
        max_depth=max_depth,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        max_bin=256,
        random_state=42,
        n_jobs=-1,
		eval_metric="rmse",
		early_stopping_rounds=200
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose
    )

    best_trees = model.best_iteration
    y_val_pred = model.predict(X_val)
    mae_val = mean_absolute_error(y_val, y_val_pred)

    print(f"[eta={eta}, max_depth={max_depth}] → best_trees={best_trees}, MAE_val={mae_val:.4f}")
    return model, best_trees, mae_val


best_model, best_trees, best_mae = train_xgb_with_early_stopping(
    X_train, y_train,
    X_val, y_val,
    eta=0.05,
    max_depth=5
)

print(best_model)
print(best_trees)
print(best_mae)
