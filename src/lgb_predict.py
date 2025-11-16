import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb  # <-- CHANGED

# ------------------------------
# Load test data
df_test = pd.read_csv("../data/test.csv", sep=";")

# ------------------------------
# Drop unusable / empty columns
drop_cols = [
    "waist_type", "woven_structure", "knit_structure",
    "archetype", "heel_shape_type", "toecap_type",
    "weekly_sales", "weekly_demand", "num_week_iso", "year",
    "Unnamed: 28", "Unnamed: 29", "Unnamed: 30", "Unnamed: 31", "Unnamed: 32",
]
df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

# ------------------------------
# Reduce to one row per product (in case there are multiple)
df_test = df_test.groupby("ID").first().reset_index()

# ------------------------------
# Load encoders, date minima, feature config, and model
with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("models/date_mins.pkl", "rb") as f:
    date_mins = pickle.load(f)

with open("models/feature_config.pkl", "rb") as f:
    config = pickle.load(f)

feature_cols = config["feature_cols"]
embedding_dim = config["embedding_dim"]

# Load LightGBM model
bst = lgb.Booster(model_file="models/lgb_mango.txt")  # <-- CHANGED

# ------------------------------
# Parse image embeddings
df_test["image_embedding"] = df_test["image_embedding"].apply(
    lambda x: np.fromstring(x, sep=",").astype(np.float32)
)

# Optional sanity check:
if len(df_test["image_embedding"].iloc[0]) != embedding_dim:
    raise ValueError(
        f"Embedding dimension mismatch: train={embedding_dim}, "
        f"test={len(df_test['image_embedding'].iloc[0])}"
    )

# ------------------------------
# Split RGB
def parse_rgb(s):
    if isinstance(s, str):
        return list(map(int, s.split(",")))
    return [0, 0, 0]

# df_test[["r", "g", "b"]] = df_test["color_rgb"].apply(parse_rgb).tolist()
df_test = df_test.drop(columns=["color_rgb"])

# ------------------------------
# Dates → numeric using SAME minima as training
date_cols = ["moment", "phase_in", "phase_out"]
for col in date_cols:
    df_test[col] = pd.to_datetime(df_test[col], dayfirst=True, errors="coerce")
    base = date_mins[col]
    df_test[col] = (df_test[col] - base).dt.days.fillna(0).astype(np.float32)

# ------------------------------
# Label encode categorical columns with train encoders
cat_cols = list(encoders.keys())

for col in cat_cols:
    le = encoders[col]
    df_test[col] = df_test[col].astype(str)

    # Map unseen categories to "UNKNOWN"
    # If "UNKNOWN" wasn't in training, we append it to classes_
    if "UNKNOWN" not in le.classes_:
        le.classes_ = np.append(le.classes_, "UNKNOWN")

    df_test[col] = df_test[col].apply(lambda x: x if x in le.classes_ else "UNKNOWN")
    df_test[col] = le.transform(df_test[col])

# ------------------------------
# Boolean → int
df_test["has_plus_sizes"] = df_test["has_plus_sizes"].astype(int)

# ------------------------------
# Build X_test: [image_embedding || tabular_features]
missing = [c for c in feature_cols if c not in df_test.columns]
if missing:
    raise ValueError(f"Missing feature columns in test data: {missing}")

emb_matrix_test = np.stack(df_test["image_embedding"].values, axis=0)
tab_matrix_test = df_test[feature_cols].to_numpy(dtype=np.float32)

X_test = np.hstack([emb_matrix_test, tab_matrix_test])

# ------------------------------
# LightGBM prediction
# No DMatrix needed; LightGBM can predict from the NumPy array directly
preds = bst.predict(X_test)  # <-- CHANGED

# ------------------------------
# Build submission
df_pred = pd.DataFrame({
    "ID": df_test["ID"].values,
    "Production": preds  # full-season demand estimate
})

df_pred.to_csv("predictions_lgb.csv", index=False)  # <-- CHANGED
print("Predictions saved to predictions_lgb.csv")
