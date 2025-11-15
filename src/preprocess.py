import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/train.csv", sep=";")

# ----------------------------------------------------
# Drop unusable columns
drop_cols = [
    "waist_type", "woven_structure", "knit_structure",
    "archetype", "heel_shape_type", "toecap_type"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ----------------------------------------------------
# Parse embeddings
df["image_embedding"] = df["image_embedding"].apply(
    lambda x: np.fromstring(x, sep=",")
)

# ----------------------------------------------------
# Split color_rgb into 3 ints
def parse_rgb(s):
    if isinstance(s, str):
        return list(map(int, s.split(",")))
    return [0,0,0]

df[["r","g","b"]] = df["color_rgb"].apply(parse_rgb).tolist()
df = df.drop(columns=["color_rgb"])

# ----------------------------------------------------
# Parse date columns into numeric day counts
date_cols = ["moment", "phase_in", "phase_out"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    df[col] = (df[col] - df[col].min()).dt.days.fillna(0)

# ----------------------------------------------------
# Label encode categorical features
cat_cols = [
    "aggregated_family","family","category","fabric","color_name",
    "length_type","silhouette_type","neck_lapel_type","sleeve_length_type",
    "print_type"
]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # save if needed later

# ----------------------------------------------------
# Boolean → int
df["has_plus_sizes"] = df["has_plus_sizes"].astype(int)

# ----------------------------------------------------
# Useful weekly feature cols
weekly_cols = ["weekly_sales","weekly_demand","num_week_iso","year"]

# Static columns (repeat per week)
static_cols = [
    "image_embedding","id_season","aggregated_family","family","category",
    "fabric","color_name","length_type","silhouette_type","neck_lapel_type",
    "sleeve_length_type","print_type","moment","phase_in","phase_out",
    "life_cycle_length","num_stores","num_sizes","has_plus_sizes","price",
    "r","g","b"
]

# Target
target_col = "Production"

# ----------------------------------------------------
# Build sequences per ID
X = []
y = []

for product_id, g in df.groupby("ID"):
    g = g.sort_values("num_week_iso")

    # ----- Build static vector (repeated)
    stat_features = []
    for col in static_cols:
        val = g[col].iloc[0]
        if col == "image_embedding":
            stat_features.extend(val)  # embedding expands to 512 dims
        else:
            stat_features.append(val)

    stat_features = np.array(stat_features)

    # ----- Build sequence (one row per week)
    seq_rows = []
    for _, row in g.iterrows():
        week_vec = []

        # weekly features
        for col in weekly_cols:
            week_vec.append(row[col])

        # static features (same every week)
        full_vec = np.concatenate([week_vec, stat_features])
        seq_rows.append(full_vec)

    seq = np.vstack(seq_rows)  # (seq_len, feat_dim)
    X.append(seq)

    # target (scalar)
    y.append(g[target_col].iloc[0])

X = np.array(X, dtype=object)  # ragged, keep object for variable lengths
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X.shape)

print("Example sequence shape:", X[0].shape)
print("Target example:", y[0])