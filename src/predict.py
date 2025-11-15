import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# Load test data
df_test = pd.read_csv("../data/test.csv", sep=";")

# ------------------------------
# Drop same unusable columns as training
drop_cols = [
    "waist_type", "woven_structure", "knit_structure",
    "archetype", "heel_shape_type", "toecap_type",
    "weekly_sales","weekly_demand","num_week_iso","year"  # no longer used
]
df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

# ------------------------------
# Parse embeddings
df_test["image_embedding"] = df_test["image_embedding"].apply(lambda x: np.fromstring(x, sep=","))

# ------------------------------
# Split RGB
def parse_rgb(s):
    if isinstance(s, str):
        return list(map(int, s.split(",")))
    return [0,0,0]

df_test[["r","g","b"]] = df_test["color_rgb"].apply(parse_rgb).tolist()
df_test = df_test.drop(columns=["color_rgb"])

# ------------------------------
# Dates → numeric
date_cols = ["moment", "phase_in", "phase_out"]
for col in date_cols:
    df_test[col] = pd.to_datetime(df_test[col], dayfirst=True, errors="coerce")
    df_test[col] = (df_test[col] - df_test[col].min()).dt.days.fillna(0)

# ------------------------------
# Label encode categorical columns
cat_cols = [
    "aggregated_family","family","category","fabric","color_name",
    "length_type","silhouette_type","neck_lapel_type","sleeve_length_type","print_type"
]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

for col, le in encoders.items():
    df_test[col] = df_test[col].astype(str)
    df_test[col] = df_test[col].apply(lambda x: x if x in le.classes_ else "UNKNOWN")
    if "UNKNOWN" not in le.classes_:
        le.classes_ = np.append(le.classes_, "UNKNOWN")
    df_test[col] = le.transform(df_test[col])

# ------------------------------
# Boolean → int
df_test["has_plus_sizes"] = df_test["has_plus_sizes"].astype(int)

# ------------------------------
# Static features
static_cols = [
    "image_embedding","id_season","aggregated_family","family","category",
    "fabric","color_name","length_type","silhouette_type","neck_lapel_type",
    "sleeve_length_type","print_type","moment","phase_in","phase_out",
    "life_cycle_length","num_stores","num_sizes","has_plus_sizes","price",
    "r","g","b"
]

# ------------------------------
# Build feature matrix per product
X_test, test_ids = [], []
for product_id, g in df_test.groupby("ID"):
    row = g.iloc[0]
    feat = []
    for col in static_cols:
        val = row[col]
        if col == "image_embedding":
            feat.extend(val)
        else:
            feat.append(val)
    X_test.append(np.array(feat, dtype=np.float32))
    test_ids.append(product_id)

X_test = np.vstack(X_test)

# ------------------------------
# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))
X_test = scaler.transform(X_test)

# ------------------------------
# Define Dataset & Dataloader
class ProductDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

test_loader = DataLoader(ProductDataset(X_test), batch_size=32, shuffle=False)

# ------------------------------
# Load FFN model
class FFNRegressor(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

feat_dim = X_test.shape[1]
model = FFNRegressor(feat_dim)
model.load_state_dict(torch.load("ffn.pt", map_location="cpu"))
model.eval()

# ------------------------------
# Load target stats
with open("target_stats.pkl", "rb") as f:
    stats = pickle.load(f)
    target_std = stats["std"]
    target_mean = stats["mean"]

# ------------------------------
# Predict
preds = []
with torch.no_grad():
    for xb in test_loader:
        out = model(xb)
        out = out * target_std + target_mean  # denormalize
        preds.extend(out.numpy())

# ------------------------------
# Save predictions
df_pred = pd.DataFrame({
    "ID": test_ids,
    "Production": preds
})
df_pred.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")

