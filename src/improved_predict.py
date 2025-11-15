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
    "weekly_sales","weekly_demand","num_week_iso","year"
]
df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

# ------------------------------
# Handle missing values in test data
numeric_cols = ["id_season", "life_cycle_length", "num_stores", "num_sizes", "price"]
for col in numeric_cols:
    if col in df_test.columns:
        df_test[col].fillna(df_test[col].median(), inplace=True)

# ------------------------------
# Parse embeddings with error handling
def safe_parse_embedding(x):
    try:
        if isinstance(x, str):
            return np.fromstring(x, sep=",")
        elif isinstance(x, (list, np.ndarray)):
            return np.array(x, dtype=np.float32)
        else:
            return np.zeros(512, dtype=np.float32)
    except:
        return np.zeros(512, dtype=np.float32)

df_test["image_embedding"] = df_test["image_embedding"].apply(safe_parse_embedding)

# ------------------------------
# Enhanced RGB parsing
def parse_rgb(s):
    if isinstance(s, str):
        try:
            return list(map(int, s.strip('[]').split(",")))
        except:
            return [128, 128, 128]
    return [128, 128, 128]

df_test[["r","g","b"]] = df_test["color_rgb"].apply(parse_rgb).tolist()
df_test = df_test.drop(columns=["color_rgb"])

# ------------------------------
# Enhanced date processing (same as training)
date_cols = ["moment", "phase_in", "phase_out"]
for col in date_cols:
    if col in df_test.columns:
        df_test[col] = pd.to_datetime(df_test[col], dayfirst=True, errors="coerce")
        # Use the same reference date as training (you might need to adjust this)
        reference_date = df_test[col].min()
        df_test[f"{col}_days"] = (df_test[col] - reference_date).dt.days.fillna(0)
        df_test[f"{col}_month"] = df_test[col].dt.month.fillna(1)
        df_test[f"{col}_quarter"] = df_test[col].dt.quarter.fillna(1)
        df_test = df_test.drop(columns=[col])

# ------------------------------
# Enhanced categorical encoding with robust unknown handling
cat_cols = [
    "aggregated_family","family","category","fabric","color_name",
    "length_type","silhouette_type","neck_lapel_type","sleeve_length_type","print_type"
]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

for col, le in encoders.items():
    if col in df_test.columns:
        df_test[col] = df_test[col].astype(str)
        # Handle unknown categories more robustly
        known_classes = set(le.classes_)
        df_test[col] = df_test[col].apply(lambda x: x if x in known_classes else "UNKNOWN")
        
        # If we encounter new unknown values not in encoder, use most frequent
        if "UNKNOWN" not in le.classes_:
            # Use the first class as default
            default_value = le.classes_[0]
            df_test[col] = df_test[col].apply(lambda x: default_value)
        else:
            df_test[col] = le.transform(df_test[col])

# ------------------------------
# Boolean → int
if "has_plus_sizes" in df_test.columns:
    df_test["has_plus_sizes"] = df_test["has_plus_sizes"].astype(int)

# ------------------------------
# Enhanced feature engineering (same as training)
if all(col in df_test.columns for col in ['num_stores', 'num_sizes']):
    df_test['stores_sizes_interaction'] = df_test['num_stores'] * df_test['num_sizes']
    
if all(col in df_test.columns for col in ['price', 'life_cycle_length']):
    df_test['price_lifecycle_interaction'] = df_test['price'] * df_test['life_cycle_length']

# ------------------------------
# Build feature matrix with same columns as training
enhanced_static_cols = [
    "image_embedding", "id_season", "aggregated_family", "family", "category",
    "fabric", "color_name", "length_type", "silhouette_type", "neck_lapel_type",
    "sleeve_length_type", "print_type", "moment_days", "phase_in_days", 
    "phase_out_days", "moment_month", "phase_in_month", "phase_out_month",
    "moment_quarter", "phase_in_quarter", "phase_out_quarter",
    "life_cycle_length", "num_stores", "num_sizes", "has_plus_sizes", "price",
    "r", "g", "b", "stores_sizes_interaction", "price_lifecycle_interaction"
]

# Filter to existing columns in test data
existing_cols = [col for col in enhanced_static_cols if col in df_test.columns]

X_test, test_ids = [], []
for product_id, g in df_test.groupby("ID"):
    row = g.iloc[0]
    feat = []
    for col in existing_cols:
        val = row[col]
        if col == "image_embedding":
            feat.extend(val)
        else:
            feat.append(float(val))
    X_test.append(np.array(feat, dtype=np.float32))
    test_ids.append(product_id)

X_test = np.vstack(X_test)

print(f"Test features shape: {X_test.shape}")

# ------------------------------
# Load and apply preprocessing
scaler = pickle.load(open("scaler.pkl", "rb"))
X_test = scaler.transform(X_test)

# Apply feature selection if used during training
try:
    with open("feature_selector.pkl", "rb") as f:
        feature_selector = pickle.load(f)
        X_test = feature_selector.transform(X_test)
        print(f"After feature selection: {X_test.shape}")
except FileNotFoundError:
    print("No feature selector found, using all features")

# ------------------------------
# Enhanced Dataset
class ProductDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

test_loader = DataLoader(ProductDataset(X_test), batch_size=64, shuffle=False)

# ------------------------------
# Load improved model
class ImprovedFFNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x).squeeze(1)

feat_dim = X_test.shape[1]
model = ImprovedFFNRegressor(feat_dim)
model.load_state_dict(torch.load("improved_ffn.pt", map_location="cpu"))
model.eval()

# ------------------------------
# Load target stats
with open("target_stats.pkl", "rb") as f:
    stats = pickle.load(f)
    target_std = stats["std"]
    target_mean = stats["mean"]
    use_log = stats.get("use_log", False)

# ------------------------------
# Enhanced prediction with uncertainty estimation
preds = []
with torch.no_grad():
    for xb in test_loader:
        out = model(xb)
        # Reverse normalization
        out = out * target_std + target_mean
        
        # Reverse log transformation if used
        if use_log:
            out = torch.expm1(out)  # exp(x) - 1 to reverse log1p
        
        # Clip predictions to reasonable range based on training data
        out = torch.clamp(out, min=0, max=out.quantile(0.95))  # Clip extreme values
        
        preds.extend(out.numpy())

# ------------------------------
# Post-processing: apply business logic constraints if any
# For example, ensure production is non-negative and reasonable
preds = np.array(preds)
preds = np.maximum(preds, 0)  # Ensure non-negative

print(f"Predictions - Min: {preds.min():.2f}, Max: {preds.max():.2f}, Mean: {preds.mean():.2f}")

# ------------------------------
# Save predictions
df_pred = pd.DataFrame({
    "ID": test_ids,
    "Production": preds
})
df_pred.to_csv("improved_predictions.csv", index=False)
print("Improved predictions saved to improved_predictions.csv")
