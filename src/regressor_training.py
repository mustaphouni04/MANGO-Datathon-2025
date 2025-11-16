import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle

# ------------------------------
# Load train data
df = pd.read_csv("../data/train.csv", sep=";")

# ------------------------------
# Drop unusable columns
drop_cols = [
    "waist_type", "woven_structure", "knit_structure",
    "archetype", "heel_shape_type", "toecap_type",
    "weekly_sales","weekly_demand","num_week_iso","year"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ------------------------------
# Parse embeddings
df["image_embedding"] = df["image_embedding"].apply(lambda x: np.fromstring(x, sep=","))

# ------------------------------
# Split RGB
def parse_rgb(s):
    if isinstance(s, str):
        return list(map(int, s.split(",")))
    return [0,0,0]

df[["r","g","b"]] = df["color_rgb"].apply(parse_rgb).tolist()
df = df.drop(columns=["color_rgb"])

# ------------------------------
# Dates → numeric
date_cols = ["moment", "phase_in", "phase_out"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    df[col] = (df[col] - df[col].min()).dt.days.fillna(0)

# ------------------------------
# Label encode categorical columns
cat_cols = [
    "aggregated_family","family","category","fabric","color_name",
    "length_type","silhouette_type","neck_lapel_type","sleeve_length_type","print_type"
]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ------------------------------
# Boolean → int
df["has_plus_sizes"] = df["has_plus_sizes"].astype(int)

# ------------------------------
# Build X, y
static_cols = [
    "image_embedding","id_season","aggregated_family","family","category",
    "fabric","color_name","length_type","silhouette_type","neck_lapel_type",
    "sleeve_length_type","print_type","moment","phase_in","phase_out",
    "life_cycle_length","num_stores","num_sizes","has_plus_sizes","price",
    "r","g","b"
]
target_col = "Production"

X, y = [], []
for product_id, g in df.groupby("ID"):
    # Take first row as representative
    row = g.iloc[0]
    feat = []
    for col in static_cols:
        val = row[col]
        if col == "image_embedding":
            feat.extend(val)
        else:
            feat.append(val)
    X.append(np.array(feat, dtype=np.float32))
    y.append(row[target_col])

X = np.vstack(X)
y = np.array(y, dtype=np.float32)

# ------------------------------
# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Normalize target
target_mean = y_train.mean()
target_std = y_train.std()
y_train = (y_train - target_mean) / target_std
y_val = (y_val - target_mean) / target_std

# ------------------------------
# Save encoders and scalers
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("target_stats.pkl", "wb") as f:
    pickle.dump({"mean": target_mean, "std": target_std}, f)

# ------------------------------
# Dataset
class ProductDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ProductDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(ProductDataset(X_val, y_val), batch_size=32, shuffle=False)

# ------------------------------
# Simple feedforward model
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

model = FFNRegressor(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------
# Training
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(yb)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * len(yb)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

torch.save(model.state_dict(), "models/ffn.pt")

