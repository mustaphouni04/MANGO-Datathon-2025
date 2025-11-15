import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    "archetype", "heel_shape_type", "toecap_type"
]
df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

# ------------------------------
# Parse embeddings
df_test["image_embedding"] = df_test["image_embedding"].apply(
    lambda x: np.fromstring(x, sep=",")
)

# ------------------------------
# Split color_rgb into 3 ints
def parse_rgb(s):
    if isinstance(s, str):
        return list(map(int, s.split(",")))
    return [0,0,0]

df_test[["r","g","b"]] = df_test["color_rgb"].apply(parse_rgb).tolist()
df_test = df_test.drop(columns=["color_rgb"])

# ------------------------------
# Parse date columns into numeric days (same as training)
date_cols = ["moment", "phase_in", "phase_out"]
for col in date_cols:
    df_test[col] = pd.to_datetime(df_test[col], dayfirst=True, errors="coerce")
    df_test[col] = (df_test[col] - df_test[col].min()).dt.days.fillna(0)

# ------------------------------
# Label encode categorical features using training encoders
cat_cols = [
    "aggregated_family","family","category","fabric","color_name",
    "length_type","silhouette_type","neck_lapel_type","sleeve_length_type","print_type"
]

# Load encoders from training (you should save them when training)
import pickle
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

for col, le in encoders.items():
    df_test[col] = df_test[col].astype(str)
    # Map unseen labels to a new class (-1)
    df_test[col] = df_test[col].apply(lambda x: x if x in le.classes_ else "UNKNOWN")
    
    # Extend the LabelEncoder to handle the new class
    if "UNKNOWN" not in le.classes_:
        le_classes = np.append(le.classes_, "UNKNOWN")
        le.classes_ = le_classes
    
    df_test[col] = le.transform(df_test[col])

#for col in cat_cols:
#    le = encoders[col]
#    df_test[col] = le.transform(df_test[col].astype(str))

# ------------------------------
# Boolean → int
df_test["has_plus_sizes"] = df_test["has_plus_sizes"].astype(int)

# ------------------------------
# Features
weekly_cols = ["weekly_sales","weekly_demand","num_week_iso","year"]
static_cols = [
    "image_embedding","id_season","aggregated_family","family","category",
    "fabric","color_name","length_type","silhouette_type","neck_lapel_type",
    "sleeve_length_type","print_type","moment","phase_in","phase_out",
    "life_cycle_length","num_stores","num_sizes","has_plus_sizes","price",
    "r","g","b"
]

# ------------------------------
# Build sequences per product ID
X_test, test_ids = [], []

for product_id, g in df_test.groupby("ID"):
    g = g.sort_values("num_week_iso")

    # static features
    stat_features = []
    for col in static_cols:
        val = g[col].iloc[0]
        if col == "image_embedding":
            stat_features.extend(val)  # 512 dims
        else:
            stat_features.append(val)
    stat_features = np.array(stat_features, dtype=np.float32)

    # weekly features + static features
    seq_rows = []
    for _, row in g.iterrows():
        week_vec = [row[col] for col in weekly_cols]
        full_vec = np.concatenate([week_vec, stat_features])
        seq_rows.append(full_vec)

    seq = np.vstack(seq_rows)
    X_test.append(seq)
    test_ids.append(product_id)

# ------------------------------
# Load scaler used in training
scaler = pickle.load(open("scaler.pkl", "rb"))
X_test = [scaler.transform(seq) for seq in X_test]

# ------------------------------
# Define Dataset & Dataloader
class ProductDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = torch.tensor(self.X[idx], dtype=torch.float32)
        return seq

def collate_fn(batch):
    lengths = [seq.shape[0] for seq in batch]
    padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return padded, torch.tensor(lengths)

test_loader = DataLoader(
    ProductDataset(X_test), batch_size=8,
    shuffle=False, collate_fn=collate_fn
)

# ------------------------------
# Load GRU model
class GRURegressor(nn.Module):
    def __init__(self, feat_dim, hidden=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h_last = h[-1]
        out = self.fc(h_last)
        return out.squeeze(1)

feat_dim = X_test[0].shape[1]
model = GRURegressor(feat_dim)
model.load_state_dict(torch.load("gru.pt", map_location="cpu"))
model.eval()


with open("target_stats.pkl", "rb") as f:
    target_std, target_mean = pickle.load(f)["std"], pickle.load(f)["mean"]

# ------------------------------
# Predict
preds = []
with torch.no_grad():
    for padded, lengths in test_loader:
        out = model(padded, lengths)
        # Denormalize using training target mean/std
        out = out * target_std + target_mean
        preds.extend(out.numpy())

# ------------------------------
# Save predictions
df_pred = pd.DataFrame({
    "ID": test_ids,
    "Production": preds
})
df_pred.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")
