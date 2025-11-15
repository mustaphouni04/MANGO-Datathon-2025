import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


df = pd.read_csv("../data/train.csv", sep=";")


drop_cols = [
    "waist_type", "woven_structure", "knit_structure",
    "archetype", "heel_shape_type", "toecap_type"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])


df["image_embedding"] = df["image_embedding"].apply(
    lambda x: np.fromstring(x, sep=",")
)

def parse_rgb(s):
    if isinstance(s, str):
        return list(map(int, s.split(",")))
    return [0,0,0]

df[["r","g","b"]] = df["color_rgb"].apply(parse_rgb).tolist()
df = df.drop(columns=["color_rgb"])

date_cols = ["moment", "phase_in", "phase_out"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    df[col] = (df[col] - df[col].min()).dt.days.fillna(0)

cat_cols = [
    "aggregated_family","family","category","fabric","color_name",
    "length_type","silhouette_type","neck_lapel_type","sleeve_length_type","print_type"
]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

df["has_plus_sizes"] = df["has_plus_sizes"].astype(int)

weekly_cols = ["weekly_sales","weekly_demand","num_week_iso","year"]
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
    g = g.sort_values("num_week_iso")

    stat_features = []
    for col in static_cols:
        val = g[col].iloc[0]
        if col == "image_embedding":
            stat_features.extend(val)  # 512 dims
        else:
            stat_features.append(val)
    stat_features = np.array(stat_features, dtype=np.float32)

    seq_rows = []
    for _, row in g.iterrows():
        week_vec = [row[col] for col in weekly_cols]
        full_vec = np.concatenate([week_vec, stat_features])
        seq_rows.append(full_vec)

    seq = np.vstack(seq_rows)
    X.append(seq)
    y.append(g[target_col].iloc[0])


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

all_train = np.vstack([seq for seq in X_train])
scaler.fit(all_train)

X_train = [scaler.transform(seq) for seq in X_train]
X_val = [scaler.transform(seq) for seq in X_val]

y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
target_mean = y_train.mean()
target_std = y_train.std()
y_train = (y_train - target_mean) / target_std
y_val = (y_val - target_mean) / target_std


with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("target_stats.pkl", "wb") as f:
    pickle.dump({"mean": target_mean, "std": target_std}, f)

class ProductDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = torch.tensor(self.X[idx], dtype=torch.float32)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        return seq, target

def collate_fn(batch):
    sequences, targets = zip(*batch)
    lengths = [seq.shape[0] for seq in sequences]
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(lengths), torch.stack(targets)

train_loader = DataLoader(
    ProductDataset(X_train, y_train), batch_size=8,
    shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    ProductDataset(X_val, y_val), batch_size=8,
    shuffle=False, collate_fn=collate_fn
)

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

feat_dim = X_train[0].shape[1]
model = GRURegressor(feat_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for padded, lengths, targets in loader:
        optimizer.zero_grad()
        preds = model(padded, lengths)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for padded, lengths, targets in loader:
            preds = model(padded, lengths)
            loss = criterion(preds, targets)
            total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)

num_epochs = 30
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

torch.save(model.state_dict(), "gru.pt")

model.eval()
seq_example, length_example, _ = next(iter(val_loader))
pred = model(seq_example, length_example)
pred = pred * target_std + target_mean  # denormalize
print("Example predictions:", pred[:5])

