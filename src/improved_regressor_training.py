import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Load train data
df = pd.read_csv("../data/train.csv", sep=";")

# ------------------------------
# Drop unusable columns - more careful selection
drop_cols = [
    "waist_type", "woven_structure", "knit_structure",
    "archetype", "heel_shape_type", "toecap_type",
    "weekly_sales","weekly_demand","num_week_iso","year"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ------------------------------
# Handle missing values before processing
print("Missing values before processing:")
print(df.isnull().sum())

# Fill missing values appropriately
numeric_cols = ["id_season", "life_cycle_length", "num_stores", "num_sizes", "price"]
for col in numeric_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# ------------------------------
# Parse embeddings with error handling
def safe_parse_embedding(x):
    try:
        if isinstance(x, str):
            return np.fromstring(x, sep=",")
        elif isinstance(x, (list, np.ndarray)):
            return np.array(x, dtype=np.float32)
        else:
            return np.zeros(512, dtype=np.float32)  # assuming 512-dim embeddings
    except:
        return np.zeros(512, dtype=np.float32)

df["image_embedding"] = df["image_embedding"].apply(safe_parse_embedding)

# ------------------------------
# Enhanced RGB parsing
def parse_rgb(s):
    if isinstance(s, str):
        try:
            return list(map(int, s.strip('[]').split(",")))
        except:
            return [128, 128, 128]  # neutral gray for errors
    return [128, 128, 128]

df[["r","g","b"]] = df["color_rgb"].apply(parse_rgb).tolist()
df = df.drop(columns=["color_rgb"])

# ------------------------------
# Enhanced date processing
date_cols = ["moment", "phase_in", "phase_out"]
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
        # Extract multiple date features
        df[f"{col}_days"] = (df[col] - df[col].min()).dt.days.fillna(0)
        df[f"{col}_month"] = df[col].dt.month.fillna(1)
        df[f"{col}_quarter"] = df[col].dt.quarter.fillna(1)
        df = df.drop(columns=[col])

# ------------------------------
# Enhanced categorical encoding with unknown handling
cat_cols = [
    "aggregated_family","family","category","fabric","color_name",
    "length_type","silhouette_type","neck_lapel_type","sleeve_length_type","print_type"
]

encoders = {}
for col in cat_cols:
    if col in df.columns:
        # Add "UNKNOWN" category for future unseen values
        le = LabelEncoder()
        unique_vals = df[col].astype(str).unique()
        le.fit(np.append(unique_vals, "UNKNOWN"))
        df[col] = df[col].astype(str)
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else "UNKNOWN")
        df[col] = le.transform(df[col])
        encoders[col] = le

# ------------------------------
# Boolean → int
if "has_plus_sizes" in df.columns:
    df["has_plus_sizes"] = df["has_plus_sizes"].astype(int)

# ------------------------------
# Enhanced feature engineering
# Create interaction features
if all(col in df.columns for col in ['num_stores', 'num_sizes']):
    df['stores_sizes_interaction'] = df['num_stores'] * df['num_sizes']
    
if all(col in df.columns for col in ['price', 'life_cycle_length']):
    df['price_lifecycle_interaction'] = df['price'] * df['life_cycle_length']

# ------------------------------
# Build X, y with enhanced features
enhanced_static_cols = [
    "image_embedding", "id_season", "aggregated_family", "family", "category",
    "fabric", "color_name", "length_type", "silhouette_type", "neck_lapel_type",
    "sleeve_length_type", "print_type", "moment_days", "phase_in_days", 
    "phase_out_days", "moment_month", "phase_in_month", "phase_out_month",
    "moment_quarter", "phase_in_quarter", "phase_out_quarter",
    "life_cycle_length", "num_stores", "num_sizes", "has_plus_sizes", "price",
    "r", "g", "b", "stores_sizes_interaction", "price_lifecycle_interaction"
]

# Filter to existing columns
existing_cols = [col for col in enhanced_static_cols if col in df.columns]
target_col = "Production"

X, y, product_ids = [], [], []
for product_id, g in df.groupby("ID"):
    if target_col not in g.columns or g[target_col].isna().all():
        continue
        
    # Take first row as representative
    row = g.iloc[0]
    if pd.isna(row[target_col]):
        continue
        
    feat = []
    for col in existing_cols:
        val = row[col]
        if col == "image_embedding":
            feat.extend(val)
        else:
            feat.append(float(val))
    X.append(np.array(feat, dtype=np.float32))
    y.append(row[target_col])
    product_ids.append(product_id)

X = np.vstack(X)
y = np.array(y, dtype=np.float32)

print(f"Final dataset shape: {X.shape}, target shape: {y.shape}")

# ------------------------------
# Handle skewed target variable
print(f"Target statistics - Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}, Std: {y.std():.2f}")

# Apply log transformation to handle skewness and large values
y_original = y.copy()
y = np.log1p(y)  # log(1 + y) to handle zeros

# ------------------------------
# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, 5)  # Stratified split
)

# ------------------------------
# Enhanced feature scaling
scaler = RobustScaler()  # More robust to outliers
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Normalize target
target_mean = y_train.mean()
target_std = y_train.std()
y_train_scaled = (y_train - target_mean) / target_std
y_val_scaled = (y_val - target_mean) / target_std

# ------------------------------
# Feature selection to reduce overfitting
if X_train.shape[1] > 50:  # Only if we have many features
    selector = SelectKBest(score_func=f_regression, k=min(50, X_train.shape[1]))
    X_train = selector.fit_transform(X_train, y_train_scaled)
    X_val = selector.transform(X_val)
    feature_selector = selector
else:
    feature_selector = None

# ------------------------------
# Save preprocessing objects
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("target_stats.pkl", "wb") as f:
    pickle.dump({
        "mean": target_mean, 
        "std": target_std,
        "use_log": True  # Flag to indicate log transformation
    }, f)
if feature_selector:
    with open("feature_selector.pkl", "wb") as f:
        pickle.dump(feature_selector, f)

# ------------------------------
# Enhanced Dataset with data augmentation
class ProductDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.augment = augment
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and self.y is not None:
            # Add small noise for regularization
            if torch.rand(1) > 0.7:
                noise = torch.normal(0, 0.01, size=x.shape)
                x = x + noise
        if self.y is not None:
            return x, self.y[idx]
        return x

train_loader = DataLoader(ProductDataset(X_train, y_train_scaled, augment=True), 
                         batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(ProductDataset(X_val, y_val_scaled, augment=False), 
                       batch_size=64, shuffle=False)

# ------------------------------
# Enhanced neural network architecture
class ImprovedFFNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        return self.net(x).squeeze(1)

# ------------------------------
# Enhanced training setup
input_dim = X_train.shape[1]
model = ImprovedFFNRegressor(input_dim)

# Use Huber loss which is more robust to outliers
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=10)
                       

# ------------------------------
# Enhanced training with early stopping
num_epochs = 200
best_val_loss = float('inf')
patience = 20
patience_counter = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * len(yb)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * len(yb)
    val_loss /= len(val_loader.dataset)
    
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_ffn.pt")
    else:
        patience_counter += 1
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(torch.load("best_ffn.pt"))
torch.save(model.state_dict(), "improved_ffn.pt")

print("Training completed!")
