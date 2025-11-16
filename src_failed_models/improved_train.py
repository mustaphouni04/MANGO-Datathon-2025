import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

os.makedirs("models", exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class QuantileLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        error = y_true - y_pred
        
        loss = torch.max((self.alpha * error), ((self.alpha - 1) * error))
        return torch.mean(loss)


class MixedInputNet(nn.Module):
    """
    A mixed-input neural network for image embeddings and tabular data.
    """
    def __init__(self, embedding_dim, num_numerical_features, cat_embedding_sizes):
        super().__init__()
        
        # --- 1. Image Embedding Branch (bigger) ---
        self.img_branch = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # --- 2. Categorical Embedding Branch ---
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(num_classes, embed_dim) for num_classes, embed_dim in cat_embedding_sizes]
        )
        total_cat_embed_dim = sum([embed_dim for _, embed_dim in cat_embedding_sizes])
        # Add a processing branch for concatenated categorical embeddings (bigger)
        self.cat_branch = nn.Sequential(
            nn.Linear(total_cat_embed_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # --- 2b. Numerical Features Branch (new, bigger) ---
        self.num_branch = nn.Sequential(
            nn.Linear(num_numerical_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # --- 3. Combined Head ---
        # Input: 256 (from img) + 128 (from num) + 256 (from cats)
        combined_input_dim = 256 + 128 + 256
        
        self.combiner = nn.Sequential(
            nn.Linear(combined_input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1) # Single output for regression
        )

    def forward(self, img_emb, num_data, cat_data):
        # Process image embedding
        x_img = self.img_branch(img_emb)
        
        # Process categorical features
        x_cats_list = []
        for i, embed_layer in enumerate(self.cat_embeddings):
            x_cats_list.append(embed_layer(cat_data[:, i]))
        x_cats = torch.cat(x_cats_list, dim=1)
        x_cats = self.cat_branch(x_cats)
        
        # Process numerical features
        x_num = self.num_branch(num_data)
        
        # Combine all features
        combined = torch.cat([x_img, x_num, x_cats], dim=1)
        
        # Get final prediction
        output = self.combiner(combined)
        return output


class DemandDataset(Dataset):
    """Custom PyTorch Dataset"""
    def __init__(self, df, num_cols, cat_cols):
        # Convert all data to numpy/tensors at initialization
        self.img_emb = np.stack(df["image_embedding"].values).astype(np.float32)
        self.num_data = df[num_cols].values.astype(np.float32)
        self.cat_data = df[cat_cols].values.astype(np.int64) # Use int64 for embedding lookup
        self.target = df["target_demand"].values.astype(np.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # Return a tuple of (inputs), target
        inputs = (
            self.img_emb[idx],
            self.num_data[idx],
            self.cat_data[idx]
        )
        target = self.target[idx]
        return inputs, target


# --- 2. Load and Preprocess Data ---
print("Loading data...")
df = pd.read_csv("data/processed_data.csv")

# Drop columns (same as your script)
drop_cols = [
    "waist_type", "woven_structure", "knit_structure",
    "archetype", "heel_shape_type", "toecap_type",
    "weekly_sales", "num_week_iso", "year"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Build product-level target (same as your script)
if "weekly_demand" not in df.columns:
    raise ValueError("weekly_demand not found in train data")

target_per_id = (
    df.groupby("ID")["weekly_demand"]
      .sum()
      .rename("target_demand")
      .reset_index()
)
df_first = df.groupby("ID").first().reset_index()
df = df_first.merge(target_per_id, on="ID")
df = df.drop(columns=["weekly_demand"])

# Parse image embeddings (same as your script)
df["image_embedding"] = df["image_embedding"].apply(
    lambda x: np.fromstring(x, sep=",").astype(np.float32)
)
embedding_dim = len(df["image_embedding"].iloc[0])
df = df.drop(columns=["color_rgb"], errors='ignore')

# Dates -> numeric (same as your script)
date_cols = ["moment", "phase_in", "phase_out"]
date_mins = {}
for col in date_cols:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    col_min = df[col].min()
    date_mins[col] = col_min
    df[col] = (df[col] - col_min).dt.days.fillna(0).astype(np.float32)

# --- 3. Preprocessing for Neural Network ---

# **NEW**: Define which columns are categorical vs. numerical
cat_cols = [
    "aggregated_family", "family", "category", "fabric", "color_name",
    "length_type", "silhouette_type", "neck_lapel_type", "sleeve_length_type", "print_type"
]
num_cols = [
    "id_season",
    "moment", "phase_in", "phase_out", # Now numerical
    "life_cycle_length", "num_stores", "num_sizes",
    "has_plus_sizes", "price"
]
# Boolean -> int (same)
df["has_plus_sizes"] = df["has_plus_sizes"].astype(int)

# Label encode categorical columns (same)
encoders = {}
for col in cat_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# **NEW**: Split before scaling
print("Splitting data...")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# **NEW**: Scale numerical features
print("Scaling numerical features...")
scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
val_df[num_cols] = scaler.transform(val_df[num_cols])

# Create Datasets and DataLoaders
print("Creating DataLoaders...")
train_dataset = DemandDataset(train_df, num_cols, cat_cols)
val_dataset = DemandDataset(val_df, num_cols, cat_cols)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

# --- 4. Initialize Model ---
print("Initializing model...")
# Determine embedding sizes for categorical features
# Rule of thumb: min(50, num_classes // 2)
cat_embedding_sizes = []
for col in cat_cols:
    num_classes = len(encoders[col].classes_)
    embed_dim = min(50, num_classes // 2 + 1)
    cat_embedding_sizes.append((num_classes, embed_dim))

model = MixedInputNet(
    embedding_dim=embedding_dim,
    num_numerical_features=len(num_cols),
    cat_embedding_sizes=cat_embedding_sizes
).to(DEVICE)

# --- 5. Training ---
quantile_alpha = 0.7 # Same as your XGBoost
criterion = QuantileLoss(alpha=quantile_alpha)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

num_epochs = 500 # NNs can take longer, but we have early stopping
early_stopping_rounds = 40 # Slightly higher patience for larger models
best_val_loss = np.inf
epochs_no_improve = 0

print(f"--- Starting Training (Epochs: {num_epochs}, Patience: {early_stopping_rounds}) ---")
for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train()
    train_loss_total = 0.0
    for (img, num, cat), target in train_loader:
        # Move data to device
        img, num, cat, target = (
            img.to(DEVICE),
            num.to(DEVICE),
            cat.to(DEVICE),
            target.to(DEVICE)
        )
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(img, num, cat)
        
        # Calculate loss
        loss = criterion(preds, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        train_loss_total += loss.item()
        
    avg_train_loss = train_loss_total / len(train_loader)

    # --- Validation Phase ---
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for (img, num, cat), target in val_loader:
            img, num, cat, target = (
                img.to(DEVICE),
                num.to(DEVICE),
                cat.to(DEVICE),
                target.to(DEVICE)
            )
            preds = model(img, num, cat)
            loss = criterion(preds, target)
            val_loss_total += loss.item()
            
    avg_val_loss = val_loss_total / len(val_loader)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d}/{num_epochs:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- Early Stopping ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), "models/nn_model.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_rounds:
            print(f"Early stopping triggered at epoch {epoch+1} with best val loss: {best_val_loss:.4f}")
            break

print("--- Training Finished ---")

# --- 6. Save Preprocessors ---
print("Saving preprocessors...")
with open("models/encoders_nn.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("models/date_mins_nn.pkl", "wb") as f:
    pickle.dump(date_mins, f)

# **NEW**: Save the scaler
with open("models/scaler_nn.pkl", "wb") as f:
    pickle.dump(scaler, f)

# **NEW**: Save the new feature config
with open("models/feature_config_nn.pkl", "wb") as f:
    pickle.dump(
        {
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "embedding_dim": embedding_dim,
            "quantile_alpha": quantile_alpha,
        },
        f,
    )
print("Training finished. Model and preprocessors saved.")
