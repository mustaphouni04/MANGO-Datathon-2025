import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler # Needed to load objects

# --- 0. Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Re-define Model and Dataset Classes ---
# (These must match the training script exactly)

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
    def __init__(self, embedding_dim, num_numerical_features, cat_embedding_sizes):
        super().__init__()
        self.img_branch = nn.Sequential(
            nn.Linear(embedding_dim, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Dropout(0.4), nn.Linear(1024, 512), nn.ReLU(),
            nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(num_classes, embed_dim) for num_classes, embed_dim in cat_embedding_sizes]
        )
        total_cat_embed_dim = sum([embed_dim for _, embed_dim in cat_embedding_sizes])
        self.cat_branch = nn.Sequential(
            nn.Linear(total_cat_embed_dim, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(),
        )
        self.num_branch = nn.Sequential(
            nn.Linear(num_numerical_features, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(),
        )
        combined_input_dim = 256 + 128 + 256
        self.combiner = nn.Sequential(
            nn.Linear(combined_input_dim, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Dropout(0.5), nn.Linear(1024, 512), nn.ReLU(),
            nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Dropout(0.3), nn.Linear(256, 1)
        )
    def forward(self, img_emb, num_data, cat_data):
        x_img = self.img_branch(img_emb)
        x_cats_list = []
        for i, embed_layer in enumerate(self.cat_embeddings):
            x_cats_list.append(embed_layer(cat_data[:, i]))
        x_cats = torch.cat(x_cats_list, dim=1)
        x_cats = self.cat_branch(x_cats)
        x_num = self.num_branch(num_data)
        combined = torch.cat([x_img, x_num, x_cats], dim=1)
        output = self.combiner(combined)
        return output

class TestDemandDataset(Dataset):
    """Custom PyTorch Dataset for test data (no targets)"""
    def __init__(self, df, num_cols, cat_cols):
        self.img_emb = np.stack(df["image_embedding"].values).astype(np.float32)
        self.num_data = df[num_cols].values.astype(np.float32)
        self.cat_data = df[cat_cols].values.astype(np.int64) # Use int64 for embedding lookup

    def __len__(self):
        return len(self.img_emb)

    def __getitem__(self, idx):
        return (
            self.img_emb[idx],
            self.num_data[idx],
            self.cat_data[idx]
        )

# --- 2. Load Test Data and Preprocessors ---
print("Loading test data and preprocessors...")
df_test = pd.read_csv("data/test.csv", sep=";")

with open("models/encoders_nn.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("models/date_mins_nn.pkl", "rb") as f:
    date_mins = pickle.load(f)
with open("models/scaler_nn.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/feature_config_nn.pkl", "rb") as f:
    config = pickle.load(f)

num_cols = config["num_cols"]
cat_cols = config["cat_cols"]
embedding_dim = config["embedding_dim"]

# --- 3. Load Model Structure (MUST match the size of the saved checkpoint) ---
print("Loading model structure...")
# Recalculate embedding sizes using the formula from the training script
cat_embedding_sizes = []
for col in cat_cols:
    # Use the original number of classes from the saved encoder.
    num_classes = len(encoders[col].classes_) 
    
    # CRITICAL: Use the exact embedding dimension formula: num_classes // 2 + 1
    embed_dim = min(50, num_classes // 2 + 1)
    
    cat_embedding_sizes.append((num_classes, embed_dim))

model = MixedInputNet(
    embedding_dim=embedding_dim,
    num_numerical_features=len(num_cols),
    cat_embedding_sizes=cat_embedding_sizes
)
# Load the state dict now that the model structure matches the checkpoint
model.load_state_dict(torch.load("models/nn_model.pth"))
model.to(DEVICE)
model.eval() # Set model to evaluation mode


# --- 4. Preprocess Test Data (Including 'UNKNOWN' handling) ---
print("Preprocessing test data...")

# Drop columns (must match training)
drop_cols = [
    "waist_type", "woven_structure", "knit_structure", "archetype", "heel_shape_type",
    "toecap_type", "weekly_sales", "weekly_demand", "num_week_iso", "year",
    "Unnamed: 28", "Unnamed: 29", "Unnamed: 30", "Unnamed: 31", "Unnamed: 32",
]
df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])
df_test = df_test.groupby("ID").first().reset_index()

# Parse image embeddings (must match training)
df_test["image_embedding"] = df_test["image_embedding"].apply(
    lambda x: np.fromstring(x, sep=",").astype(np.float32)
)
df_test = df_test.drop(columns=["color_rgb"], errors='ignore')

# Dates -> numeric (must match training)
date_cols = ["moment", "phase_in", "phase_out"]
for col in date_cols:
    df_test[col] = pd.to_datetime(df_test[col], dayfirst=True, errors="coerce")
    base = date_mins[col]
    df_test[col] = (df_test[col] - base).dt.days.fillna(0).astype(np.float32)

# Boolean -> int (must match training)
df_test["has_plus_sizes"] = df_test["has_plus_sizes"].astype(int)

# Label encode categorical (CRITICAL: Handle 'UNKNOWN' categories)
for col in cat_cols:
    le = encoders[col]
    df_test[col] = df_test[col].astype(str)

    # The maximum valid index the model can accept is N_train - 1.
    max_valid_index = len(le.classes_) - 1

    # Add 'UNKNOWN' to the encoder's classes if needed for safe transformation.
    if "UNKNOWN" not in le.classes_:
        le.classes_ = np.append(le.classes_, "UNKNOWN") 

    # Apply transformation, replacing unseen values with 'UNKNOWN' first
    df_test[col] = df_test[col].apply(lambda x: x if x in le.classes_ else "UNKNOWN")
    transformed_data = le.transform(df_test[col])
    
    # Clip indices to prevent IndexErrors in PyTorch. 
    # Any new 'UNKNOWN' class index (which is N_train) is forced to the max valid index (N_train - 1).
    df_test[col] = np.clip(transformed_data, a_min=0, a_max=max_valid_index)

# Scale numerical features (must match training)
df_test[num_cols] = scaler.transform(df_test[num_cols])


# --- 5. Create Test Dataset and Loader ---
print("Creating test loader...")
test_dataset = TestDemandDataset(df_test, num_cols, cat_cols)
# Note: num_workers is often safe at 0 or 1 for evaluation
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# --- 6. Prediction ---
print("Running predictions...")
all_preds = []
with torch.no_grad():
    for (img, num, cat) in test_loader:
        img, num, cat = (
            img.to(DEVICE),
            num.to(DEVICE),
            cat.to(DEVICE)
        )
        
        preds = model(img, num, cat)
        all_preds.append(preds.cpu().numpy())

# Concatenate all batches
final_preds = np.concatenate(all_preds).flatten()
# NNs can sometimes predict negative, clip at 0
final_preds[final_preds < 0] = 0

# --- 7. Build Submission ---
df_pred = pd.DataFrame({
    "ID": df_test["ID"].values,
    "Production": final_preds
})

df_pred.to_csv("predictions_nn.csv", index=False)
print("Predictions saved to predictions_nn.csv")
