# eric validation
import pandas as pd
import numpy as np
import pickle
from pandas.core.indexes.period import validate_dtype_freq
import xgboost as xgb

# ------------------------------
# Load test data
df_test = pd.read_csv("data/test.csv", sep=";")

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

bst = xgb.Booster()
bst.load_model("models/xgb_mango.json")

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
# XGBoost prediction
dtest = xgb.DMatrix(X_test)
test_preds = bst.predict(dtest)


#######################################################
##################### TRAINING XGB ####################
#######################################################
# ------------------------------

df_train = pd.read_csv("data/train.csv", sep=";")
df_train = df_train.drop(columns=[c for c in drop_cols if c in df_test.columns])
df_train = df_train.groupby("ID").first().reset_index()

with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("models/date_mins.pkl", "rb") as f:
    date_mins = pickle.load(f)

with open("models/feature_config.pkl", "rb") as f:
    config = pickle.load(f)

feature_cols = config["feature_cols"]
embedding_dim = config["embedding_dim"]

bst = xgb.Booster()
bst.load_model("models/xgb_mango.json")

# ------------------------------
# Parse image embeddings
df_train["image_embedding"] = df_train["image_embedding"].apply(
    lambda x: np.fromstring(x, sep=",").astype(np.float32)
)

# Optional sanity check:
if len(df_train["image_embedding"].iloc[0]) != embedding_dim:
    raise ValueError(
        f"Embedding dimension mismatch: train={embedding_dim}, "
        f"test={len(df_train['image_embedding'].iloc[0])}"
    )

# df_test[["r", "g", "b"]] = df_test["color_rgb"].apply(parse_rgb).tolist()
df_train = df_train.drop(columns=["color_rgb"])

# ------------------------------
# Dates → numeric using SAME minima as training
date_cols = ["moment", "phase_in", "phase_out"]
for col in date_cols:
    df_train[col] = pd.to_datetime(df_train[col], dayfirst=True, errors="coerce")
    base = date_mins[col]
    df_train[col] = (df_train[col] - base).dt.days.fillna(0).astype(np.float32)

# ------------------------------
# Label encode categorical columns with train encoders
cat_cols = list(encoders.keys())

for col in cat_cols:
    le = encoders[col]
    df_train[col] = df_train[col].astype(str)

    # Map unseen categories to "UNKNOWN"
    # If "UNKNOWN" wasn't in training, we append it to classes_
    if "UNKNOWN" not in le.classes_:
        le.classes_ = np.append(le.classes_, "UNKNOWN")

    df_train[col] = df_train[col].apply(lambda x: x if x in le.classes_ else "UNKNOWN")
    df_train[col] = le.transform(df_train[col])

# ------------------------------
# Boolean → int
df_train["has_plus_sizes"] = df_train["has_plus_sizes"].astype(int)

# ------------------------------
# Build X_test: [image_embedding || tabular_features]
missing = [c for c in feature_cols if c not in df_train.columns]
if missing:
    raise ValueError(f"Missing feature columns in test data: {missing}")

emb_matrix_train = np.stack(df_train["image_embedding"].values, axis=0)
tab_matrix_train = df_train[feature_cols].to_numpy(dtype=np.float32)

X_train = np.hstack([emb_matrix_train, tab_matrix_train])

# ------------------------------
# XGBoost prediction
dtrain = xgb.DMatrix(X_train)
train_preds = bst.predict(dtrain)


#######################################################
##################### KNN TRAIN #######################
#######################################################

# create total_demand column
df_demand = df_train["life_cycle_length"] * df_train["weekly_demand"]

def str_emb_to_array_emb(emb_str):
    print(emb_str)
    return np.array(list(map(float, emb_str.split(","))))

def list_to_array(lst):
    return np.array(lst)

from sklearn.neighbors import NearestNeighbors

X_train_emb = df_train["image_embedding"].values

X_train_emb = np.array(list(map(list_to_array, X_train_emb)))
y_train = df_demand.values  # demanda real total (target)

# Creamos el índice kNN sobre los embeddings del train
knn = NearestNeighbors(
    n_neighbors=10,        # k vecinos
    metric="cosine",       # similitud de coseno
	n_jobs=-1)
knn.fit(X_train_emb) # modelo entrenado

###########
##### KNN TRAIN
#####################

# X_val_emb = np.array(list(map(list_to_array, X_val_emb)))
distances, indices = knn.kneighbors(X_train_emb, n_neighbors=10)
similarities = 1.0 - distances  # porque metric="cosine" devuelve distancia

# Evitamos divisiones por cero
similarities = np.clip(similarities, a_min=1e-6, a_max=None)
# Normalizamos para que las similitudes de cada fila sumen 1
weights = similarities / similarities.sum(axis=1, keepdims=True)
# Obtenemos las demandas reales de los vecinos
neighbor_targets = y_train[indices]  # shape (n_val, k)
# Predicción kNN = media ponderada de las demandas de los vecinos
knn_pred_train = (weights * neighbor_targets).sum(axis=1)  # shape (n_val,)

from sklearn.metrics import mean_absolute_error  # o la que use el datathon

boost_pred = train_preds

alphas = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, 0.10, ..., 1.0
best_alpha = None
best_score = np.inf

y_val_true = df_demand.values  # demanda real en validación

for alpha in alphas:
    final_pred = alpha * boost_pred + (1 - alpha) * knn_pred_train
    score = mean_absolute_error(y_val_true, final_pred)  # o métrica oficial
    if score < best_score:
        best_score = score
        best_alpha = alpha

print("Best alpha:", best_alpha, "with score:", best_score)

#######################################################
##################### KNN  TRAIN #######################
#######################################################
X_val_emb = df_test["image_embedding"].values
X_val_emb = np.array(list(map(list_to_array, X_val_emb)))
distances, indices = knn.kneighbors(X_val_emb, n_neighbors=10)
similarities = 1.0 - distances  # porque metric="cosine" devuelve distancia

# Evitamos divisiones por cero
similarities = np.clip(similarities, a_min=1e-6, a_max=None)
# Normalizamos para que las similitudes de cada fila sumen 1
weights = similarities / similarities.sum(axis=1, keepdims=True)
# Obtenemos las demandas reales de los vecinos
neighbor_targets = y_train[indices]  # shape (n_val, k)
# Predicción kNN = media ponderada de las demandas de los vecinos
knn_pred_test = (weights * neighbor_targets).sum(axis=1)  # shape (n_val,)



############################
###### FINAL PREDICTION ####
############################
final_pred_test = best_alpha * test_preds + (1 - best_alpha) * knn_pred_test
# raise 10% to all values
final_pred_test = final_pred_test * 1.1

# ------------------------------
# Build submission
df_pred = pd.DataFrame({
    "ID": df_test["ID"].values,
    "Production": final_pred_test  # full-season demand estimate
})

df_pred.to_csv("predictions_knn.csv", index=False)
print("Predictions saved to predictions_xgb.csv")





