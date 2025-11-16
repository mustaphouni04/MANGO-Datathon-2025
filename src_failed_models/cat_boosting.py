import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

# 1. Cargar datos
df = pd.read_csv("data/processed_data.csv")

# 2. Construir TARGET como suma de weekly_demand por producto (ID)
if "weekly_demand" not in df.columns:
    raise ValueError("weekly_demand not found in train data; make sure processed_data.csv includes it")

target_per_id = (
    df.groupby("ID")["weekly_demand"]
      .sum()
      .rename("target_demand")
      .reset_index()
)

# Reducir a una fila por producto para las features (tomar la primera fila como representativa)
df_first = df.groupby("ID").first().reset_index()

# Unir el objetivo
df = df_first.merge(target_per_id, on="ID")

# Elegir y
y = df["target_demand"]

# 3. Preprocesado básico
# # Fechas (si vienen como string)
for col in ["phase_in", "phase_out"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Crear algunas features temporales simples
if "phase_in" in df.columns:
    df["phase_in_month"] = df["phase_in"].dt.month
    df["phase_in_week"] = df["phase_in"].dt.isocalendar().week.astype("Int64")

if "phase_out" in df.columns:
    df["phase_out_month"] = df["phase_out"].dt.month
    df["phase_out_week"] = df["phase_out"].dt.isocalendar().week.astype("Int64")

if {"phase_in", "phase_out"} <= set(df.columns):
    df["season_length_weeks"] = (
        (df["phase_out"] - df["phase_in"]).dt.days / 7.0
    )

# (Opcional) Eliminar columnas que ya no queremos directamente
drop_cols = [
    "Production",
    "ID",          # si es solo identificador
    "phase_in",
    "phase_out",
    "target_demand",
	"color_rgb",
	"image_embedding",
	"life_cycle_length",
	"num_stores",
	"num_sizes",
	"has_plus_sizes",
	"price",
	"year",
	"num_week_iso",
	"weekly_sales",
	"weekly_demand",
	"weekly_production_per_store",
]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)

# 4. Definir features categóricas
cat_cols = [
    "id_season",
    "aggregated_family",
    "family",
    "category",
    "fabric",
    "color_name",
    "length_type",
    "silhouette_type",
    "waist_type",
    "sleeve_length_type",
    "heel_shape_type",
    "toecap_type",
    "woven_structure",
    "knit_structure",
    "print_type",
    "archetype",
    "moment",
    "ocassion",
]

# Detectar columnas no numéricas y asegurarlas como strings sin NaN
obj_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
base_cat_cols = [c for c in cat_cols if c in X.columns]
cols_to_stringify = sorted(set(obj_cols + base_cat_cols))
for c in cols_to_stringify:
    X[c] = X[c].fillna("NA").astype(str)

# Lista final de columnas categóricas a pasar a CatBoost (por nombre)
cat_features_all = cols_to_stringify

# Persist exact feature names and categorical columns used for CatBoost
feature_names_for_training = list(X.columns)
cat_features_for_training = list(cat_features_all)

# 5. Train / validation split
#    (Para algo más serio mejor hacer split por temporada)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cat_features_present = [c for c in cat_features_all if c in X_train.columns]
train_pool = Pool(X_train, y_train, cat_features=cat_features_present)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features_present)

# 6. Definir y entrenar CatBoost
model = CatBoostRegressor(
    loss_function="RMSE",    # para empezar; luego puedes probar Quantile
    depth=8,
    learning_rate=0.05,
    n_estimators=1000,
    random_seed=42,
    eval_metric="RMSE",
    od_type="Iter",          # early stopping
    od_wait=50,
    verbose=100,
)

model.fit(
    train_pool,
    eval_set=valid_pool,
    use_best_model=True,
)

# 7. Evaluar y guardar
from sklearn.metrics import mean_squared_error
import numpy as np

pred_valid = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
print(f"Valid RMSE: {rmse:.4f}")

# ------------------------------
# Define feature columns (tabular, not including image_embedding)
feature_cols = [
    "id_season",
    "aggregated_family", "family", "category", "fabric", "color_name",
    "length_type", "silhouette_type", "neck_lapel_type", "sleeve_length_type", "print_type",
    "moment", "phase_in", "phase_out",
    "life_cycle_length", "num_stores", "num_sizes",
    "has_plus_sizes", "price"
]

# Safety check all feature columns exist
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns in train data: {missing}")
	# ------------------------------
# XGBoost: quantile regression to penalize under-prediction more
# NOTE: requires xgboost >= 1.7 for "reg:quantileerror".
quantile_alpha = 0.6  # >0.5 means preferring over-production vs under-production


# ------------------------------
# Parse image embeddings
df["image_embedding"] = df["image_embedding"].apply(
    lambda x: np.fromstring(x, sep=",").astype(np.float32)
)

embedding_dim = len(df["image_embedding"].iloc[0])

# ------------------------------
# Label encode categorical columns
cat_cols = [
    "aggregated_family", "family", "category", "fabric", "color_name",
    "length_type", "silhouette_type", "neck_lapel_type", "sleeve_length_type", "print_type"
]

encoders = {}
for col in cat_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


params = {
    "objective": "reg:quantileerror",   # if this fails, fall back to "reg:squarederror"
    "quantile_alpha": quantile_alpha,
    "eval_metric": "mae",
    "tree_method": "hist",              # or "gpu_hist" if you have GPU
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "max_bin": 512,
}


# ------------------------------
# Save model + preprocessors
model.save_model("models/cat_boosting.json")

with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("models/feature_config.pkl", "wb") as f:
    pickle.dump(
        {
            "feature_cols": feature_cols,
            "embedding_dim": embedding_dim,
            "quantile_alpha": quantile_alpha,
            "catboost_feature_names": feature_names_for_training,
            "catboost_cat_features": cat_features_for_training,
        },
        f,
    )

print("Training finished. Model and preprocessors saved.")
