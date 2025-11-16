# -*- coding: utf-8 -*-
"""
ensemble_mango.py
------------------
Script completo para entrenar un ensemble:
- Modelos base: XGBoost, LightGBM, CatBoost.
- Meta-modelo: Ridge Regression (stacking).
- Preprocesa el dataset de Mango (moda) con las columnas indicadas.
- Genera un CSV de submission con la predicción final del ensemble.

Suposiciones:
- Tienes "train.csv" y "test.csv" en una carpeta "data/".
- La columna target es "Production" (puedes cambiarla a "weekly_demand" si quieres).
- La columna ID se llama "ID" (ajusta si el nombre es distinto).
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ==========================
# CONFIGURACIÓN BÁSICA
# ==========================

TRAIN_PATH = "data/processed_data.csv"
TEST_PATH = "data/test.csv"

# El target que quieres predecir:
#   - "Production" si quieres predecir unidades producidas
#   - "weekly_demand" si quieres predecir demanda semanal
TARGET_COL = "Production"

# Nombre de la columna ID para la submission
ID_COL = "ID"

RANDOM_STATE = 42
VALID_SIZE = 0.2  # 20% del train para validación interna del meta-modelo


# ==========================
# 1. PREPROCESADO
# ==========================

def load_and_preprocess(train_path: str, test_path: str, target_col: str):
    """
    Carga train y test desde CSV y realiza un preprocesado básico:
    - Convierte phase_in y phase_out a datetime.
    - Crea features temporales (mes, semana) y duración de la temporada.
    - Convierte color_rgb ("R,G,B") en 3 columnas numéricas.
    - Elimina columnas que no queremos como features (ID, target, etc.).
    Devuelve: X (train features), y (train target), X_test (test features).
    """

    # 1. Cargar dataframes
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # ------------------------------
    # Drop clearly unusable / empty columns
    drop_cols = [
        "waist_type", "woven_structure", "knit_structure",
        "archetype", "heel_shape_type", "toecap_type",
        "weekly_sales", "num_week_iso", "year"  # keep weekly_demand for target
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ------------------------------
    # Build product-level target: total demand per ID
    # (sum of weekly_demand over the lifecycle)
    if "weekly_demand" not in df.columns:
        raise ValueError("weekly_demand not found in train data; make sure processed_data.csv includes it")

    target_per_id = (
        df.groupby("ID")["weekly_demand"]
        .sum()
        .rename("target_demand")
        .reset_index()
    )

    # Reduce to one row per product (take first row as representative of static features)
    df_first = df.groupby("ID").first().reset_index()

    # Merge target
    df = df_first.merge(target_per_id, on="ID")

    # Drop leftover per-week demand column from features
    df = df.drop(columns=["weekly_demand"])

    # ------------------------------
    # Parse image embeddings
    df["image_embedding"] = df["image_embedding"].apply(
        lambda x: np.fromstring(x, sep=",").astype(np.float32)
    )

    embedding_dim = len(df["image_embedding"].iloc[0])

    # ------------------------------
    # Split RGB
    def parse_rgb(s):
        if isinstance(s, str):
            return list(map(int, s.split(",")))
        return [0, 0, 0]

    # df[["r", "g", "b"]] = df["color_rgb"].apply(parse_rgb).tolist()
    df = df.drop(columns=["color_rgb"])

    # ------------------------------
    # Dates → numeric (store mins so test uses same reference)
    date_cols = ["moment", "phase_in", "phase_out"]
    date_mins = {}

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
        col_min = df[col].min()
        date_mins[col] = col_min
        df[col] = (df[col] - col_min).dt.days.fillna(0).astype(np.float32)

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

    # ------------------------------
    # Boolean → int
    df["has_plus_sizes"] = df["has_plus_sizes"].astype(int)

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
    # Build X, y
    #  - X = [image_embedding || tabular_features]
    #  - y = total demand per product
    emb_matrix = np.stack(df["image_embedding"].values, axis=0)  # shape (N, D)
    tab_matrix = df[feature_cols].to_numpy(dtype=np.float32)

    X = np.hstack([emb_matrix, tab_matrix])  # shape (N, D + num_tab_features)
    y = df["target_demand"].to_numpy(dtype=np.float32)

    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)



# ==========================
# 2. ENTRENAR MODELOS BASE
# ==========================

def train_xgboost(X_tr, y_tr):
    """
    Entrena un XGBRegressor básico.
    """
    model_xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
    )
    model_xgb.fit(X_tr, y_tr)
    return model_xgb


def train_lightgbm(X_tr, y_tr):
    """
    Entrena un LGBMRegressor básico.
    """
    model_lgb = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    model_lgb.fit(X_tr, y_tr)
    return model_lgb


def train_catboost(X_tr, y_tr, X_val, y_val, cat_cols):
    """
    Entrena un CatBoostRegressor usando columnas categóricas.
    Devuelve el modelo entrenado y el best_iteration (para reentrenar luego).
    """
    # Índices de columnas categóricas
    cat_indices = [X_tr.columns.get_loc(c) for c in cat_cols]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)

    model_cat = CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.05,
        n_estimators=1000,
        random_seed=RANDOM_STATE,
        eval_metric="RMSE",
        od_type="Iter",
        od_wait=50,
        verbose=100,
    )

    model_cat.fit(train_pool, eval_set=val_pool, use_best_model=True)

    best_it = model_cat.get_best_iteration()
    if best_it is None or best_it <= 0:
        best_it = 1000  # por si acaso

    return model_cat, best_it, cat_indices


# ==========================
# 3. STACKING (META-MODELO)
# ==========================

def build_meta_features(models, X, X_is_pool=False, cat_indices=None):
    """
    Genera las features para el meta-modelo a partir de las predicciones de:
    - models["xgb"]
    - models["lgb"]
    - models["cat"]

    Si X_is_pool=True, asume que X es un Pool para CatBoost.
    Si X_is_pool=False, crea el Pool internamente si se proporcionan cat_indices.
    """
    model_xgb = models["xgb"]
    model_lgb = models["lgb"]
    model_cat = models["cat"]

    # Predicciones XGBoost y LGBM (usan numpy/pandas directamente)
    pred_xgb = model_xgb.predict(X)
    pred_lgb = model_lgb.predict(X)

    # Predicciones CatBoost (necesita Pool si hay categóricas)
    if X_is_pool:
        pred_cat = model_cat.predict(X)
    else:
        if cat_indices is None:
            raise ValueError("Necesitas cat_indices para crear el Pool de CatBoost")
        pool = Pool(X, cat_features=cat_indices)
        pred_cat = model_cat.predict(pool)

    # Matriz (n_samples, 3)
    X_meta = np.column_stack([pred_xgb, pred_lgb, pred_cat])
    return X_meta


# ==========================
# 4. MAIN: FLUJO COMPLETO
# ==========================

def main():
  # ------------------------------
    # 4.1. Cargar y preprocesar
    # ------------------------------
    print("Cargando y preprocesando datos...")
    X, y, X_test, train_df, test_df = load_and_preprocess(
        TRAIN_PATH, TEST_PATH, TARGET_COL
    )

    print(f"Shape X train: {X.shape}, X test: {X_test.shape}")

    # Columnas categóricas para CatBoost
    cat_cols = get_categorical_columns(X)
    print(f"Columnas categóricas usadas en CatBoost: {cat_cols}")
    cat_indices = [X.columns.get_loc(c) for c in cat_cols]

    # ------------------------------
    # 4.2. Split train / valid interno
    # ------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VALID_SIZE, random_state=RANDOM_STATE
    )

    print(f"Train interno: {X_tr.shape}, Valid interno: {X_val.shape}")

    # ------------------------------
    # 4.3. Entrenar modelos base
    # ------------------------------
    print("\nEntrenando XGBoost...")
    model_xgb = train_xgboost(X_tr, y_tr)

    print("\nEntrenando LightGBM...")
    model_lgb = train_lightgbm(X_tr, y_tr)

    print("\nEntrenando CatBoost...")
    model_cat, best_it, cat_indices_tr = train_catboost(
        X_tr, y_tr, X_val, y_val, cat_cols
    )
    print(f"Best iteration de CatBoost: {best_it}")

    # ------------------------------
    # 4.4. Evaluar modelos base en valid
    # ------------------------------
    print("\nEvaluando modelos base en conjunto de validación...")

    # XGBoost
    val_pred_xgb = model_xgb.predict(X_val)
    rmse_xgb = mean_squared_error(y_val, val_pred_xgb, squared=False)
    print(f"RMSE XGBoost: {rmse_xgb:.4f}")

    # LightGBM
    val_pred_lgb = model_lgb.predict(X_val)
    rmse_lgb = mean_squared_error(y_val, val_pred_lgb, squared=False)
    print(f"RMSE LightGBM: {rmse_lgb:.4f}")

    # CatBoost
    val_pool = Pool(X_val, y_val, cat_features=cat_indices_tr)
    val_pred_cat = model_cat.predict(val_pool)
    rmse_cat = mean_squared_error(y_val, val_pred_cat, squared=False)
    print(f"RMSE CatBoost: {rmse_cat:.4f}")

    # ------------------------------
    # 4.5. Construir X_meta_train y entrenar meta-modelo (Ridge)
    # ------------------------------
    print("\nConstruyendo features para meta-modelo...")

    base_models = {
        "xgb": model_xgb,
        "lgb": model_lgb,
        "cat": model_cat,
    }

    X_meta_train = build_meta_features(
        base_models,
        X_val,
        X_is_pool=True,  # ya hemos creado el Pool antes (val_pool)
        cat_indices=None,  # no hace falta porque pasamos Pool
    )
    y_meta_train = y_val

    print("Entrenando meta-modelo (Ridge)...")
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta_train, y_meta_train)

    meta_pred_val = meta_model.predict(X_meta_train)
    rmse_meta = mean_squared_error(y_val, meta_pred_val, squared=False)
    print(f"RMSE Ensemble (meta-modelo) en validación: {rmse_meta:.4f}")

    # ------------------------------
    # 4.6. Reentrenar modelos base en TODO el train
    # ------------------------------
    print("\nReentrenando modelos base en TODO el train...")

    # XGBoost full
    model_xgb_full = train_xgboost(X, y)

    # LightGBM full
    model_lgb_full = train_lightgbm(X, y)

    # CatBoost full con best_iteration
    pool_full = Pool(X, y, cat_features=cat_indices)

    model_cat_full = CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.05,
        n_estimators=best_it,
        random_seed=RANDOM_STATE,
        eval_metric="RMSE",
        verbose=0,
    )
    model_cat_full.fit(pool_full)

    full_models = {
        "xgb": model_xgb_full,
        "lgb": model_lgb_full,
        "cat": model_cat_full,
    }

    # ------------------------------
    # 4.7. Construir meta-features para TODO el train y para test
    # ------------------------------
    print("\nConstruyendo meta-features para TODO el train y para test...")

    # Train completo
    X_meta_full_train = build_meta_features(
        full_models,
        X,
        X_is_pool=False,
        cat_indices=cat_indices,
    )
    y_meta_full_train = y

    # Test
    X_meta_test = build_meta_features(
        full_models,
        X_test,
        X_is_pool=False,
        cat_indices=cat_indices,
    )

    # ------------------------------
    # 4.8. Entrenar meta-modelo final y predecir test
    # ------------------------------
    print("Entrenando meta-modelo FINAL en todo el train...")

    meta_model_final = Ridge(alpha=1.0)
    meta_model_final.fit(X_meta_full_train, y_meta_full_train)

    final_test_pred = meta_model_final.predict(X_meta_test)

    # ------------------------------
    # 4.9. Guardar submission
    # ------------------------------
    print("\nGuardando submission_ensemble.csv ...")

    if ID_COL not in test_df.columns:
        raise ValueError(f"La columna ID '{ID_COL}' no está en test.csv")

    submission = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        TARGET_COL: final_test_pred,
    })

    submission.to_csv("submission_ensemble.csv", index=False)
    print("Listo. Archivo generado: submission_ensemble.csv")


if __name__ == "__main__":
    main()