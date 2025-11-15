# entrena modelo principal y guarda .pkl
"""
train_model.py
- Carga el dataset a nivel producto de train.
- Aplica feature engineering.
- Realiza un split de validación (por temporada).
- Entrena un modelo (XGBoost / LightGBM / CatBoost).
- Guarda el modelo entrenado y cualquier artefacto necesario (PCA, encoder).
"""

import pickle
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor  # o LightGBM/CatBoost

import config as cfg
from features import (
    parse_image_embedding_column,
    apply_pca_to_embeddings,
    add_basic_ratios,
    encode_categoricals_one_hot,
)
from evaluate import train_val_split_by_season, print_regression_metrics


def main() -> None:
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar train a nivel producto
    df = pd.read_parquet(cfg.TRAIN_PRODUCTS_PATH)

    # 2. Parsear embeddings (si hace falta)
    df = parse_image_embedding_column(df, col="image_embedding")

    # 3. Aplicar PCA a embeddings
    df, pca_model = apply_pca_to_embeddings(
        df, col="image_embedding", n_components=20, fit_pca=True
    )

    # 4. Añadir ratios básicos y demás features
    df = add_basic_ratios(df)

    # 5. Elegir columnas categóricas a codificar
    categorical_cols = [
        "aggregated_family",
        "family",
        "category",
        "fabric",
        "color_name",
        "length_type",
        "silhouette_type",
        "waist_type",
        "sleeve_length_type",
        "archetype",
        "moment",
        "ocassion",
        # Añadir las que veáis útiles
    ]

    # Filtrar por si alguna no existe
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # 6. Separar features numéricas + categóricas
    df_encoded, encoder = encode_categoricals_one_hot(df, categorical_cols)

    # 7. Separar X, y
    y = df_encoded[cfg.TARGET_COL].values
    X = df_encoded.drop(columns=[cfg.TARGET_COL])

    # 8. Split train/val por temporada
    X_train, X_val, y_train, y_val = train_val_split_by_season(
        X, y, seasons=df_encoded[cfg.SEASON_COL]
    )

    # 9. Definir modelo
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=cfg.RANDOM_STATE,
        n_jobs=cfg.N_JOBS,
        tree_method="hist",
    )

    # 10. Entrenar
    model.fit(X_train, y_train)

    # 11. Evaluar en validación
    y_val_pred = model.predict(X_val)
    print_regression_metrics(y_val, y_val_pred, label="Validación")

    # 12. Guardar modelo y artefactos (PCA + encoder + columnas)
    artifacts = {
        "model": model,
        "pca": pca_model,
        "encoder": encoder,
        "feature_cols": X.columns.tolist(),
    }

    with open(cfg.XGB_MODEL_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Modelo y artefactos guardados en: {cfg.XGB_MODEL_PATH}")


if __name__ == "__main__":
    main()
