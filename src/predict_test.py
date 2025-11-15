# carga modelo + test_products → genera submission
"""
predict_test.py
- Carga el modelo entrenado y los artefactos (PCA, encoder).
- Carga el dataset de test a nivel producto.
- Aplica las mismas transformaciones de features que en train.
- Genera predicciones para cada ID.
- Aplica, si queréis, un factor de calibración (para evitar subestimar).
- Guarda el CSV de submission en submissions/.
"""

import pickle
from datetime import datetime

import pandas as pd

import config as cfg
from features import (
    parse_image_embedding_column,
    apply_pca_to_embeddings,
    add_basic_ratios,
    apply_encoder_to_test,
)


def main(calibration_factor: float = 1.10) -> None:
    cfg.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar artefactos (modelo, PCA, encoder, feature_cols)
    with open(cfg.XGB_MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)

    model = artifacts["model"]
    pca_model = artifacts["pca"]
    encoder = artifacts["encoder"]
    feature_cols = artifacts["feature_cols"]

    # 2. Cargar test_products
    df_test = pd.read_parquet(cfg.TEST_PRODUCTS_PATH)

    # 3. Parsear embeddings
    df_test = parse_image_embedding_column(df_test, col="image_embedding")

    # 4. Aplicar PCA usando el mismo modelo
    df_test, _ = apply_pca_to_embeddings(
        df_test, col="image_embedding", n_components=pca_model.n_components_, fit_pca=False, pca_model=pca_model
    )

    # 5. Añadir ratios
    df_test = add_basic_ratios(df_test)

    # 6. Aplicar encoder one-hot a categóricas (mismas columnas que en train)
    #   IMPORTANTE: deben ser las mismas que usasteis en train_model
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
    ]
    categorical_cols = [c for c in categorical_cols if c in df_test.columns]

    df_test_encoded = apply_encoder_to_test(df_test, categorical_cols, encoder)

    # 7. Asegurar que las columnas de features coinciden con las de train
    #    (si falta alguna, la añadimos con 0; si sobra, la quitamos)
    for col in feature_cols:
        if col not in df_test_encoded.columns:
            df_test_encoded[col] = 0.0

    X_test = df_test_encoded[feature_cols]

    # 8. Predecir
    y_pred = model.predict(X_test)

    # 9. Aplicar factor de calibración (opcional, ajustable)
    y_pred_adj = y_pred * calibration_factor

    # 10. Crear DataFrame de submission
    submission = pd.DataFrame({
        "ID": df_test[cfg.ID_COL].values,
        "TARGET": y_pred_adj,
    })

    # 11. Guardar CSV con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = cfg.SUBMISSIONS_DIR / f"submission_xgb_{timestamp}.csv"
    submission.to_csv(out_path, index=False)

    print(f"Submission guardada en: {out_path}")


if __name__ == "__main__":
    main()
