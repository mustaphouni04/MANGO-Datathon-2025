# pasar de weekly a product-level (train/test)
"""
aggregate_by_product.py
- Convierte los datos semanales (una fila por semana e ID)
  en datos a nivel producto (una fila por ID).
- Agrega demanda/ventas semanales y selecciona columnas estáticas.
"""

import pandas as pd
import numpy as np
import config as cfg


def aggregate_train(train_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Crea un DataFrame con una fila por ID para train.
    Incluye:
    - Features agregadas a partir de weekly_demand / weekly_sales.
    - Variables estáticas de producto (family, category, etc.).
    - Columna target (demanda total u otra variable elegida).
    """
    id_col = cfg.ID_COL

    # Agregados numéricos por producto
    agg_dict = {
        cfg.WEEKLY_DEMAND_COL: ["sum", "mean", "max"],
        cfg.WEEKLY_SALES_COL: ["sum", "mean", "max"],
        "num_week_iso": ["nunique"],
    }

    agg_df = train_weekly.groupby(id_col).agg(agg_dict)
    # Renombrar columnas agregadas
    agg_df.columns = [
        f"{col}_{stat}" for col, stat in agg_df.columns.to_flat_index()
    ]
    agg_df = agg_df.reset_index()

    # Renombrar el target principal (por ejemplo, suma de weekly_demand)
    agg_df.rename(
        columns={f"{cfg.WEEKLY_DEMAND_COL}_sum": cfg.TARGET_COL},
        inplace=True,
    )

    # Seleccionar columnas estáticas de producto (tomamos la primera ocurrencia)
    static_cols = [id_col, cfg.SEASON_COL] + cfg.STATIC_PRODUCT_COLS
    static_df = (
        train_weekly[static_cols]
        .drop_duplicates(subset=id_col)
    )

    # Merge de agregados numéricos con columnas estáticas
    train_products = static_df.merge(agg_df, on=id_col, how="left")

    return train_products


def aggregate_test(test_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Versión para test: no hay target semanal real.
    - Agregamos si hay variables numéricas semanales útiles.
    - Añadimos columnas estáticas de producto.
    """
    id_col = cfg.ID_COL

    # Ejemplo: si test también tiene weekly_sales (depende del dataset)
    # podéis agregar cosas similares. Si no, podéis omitir.
    agg_df = (
        test_weekly
        .groupby(id_col)["num_week_iso"]
        .nunique()
        .reset_index()
        .rename(columns={"num_week_iso": "n_weeks"})
    )

    static_cols = [id_col, cfg.SEASON_COL] + cfg.STATIC_PRODUCT_COLS
    static_df = (
        test_weekly[static_cols]
        .drop_duplicates(subset=id_col)
    )

    test_products = static_df.merge(agg_df, on=id_col, how="left")

    return test_products


def main() -> None:
    """Pipeline principal para generar train_products y test_products."""
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_weekly = pd.read_parquet(cfg.TRAIN_WEEKLY_PATH)
    test_weekly = pd.read_parquet(cfg.TEST_WEEKLY_PATH)

    train_products = aggregate_train(train_weekly)
    test_products = aggregate_test(test_weekly)

    train_products.to_parquet(cfg.TRAIN_PRODUCTS_PATH, index=False)
    test_products.to_parquet(cfg.TEST_PRODUCTS_PATH, index=False)

    print(f"Guardado train_products en: {cfg.TRAIN_PRODUCTS_PATH}")
    print(f"Guardado test_products en: {cfg.TEST_PRODUCTS_PATH}")


if __name__ == "__main__":
    main()
