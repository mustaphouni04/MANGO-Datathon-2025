# todas las funciones de feature engineering
  # (ratios, encoding categóricas, embeddings PCA, etc.)
"""
features.py
- Funciones puras de feature engineering.
- Aquí añadimos:
  - ratios numéricos,
  - combinaciones de columnas,
  - encoding de categóricas,
  - reducción de embeddings, etc.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA


def parse_image_embedding_column(df: pd.DataFrame, col: str = "image_embedding") -> pd.DataFrame:
    """
    Convierte la columna 'image_embedding' de string/list a array numérico.
    Supone que el embedding viene como lista en string (ej. '[0.1, 0.2, ...]').
    Ajustar según el formato real.
    """
    # TODO: adaptar al formato exacto de image_embedding
    def parse(x):
        if isinstance(x, str):
            try:
                # Ej: "[0.1, 0.2, ...]" → lista de floats
                vals = x.strip("[]").split(",")
                return np.array([float(v) for v in vals])
            except Exception:
                return np.nan
        return x

    df[col] = df[col].apply(parse)
    return df


def apply_pca_to_embeddings(
    df: pd.DataFrame,
    col: str = "image_embedding",
    n_components: int = 20,
    fit_pca: bool = True,
    pca_model: PCA | None = None,
) -> tuple[pd.DataFrame, PCA]:
    """
    Aplica PCA a los embeddings para reducir dimensión.
    - Si fit_pca=True, entrena un nuevo PCA.
    - Si fit_pca=False, usa el PCA pasado (para test).
    Devuelve (df_modificado, pca_model).
    """
    # Filtrar filas con embedding no nulo
    emb_matrix = np.stack(df[col].dropna().values)
    if fit_pca:
        pca_model = PCA(n_components=n_components, random_state=42)
        emb_reduced = pca_model.fit_transform(emb_matrix)
    else:
        assert pca_model is not None, "Se requiere un PCA ya entrenado."
        emb_reduced = pca_model.transform(emb_matrix)

    # Crear un DataFrame auxiliar para pegar los componentes reducidos
    emb_df = pd.DataFrame(
        emb_reduced,
        index=df[col].dropna().index,
        columns=[f"emb_pca_{i+1}" for i in range(emb_reduced.shape[1])],
    )

    df = df.join(emb_df)

    return df, pca_model


def add_basic_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade ratios y combinaciones numéricas simples.
    Ajustar según las columnas agregadas disponibles.
    """
    # Ejemplo: ratios de ventas/demanda
    sales_sum_col = "weekly_sales_sum"
    demand_sum_col = "weekly_demand_sum"

    if sales_sum_col in df.columns and demand_sum_col in df.columns:
        df["sales_over_demand"] = df[sales_sum_col] / (df[demand_sum_col] + 1e-6)

    # Alcance potencial: tiendas * tallas
    if "num_stores" in df.columns and "num_sizes" in df.columns:
        df["stores_times_sizes"] = df["num_stores"] * df["num_sizes"]

    # TODO: añadir aquí más ratios que se os ocurran

    return df


def encode_categoricals_one_hot(
    df: pd.DataFrame, categorical_cols: list[str]
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """
    Aplica OneHotEncoder a las columnas categóricas.
    Devuelve:
    - df con columnas numéricas nuevas
    - el encoder (para reutilizar en test)
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_data = encoder.fit_transform(df[categorical_cols])

    cat_df = pd.DataFrame(
        cat_data,
        index=df.index,
        columns=encoder.get_feature_names_out(categorical_cols),
    )

    df = pd.concat([df.drop(columns=categorical_cols), cat_df], axis=1)
    return df, encoder


def apply_encoder_to_test(
    df: pd.DataFrame, categorical_cols: list[str], encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Aplica un OneHotEncoder ya entrenado a un DataFrame de test.
    """
    cat_data = encoder.transform(df[categorical_cols])
    cat_df = pd.DataFrame(
        cat_data,
        index=df.index,
        columns=encoder.get_feature_names_out(categorical_cols),
    )

    df = pd.concat([df.drop(columns=categorical_cols), cat_df], axis=1)
    return df
