# funciones de métricas internas
"""
evaluate.py
- Funciones para hacer splits de train/validación
  y calcular métricas sencillas (MAE, RMSE, etc.).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import config as cfg


def train_val_split_by_season(
    X: pd.DataFrame,
    y: np.ndarray,
    seasons: pd.Series,
    val_season: int | None = None,
):
    """
    Split de train/validación utilizando la columna de temporada.
    - Si val_season es None, toma la última temporada como validación.
    """
    unique_seasons = sorted(seasons.unique())

    if val_season is None:
        val_season = unique_seasons[-1]

    train_mask = seasons != val_season
    val_mask = seasons == val_season

    X_train = X.loc[train_mask]
    X_val = X.loc[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]

    print(f"Temporadas de train: {[s for s in unique_seasons if s != val_season]}")
    print(f"Temporada de validación: {val_season}")
    print(f"Tamaño train: {X_train.shape}, tamaño val: {X_val.shape}")

    return X_train, X_val, y_train, y_val


def print_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label: str = ""
) -> None:
    """Imprime MAE y RMSE para inspección rápida."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print(f"=== Métricas {label} ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
