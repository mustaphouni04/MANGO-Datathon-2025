# carga, limpieza básica, guardado en interim
"""
data_prep.py
- Carga los CSV crudos (train/test) desde data/raw.
- Realiza limpieza mínima y normaliza tipos de datos.
- Guarda versiones "weekly" limpias en data/interim como parquet.
"""

import pandas as pd
from pathlib import Path
import config as cfg


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee train.csv y test.csv desde data/raw."""
    train = pd.read_csv(cfg.TRAIN_RAW_PATH)
    test = pd.read_csv(cfg.TEST_RAW_PATH)
    return train, test


def basic_cleaning(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Limpieza muy básica:
    - Conversión de fechas si hace falta.
    - Asegurar tipos numéricos en columnas clave.
    - Eliminar duplicados si los hubiera.
    - (Añadir aquí lo que vayáis descubriendo en el EDA).
    """
    # TODO: convertir phase_in/phase_out a datetime si son strings
    # df["phase_in"] = pd.to_datetime(df["phase_in"])
    # df["phase_out"] = pd.to_datetime(df["phase_out"])

    # Ejemplo: asegurar que num_stores y num_sizes son numéricos
    # numeric_cols = ["num_stores", "num_sizes", "price"]
    # for col in numeric_cols:
    #     df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eliminar duplicados por seguridad
    df = df.drop_duplicates()

    # TODO: cualquier otra limpieza que veáis necesaria
    return df


def main() -> None:
    """Pipeline principal de preparación básica de datos semanales."""
    cfg.INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_raw_data()

    train_clean = basic_cleaning(train_raw, is_train=True)
    test_clean = basic_cleaning(test_raw, is_train=False)

    # Guardamos como parquet para que sea más rápido de leer luego
    train_clean.to_parquet(cfg.TRAIN_WEEKLY_PATH, index=False)
    test_clean.to_parquet(cfg.TEST_WEEKLY_PATH, index=False)

    print(f"Guardado train weekly en: {cfg.TRAIN_WEEKLY_PATH}")
    print(f"Guardado test weekly en: {cfg.TEST_WEEKLY_PATH}")


if __name__ == "__main__":
    main()
