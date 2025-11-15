# rutas de ficheros, lista de columnas, etc.
"""
config.py
- Centraliza rutas de datos, nombres de columnas y parámetros globales.
- Así evitamos "strings mágicos" repartidos por todo el repo.
"""

from pathlib import Path

# =========
# RUTAS
# =========

# Directorio raíz del proyecto (ajustar si es necesario)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Ficheros de datos
TRAIN_RAW_PATH = RAW_DIR / "train.csv"
TEST_RAW_PATH = RAW_DIR / "test.csv"
SAMPLE_SUB_PATH = RAW_DIR / "sample_submission.csv"

TRAIN_WEEKLY_PATH = INTERIM_DIR / "train_weekly.parquet"
TEST_WEEKLY_PATH = INTERIM_DIR / "test_weekly.parquet"

TRAIN_PRODUCTS_PATH = PROCESSED_DIR / "train_products.parquet"
TEST_PRODUCTS_PATH = PROCESSED_DIR / "test_products.parquet"

# Ficheros de modelo
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.pkl"

# =========
# COLUMNAS
# =========

ID_COL = "ID"
SEASON_COL = "id_season"

TARGET_COL = "total_weekly_demand"  # o "Production" si preferís
WEEKLY_DEMAND_COL = "weekly_demand"
WEEKLY_SALES_COL = "weekly_sales"

# Columnas de producto que no cambian por semana (se agregan por ID)
STATIC_PRODUCT_COLS = [
    "aggregated_family",
    "family",
    "category",
    "fabric",
    "color_name",
    "color_rgb",
    "image_embedding",
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
    "phase_in",
    "phase_out",
    "life_cycle_length",
    "num_stores",
    "num_sizes",
    "has_plus_size",
    "price",
    "year",
]

# Columnas temporales semanales
TIME_COLS = ["num_week_iso"]

# Columnas que usaremos como target semanal (para agregarlas)
AGG_WEEKLY_COLS = [WEEKLY_DEMAND_COL, WEEKLY_SALES_COL]

# Parámetros del modelo (podéis modificar/usar o no)
RANDOM_STATE = 42
N_JOBS = -1
