# ============================
# Configurações gerais do projeto
# ============================

# Reprodutibilidade
RANDOM_STATE = 42

# Dataset
DATASET_NAME = "Airline Delay Cause"

# Colunas principais
TARGET_COL = "atraso_medio_por_voo"

ID_COLS = [
    "year",
    "month",
    "carrier",
    "airport",
]

# Colunas numéricas básicas (antes de feature engineering)
NUMERIC_COLS = [
    "arr_flights",
    "arr_del15",
    "arr_cancelled",
    "arr_diverted",
]

# Horizonte de previsão (em meses)
FORECAST_HORIZON = 1
