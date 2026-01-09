RANDOM_STATE = 42

# Problema
PROBLEM_TYPE = "classification"
TARGET_NAME = "delay_probability"
TARGET_BINARY_NAME = "target"

# Definição do atraso
DELAY_THRESHOLD_MINUTES = 15
TARGET_QUANTILE = 0.75

# Colunas proibidas (leakage)
LEAKAGE_COLS = [
    "arr_delay",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
    "carrier_ct",
    "weather_ct",
    "nas_ct",
    "security_ct",
    "late_aircraft_ct",
]

# Split temporal
TRAIN_END_YEAR = 2018
