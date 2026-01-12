import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load import load_raw_data
from src import config
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.models.train import train_pipeline
from src.models.preprocessing import make_preprocessor
from src.features.selection import get_advanced_features
from src.features.engineering import add_month_cyclical_features
from src.features.target import build_delay_probability, binarize_target
from src.data.clean import (
    drop_invalid_rows,
    drop_leakage_columns,
    drop_missing,
)


# ============================
# Setup de path do projeto
# ============================



# ============================
# Imports do projeto
# ============================


# ============================
# Libs externas
# ============================


# ============================
# Caminhos de saída
# ============================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "delay_model.joblib"


def main():
    print("🔹 Carregando dados...")
    df = load_raw_data()

    print("🔹 Preparando dados...")
    df = drop_invalid_rows(df)
    df = build_delay_probability(df)
    df = binarize_target(df, config.TARGET_QUANTILE)
    df = drop_leakage_columns(df, config.LEAKAGE_COLS)
    df = add_month_cyclical_features(df)
    df = drop_missing(df)

    print("🔹 Selecionando features (modelo sazonal)...")
    features = get_advanced_features()
    features.remove("year")  # 🔥 modelo sazonal

    X = df[features]
    y = df[config.TARGET_BINARY_NAME]

    print("🔹 Criando pipeline...")
    categorical_cols = ["airport", "carrier"]
    numeric_cols = ["month_sin", "month_cos", "arr_flights"]

    preprocessor = make_preprocessor(categorical_cols, numeric_cols)

    model = RandomForestClassifier(
        random_state=config.RANDOM_STATE,
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=50,
        class_weight="balanced",
        n_jobs=1,
    )

    print("🔹 Treinando modelo...")
    pipeline = train_pipeline(preprocessor, model, X, y)

    print(f"🔹 Salvando modelo em: {MODEL_PATH}")
    joblib.dump(pipeline, MODEL_PATH)

    print("✅ Modelo treinado e salvo com sucesso!")


if __name__ == "__main__":
    main()
