import pandas as pd

from src import config
from src.data.load import load_raw_data
from src.data.clean import (
    drop_invalid_rows,
    drop_leakage_columns,
    drop_missing,
)
from src.features.target import build_delay_probability, binarize_target
from src.features.engineering import add_month_cyclical_features
from src.features.selection import get_advanced_features
from src.models.preprocessing import make_preprocessor
from src.models.train import train_pipeline

from sklearn.ensemble import RandomForestClassifier


def run_offline_inference() -> pd.DataFrame:
    """
    Executa inferência offline usando o modelo final.
    Retorna um DataFrame com probabilidades de atraso.
    """

    # 1️⃣ Carregar dados
    df = load_raw_data()

    # 2️⃣ Aplicar exatamente o mesmo pipeline de preparação
    df = drop_invalid_rows(df)
    df = build_delay_probability(df)
    df = binarize_target(df, config.TARGET_QUANTILE)
    df = drop_leakage_columns(df, config.LEAKAGE_COLS)
    df = add_month_cyclical_features(df)
    df = drop_missing(df)

    # 3️⃣ Seleção de features
    features = get_advanced_features()
    X = df[features]

    # 4️⃣ Reconstruir modelo final
    categorical_cols = ["airport", "carrier"]
    numeric_cols = ["year", "month_sin", "month_cos", "arr_flights"]

    preprocessor = make_preprocessor(categorical_cols, numeric_cols)

    model = RandomForestClassifier(
        random_state=config.RANDOM_STATE,
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=50,
        class_weight="balanced",
        n_jobs=1,
    )

    pipeline = train_pipeline(preprocessor, model, X,
                              df[config.TARGET_BINARY_NAME])

    # 5️⃣ Inferência
    df["delay_probability_pred"] = pipeline.predict_proba(X)[:, 1]

    # 6️⃣ Selecionar saída final
    output_cols = [
        "year",
        "month",
        "airport",
        "carrier",
        "delay_probability_pred",
    ]

    return df[output_cols]
