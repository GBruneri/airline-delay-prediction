import pandas as pd

from src.paths import DATA_RAW_DIR


RAW_FILENAME = "Airline_Delay_Cause.csv"


def load_raw_data() -> pd.DataFrame:
    """
    Carrega o dataset bruto de atrasos de voos.

    Retorna
    -------
    pd.DataFrame
        DataFrame com os dados brutos.
    """
    file_path = DATA_RAW_DIR / RAW_FILENAME

    if not file_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado em {file_path}. "
            "Execute src/data/download.py antes."
        )

    df = pd.read_csv(file_path)

    return df
