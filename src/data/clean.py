import pandas as pd


def drop_leakage_columns(df: pd.DataFrame, leakage_cols: list) -> pd.DataFrame:
    return df.drop(columns=leakage_cols, errors="ignore")


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["arr_flights"] > 0]


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()
