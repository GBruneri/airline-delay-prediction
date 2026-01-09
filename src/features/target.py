import pandas as pd


def build_delay_probability(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delay_probability"] = df["arr_del15"] / df["arr_flights"]
    return df


def binarize_target(
    df: pd.DataFrame,
    quantile: float,
    target_col: str = "delay_probability",
    binary_col: str = "target",
) -> pd.DataFrame:
    threshold = df[target_col].quantile(quantile)
    df = df.copy()
    df[binary_col] = (df[target_col] >= threshold).astype(int)
    return df
