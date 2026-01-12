from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def make_preprocessor(categorical_cols, numeric_cols):
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            ("num", "passthrough", numeric_cols),
        ]
    )
