from src.features.engineering import add_month_cyclical_features
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import sys
from pathlib import Path
import joblib

# ============================
# Setup de path do projeto
# ============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================
# Imports do projeto
# ============================


# ============================
# Inicialização da API
# ============================

app = FastAPI(title="Airline Delay Risk API")

# ============================
# Carregamento do modelo (UMA VEZ)
# ============================

MODEL_PATH = PROJECT_ROOT / "artifacts" / "delay_model.joblib"

pipeline = joblib.load(MODEL_PATH)

# ============================
# Schema de entrada
# ============================


class PredictionRequest(BaseModel):
    month: int
    airport: str
    carrier: str
    arr_flights: float

# ============================
# Endpoint de inferência
# ============================


@app.post("/predict")
def predict(request: PredictionRequest):
    # Cria DataFrame com uma linha
    input_df = pd.DataFrame([request.dict()])

    # Feature engineering
    input_df = add_month_cyclical_features(input_df)

    # Seleciona features (modelo sazonal)
    X = input_df[
        ["month_sin", "month_cos", "airport", "carrier", "arr_flights"]
    ]

    # Inferência
    prob = pipeline.predict_proba(X)[0, 1]

    return {"delay_probability": prob}
