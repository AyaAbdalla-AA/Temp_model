from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import os

app = FastAPI()

MODEL_URI = 'model'
model = None  # IMPORTANT: do NOT load at import time


class PredictionInput(BaseModel):
    X_fahrenheit: float


def get_model():
    """
    Lazy load model to avoid CI import crashes
    """
    global model
    if model is None:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    return model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictionInput):
    mdl = get_model()

    df = pd.DataFrame([{
        "X_fahrenheit": data.X_fahrenheit
    }])

    prediction = mdl.predict(df)

    return {"prediction": prediction.tolist()}
