from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

logged_model = 'keras_artifacts/59a3729a0eed411a95fa84a9b3150361/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)


class PredictionInput(BaseModel):
    X_fahrenheit: float
    # feature2: float
    # feature3: float

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.post("/predict")
async def predict(data: PredictionInput):
    # Convert input to DataFrame
    features = pd.DataFrame([{
        "X_fahrenheit": data.X_fahrenheit }])
    prediction = loaded_model.predict(pd.DataFrame(features))


    return {
        "prediction": prediction.tolist()
    }
