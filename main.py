from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

class PredictionInput(BaseModel):
    X_fahrenheit: float

def predict_regression_model(uri, data):
  model = mlflow.sklearn.load_model(uri)
  return model.predict(data)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.post("/predict")
async def predict(data: PredictionInput):
    # Convert input to DataFrame
    features = pd.DataFrame([{
        "X_fahrenheit": data.X_fahrenheit }])
    url="model"
    prediction = predict_regression_model(url, features)


    return {
        "prediction": prediction.tolist()
    }
# url = 'keras_artifacts/59a3729a0eed411a95fa84a9b3150361/artifacts/model'
