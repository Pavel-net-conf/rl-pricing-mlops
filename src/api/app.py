from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(title="Credit Default API")

# загрузка модели при старте сервера
MODEL_PATH = "models/credit_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


# схема ожидаемых данных 
class ClientData(BaseModel):
    LIMIT_BAL: float = Field(..., ge=0.0)
    AGE: int = Field(..., ge=20, le=100)
    EDUCATION: int = Field(..., ge=0, le=6)
    avg_payment: float = Field(...)
    age_group: str = Field(...)


@app.get("/")
def read_root():
    return {"message": "MLOps API работает!"}


@app.post("/predict")
def predict_default(data: ClientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не найдена. Обучите ее перед запуском API.")

    try:
        #  JSON от пользователя в DataFrame
        df = pd.DataFrame([data.model_dump()])

        #  предсказание
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return {
            "default_prediction": int(pred),
            "probability": float(prob)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
