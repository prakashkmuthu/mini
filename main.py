from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/model.pkl")


class StudentData(BaseModel):
    study_hours: float
    attendance: float
    previous_score: float
    assignment_score: float
    sleep_hours: float
    internet_usage: float


@app.get("/")
def home():
    return {"message": "Student Pass/Fail Prediction API"}


@app.post("/predict")
def predict(data: StudentData):
    input_data = pd.DataFrame([{
        "study_hours": data.study_hours,
        "attendance": data.attendance,
        "previous_score": data.previous_score,
        "assignment_score": data.assignment_score,
        "sleep_hours": data.sleep_hours,
        "internet_usage": data.internet_usage
    }])

    prediction = model.predict(input_data)[0]

    result = "Pass" if prediction == 1 else "Fail"

    return {
        "prediction": int(prediction),
        "result": result
    }