import json

import joblib
from fastapi import FastAPI, HTTPException
from keras.src.applications.convnext import preprocess_input
from pydantic import BaseModel
from typing import List, Optional
import requests
from sklearn.preprocessing import LabelEncoder

app = FastAPI()
ml_model = joblib.load('Final.pkl')

label_encoder = LabelEncoder()

class InputFeatures(BaseModel):
    Gender: str
    ParentEduc: str
    LunchType: str
    TestPrep: str
    ParentMaritalStatus: str
    PracticeSport: str
    IsFirstChild: str
    NrSiblings: int
    WklyStudyHours: str
    MathScore: float
    ReadingScore: float
    WritingScore: float

  # "Gender": "male",
  # "ParentEduc": "bachelor's degree",
  # "LunchType": "standard",
  # "TestPrep": "none",
  # "ParentMaritalStatus": "married",
  # "PracticeSport": "regularly",
  # "IsFirstChild": "yes",
  # "NrSiblings": 4,
  # "WklyStudyHours": "5 - 10",
  # "MathScore": 80,
  # "ReadingScore": 80,
  # "WritingScore": 80

@app.post("/predict/")
def predict(data: InputFeatures):
    # Preprocess input data
    processed_data = preprocess_input(dict(data))
    for x in processed_data.keys():
        if isinstance(processed_data[x], str):
           processed_data[x] = label_encoder.fit_transform([processed_data[x]])[0]

    # Prepare input for model prediction
    model_input = [processed_data['Gender'], processed_data['ParentEduc'], processed_data['LunchType'],
                   processed_data['TestPrep'], processed_data['ParentMaritalStatus'], processed_data['PracticeSport'],
                   processed_data['IsFirstChild'], processed_data['NrSiblings'], processed_data['WklyStudyHours'],
                   processed_data['MathScore'], processed_data['ReadingScore'], processed_data['WritingScore']]

    # Make prediction
    prediction = ml_model.predict([model_input])[0]

    return {"prediction": prediction}