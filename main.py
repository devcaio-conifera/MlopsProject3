# Put the code for your API here.
from fastapi import FastAPI
import joblib
from pydantic import BaseModel, Field
import pandas as pd
from starter.starter.ml.data import process_data
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import json
import numpy as np
from fastapi.encoders import jsonable_encoder
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
    
# path = os.getcwd()
# print(sys.path)
def underscore_to_hyphen_replace(string: str) -> str:
    return string.replace('_', '-')

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

keys_conversion =  {
    "marital_status": "marital-status",
    "capital_gain": "capital-gain",
    "capital_loss": "capital-loss",
    "hours_per_week": "hours-per-week",
    "native_country": "native-country"
    }
# Instantiate the app.
app = FastAPI()

class census_inputs(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 52,
                "workclass": "Self-emp-inc",
                "fnlgt": 287927,
                "education": "HS-grad",
                "education-num": 9,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Wife",
                "race": "White",
                "sex": "Female",
                "capital-gain": 15024,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }
        alias_generator = underscore_to_hyphen_replace

    

@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post("/items/")
async def create_item(df_inputs: census_inputs):
    data = pd.DataFrame(jsonable_encoder(df_inputs), index=[0])
    encoder =joblib.load(os.path.join("starter", "model", "transform_dataset.pkl"))
    lb =joblib.load(os.path.join("starter","model", "transform_dataset_y.pkl"))
    model_grid = joblib.load(os.path.join("starter", "model", "final_model1.pkl"))
    X,_, _, _ = process_data(
    data, categorical_features=cat_features, training=False, 
    encoder= encoder, 
    lb= lb)   
    prediction = model_grid.predict(X)[0]
    pred_final = prediction.item()
    return {
        'prediction': pred_final
    }