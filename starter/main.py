# Put the code for your API here.
from fastapi import FastAPI
import joblib
from pydantic import BaseModel, Field
import pandas as pd
from starter.ml.data import process_data
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import json
import numpy as np

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
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
    

@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post("/items/")
async def create_item(df_inputs: census_inputs):
    data = df_inputs.dict()
    data_as_list = pd.DataFrame([data])
    data_as_list.rename(columns = keys_conversion, inplace = True)
    data_transform = joblib.load("model/transform_dataset.pkl")
    model_grid = joblib.load("model/final_model1.pkl")
    data_as_list_categorical = data_as_list[cat_features].values
    data_as_list_continuous = data_as_list.drop(*[cat_features], axis=1)
    dataset_transformed = data_transform.transform(data_as_list_categorical)
    df_final = np.concatenate([data_as_list_continuous, dataset_transformed], axis=1)
    prediction = model_grid.predict(df_final)
    pred_final = prediction.item()
    return {
        'prediction': json.dumps(pred_final)
    }