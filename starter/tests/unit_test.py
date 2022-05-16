import pandas as pd
import pytest
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics
import os
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import sys

# os.chdir("../")
sys.path.append('../')

@pytest.fixture
def data():
    data_dict = {
                "age": 52,
                "workclass": "Self-emp-not-inc",
                "fnlwgt": 209642,
                "education": "HS-grad",
                "education_num": 9,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 45,
                "native-country": "United-States",
                "salary":"<=50K"
                }
    return pd.DataFrame([data_dict], index=[0])
    


def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."

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
def test_process_data_columns(data):
    encoder =joblib.load(os.path.join(os.getcwd(), "model", "transform_dataset.pkl"))
    lb =joblib.load(os.path.join(os.getcwd(), "model", "transform_dataset_y.pkl"))
    X,_,_,_= process_data(data, categorical_features=cat_features, training=False,
    encoder=encoder, lb=lb)
    expected_X_result = 109
    num_rows, num_cols = X.shape
    assert num_cols == expected_X_result


def test_data_columns(data):
    expected_X_rows_result = 15
    assert len(data.columns) == expected_X_rows_result


# def test_metrics(data):
#     encoder = joblib.load("model/transform_dataset.pkl")
#     lb = joblib.load("model/transform_dataset_y.pkl")
#     model_grid = joblib.load("model/final_model1.pkl")
#     X,y,_,_= process_data(data, label='salary', encoder=encoder, lb=lb,
#     categorical_features=cat_features, training=False)
#     preds = model_grid.predict(X)[0]
#     pred_final = preds.item()
#     precision, recall, fbeta = compute_model_metrics(y, pred_final)
#     assert 0 >= pred_final <= 1
