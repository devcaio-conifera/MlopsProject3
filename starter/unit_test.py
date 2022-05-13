import pandas as pd
import pytest
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics
import os
import numpy as np
from sklearn.model_selection import train_test_split

@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""
    df_header =["age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "salary"]
    # os.chdir("../../")
    path = os.getcwd()
    data_path = os.path.join(path, 'data/adult.csv')
    df = pd.read_csv(data_path, header=None, names= df_header, index_col=False)
    df = df.head(200)
    return df
    


# def test_data_shape(data):
#     """ If your data is assumed to have no null values then this is a valid test. """
#     assert data.shape == data.dropna().shape, "Dropping null changes shape."


# def test_slice_averages(data):
#     """ Test to see if our mean per categorical slice is in the range 1.5 to 2.5."""
#     for cat_feat in data["categorical_feat"].unique():
#         avg_value = data[data["categorical_feat"] == cat_feat]["numeric_feat"].mean()
#         assert (
#             2.5 > avg_value > 1.5
#         ), f"For {cat_feat}, average of {avg_value} not between 2.5 and 3.5."


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
    X, y, encoder, lb= process_data(data, categorical_features=cat_features, 
    label='salary', training=True)
    expected_X_result = 74
    num_rows, num_cols = X.shape
    assert num_cols == expected_X_result

def test_process_data_rows(data):
    X, y, encoder, lb= process_data(data, categorical_features=cat_features, 
    label='salary', training=True)
    expected_X_result = 200
    num_rows, num_cols = X.shape
    assert num_rows == expected_X_result

def test_data_columns(data):
    expected_X_rows_result = 15
    assert len(data.columns) == expected_X_rows_result

def test_data_rows(data):
    expected_X_rows_result = 200
    assert len(data) == expected_X_rows_result

def test_metrics(data):
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb= process_data(train, categorical_features=cat_features, 
    label='salary', training=True)
    X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label='salary', training=False, 
    encoder= encoder, lb= lb)
    grid_model= train_model(X_train,y_train)
    preds = inference(grid_model,X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert 0 >= preds.all() <= 1
