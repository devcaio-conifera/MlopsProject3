from fastapi.testclient import TestClient
import json
import pytest
# Import our app from main.py.
from main import app
import sys

sys.path.append('../../')

@pytest.fixture()
def data_less_than_50k():
    df = {
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

    return df

@pytest.fixture()
def data_more_than_50k():
    df = {
            "age": 22,
            "workclass": "Private",
            "fnlgt": 201490,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Own-child",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 20,
            "native-country": "United-States",
            }

    return df

      

# Instantiate the testing client with our app.
client = TestClient(app)

# headers = {'Content-type': 'application/json', 'accept': 'application/json'}

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}

def test_api_post_higher(data_less_than_50k):
    r = client.post("/items/",json=data_less_than_50k)
    assert r.json() == { "prediction": 1}
    assert r.status_code == 200
def test_api_post_less(data_more_than_50k):
    r = client.post("/items/",json=data_more_than_50k)
    assert r.json() == { "prediction": 0}
    assert r.status_code == 200

def test_api_status_get_root():
    r = client.post("/")
    assert r.status_code != 200

def test_api_locally_post_root():
    r = client.post("/result")
    assert r.status_code != 200