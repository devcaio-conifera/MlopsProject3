from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

census_query_higer_than_50k = {
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
  "native-country": "United-States"
}

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}

def test_api_post_root():
     r = client.post("/items/",json=census_query_higer_than_50k)
     assert r.json() == 1

def test_api_status_get_root():
    r = client.post("/")
    assert r.status_code == 200

def test_api_locally_post_root():
    r = client.post("/result")
    assert r.status_code != 200