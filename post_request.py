import json
import logging
import requests


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

url = "https://mlopsproject3final.herokuapp.com/items/"

headers = {'content-type': 'application/json'}

request_d = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 209642,
    "education": "HS-grad",
    "education-num": 9,
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

if __name__ == '__main__':
    r = requests.post(url, data=json.dumps(request_d), headers=headers)
    logging.info(f"Response status code : {r.status_code}")
    logging.info("Predicted salary  : %s", r.json())
    logging.info(f"Response body : {r.json()}")