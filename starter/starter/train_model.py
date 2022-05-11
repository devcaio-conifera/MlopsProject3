# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
import pickle
import logging
import os
import sys
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
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

logger.info(f"Reading dataset...")
os.chdir("../")
path = os.getcwd()
print(path)
data_path = os.path.join(path, 'data/adult.csv')
df = pd.read_csv(data_path, header=None, names= df_header, index_col=False)
# df = df.head(200)

#train-test split.
logger.info("features and target split..")
logger.info("training and test split")
train, test = train_test_split(df, test_size=0.20)
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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label='salary', training=True
)

logger.info(" Preparing x an y test data")
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label='salary', training=False, 
    encoder= encoder, 
    lb= lb
)

logger.info(" Preparing training model ....")
grid_model = train_model(X_train,y_train)

logger.info(" Preparing inference...")
preds = inference(grid_model, X_test)

logger.info(" Preparing model metrics....")
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logger.info(f'Precision: {precision}')
logger.info(f'Recall: {recall}')
logger.info(f'fbeta: {fbeta}')


# Process the test data with the process_data function.

# save the model to disk
filename = 'final_model1.pkl'
path = os.getcwd()
os.chdir(os.path.join(path, 'model'))
pickle.dump(grid_model, open(filename, 'wb'))
