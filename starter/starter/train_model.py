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
from .ml.data import process_data


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
">50K"]

logger.info(f"Reading dataset...")
os.chdir("../../")
path = os.getcwd()
print(path)
data = os.path.join(path, 'adult.csv')
df = pd.read_csv(data, header=None, names= df_header, index_col=False)
df = df.head(200)
# my_object_df =df.select_dtypes(include= 'object')
# my_numeric_df =df.select_dtypes(exclude= 'object')
# df_dummies = pd.get_dummies(my_object_df, drop_first = True)
# logger.info( "Concatanating dummie features and numeric" )
# final_df = pd.concat([my_numeric_df , df_dummies], axis=1)



#train-test split.
logger.info("features and target split..")
X = final_df.drop(">50K_>50K",axis=1)
y = final_df[">50K_>50K"]
logger.info("training and test split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

logger.info(" Preparing Standard scaling pre processing")
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

log_model = LogisticRegression(solver='saga',multi_class="ovr",max_iter=5000)
# Penalty Type
penalty = ['l1', 'l2']

# Use logarithmically spaced C values (recommended in official docs)
C = np.logspace(0, 4, 10)
# CV and fit model
logger.info(" Hyperparameter Optimization with GridSearchCV ")
grid_model = GridSearchCV(log_model,param_grid={'C':C,'penalty':penalty})
logger.info("model fitting.....")
grid_model.fit(scaled_X_train,y_train)

logger.info("Model prediction...")
y_pred = grid_model.predict(scaled_X_test)
accuracy_score_final = accuracy_score(y_test,y_pred)
logger.info(f'Accuracy score: {accuracy_score_final}')
cm = confusion_matrix(y_test,y_pred)
logger.info(f'Confusion matrix: {cm}')

plot_confusion_matrix(grid_model,scaled_X_test,y_test)
# train, test = train_test_split(data, test_size=0.20)

# cat_features = [
#     "workclass",
#     "education",
#     "marital-status",
#     "occupation",
#     "relationship",
#     "race",
#     "sex",
#     "native-country",
# ]
# X_train, y_train, encoder, lb = process_data(
#     train, categorical_features=cat_features, label="salary", training=True
# )

# Proces the test data with the process_data function.

# save the model to disk
filename = 'final_model.pkl'
path = os.getcwd()
os.chdir(os.path.join(path, 'starter/model'))
pickle.dump(grid_model, open(filename, 'wb'))
