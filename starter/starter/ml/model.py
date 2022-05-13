from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import logging
from sklearn.ensemble import GradientBoostingClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # scaler = StandardScaler()
    # scaled_X_train = scaler.fit_transform(X_train)
    # scaled_X_test = scaler.transform(X_test)
    # log_model = LogisticRegression(solver='saga',multi_class="ovr",max_iter=5000)
    # grid_model= LogisticRegression(random_state=0)
    grid_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=1, random_state=0)
    # Penalty Type
    # penalty = ['l1', 'l2']

    # Use logarithmically spaced C values (recommended in official docs)
    # C = np.logspace(0, 4, 10)
    # CV and fit model
    logger.info(" Hyperparameter Optimization with GridSearchCV ")
    # grid_model = GridSearchCV(log_model,param_grid={'C':C,'penalty':penalty})
    logger.info("model fitting.....")
    return grid_model.fit(X_train,y_train)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    logger.info("Model prediction...")
    y_pred = model.predict(X)
    return y_pred
    
