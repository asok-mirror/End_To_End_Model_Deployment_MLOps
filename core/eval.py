from pandas.core.frame import DataFrame
from sklearn.metrics import classification_report, roc_auc_score 
from sklearn.pipeline import Pipeline
from config import logging
from typing import Any

def evaluation(pipeline : Pipeline, X_test : DataFrame, y_test : DataFrame) -> Any: 
    # Calculate Area Under the Receiver Operating Characteristic Curve 
    y_predicted = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, probs[:, 1])
    estimator_report = classification_report(y_test, y_predicted)
    logging.info('roc score : ', roc_auc)
    logging.info('classification report : ', estimator_report)   
    return roc_auc, estimator_report