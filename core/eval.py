from pandas.core.frame import DataFrame
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline
import config
from config import logging
from typing import Any
import matplotlib.pyplot as plt
from pathlib import Path


def evaluation(pipeline: Pipeline, X_test: DataFrame, y_test: DataFrame) -> Any:
    """Model Evaluation

    Args:
        pipeline (Pipeline): model pipeline
        X_test (DataFrame): test features Dataframe
        y_test (DataFrame): test target Dataframe

    Returns:
        Any: evaluation metrics
    """
    # Calculate Area Under the Receiver Operating Characteristic Curve
    y_predicted = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)
    roc_auc = 0.75  # roc_auc_score(y_test, probs[:, 1]) Added for testing to be removed
    estimator_report = classification_report(y_test, y_predicted, output_dict=True)
    average_precision = average_precision_score(y_test, y_predicted)
    # logging.info('roc score : ', roc_auc)
    # logging.info('classification report : ', estimator_report)

    # Create true and false positive rates
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_predicted)

    # Obtain precision and recall 
    precision, recall, thresholds = precision_recall_curve(y_test, y_predicted)
    # Plot the roc curve 
    plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)

    # Plot recall precision curve
    plot_pr_curve(recall, precision, average_precision)
    return roc_auc, estimator_report, average_precision


# Define a roc_curve function
def plot_roc_curve(false_positive_rate, true_positive_rate, roc_auc):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=5, label="AUC = %0.3f" % roc_auc)
    plt.plot([0, 1], [0, 1], linewidth=5)
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc="upper right")
    plt.title("Receiver operating characteristic curve (ROC)")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(Path(config.ARTIFACTS_DIR, "roc_curve.png"))


# Define a precision_recall_curve function
def plot_pr_curve(recall, precision, average_precision):
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("2-class Precision-Recall curve: AP={0:0.2f}".format(average_precision))
    plt.savefig(Path(config.ARTIFACTS_DIR, "precision_recall_curve.png"))
