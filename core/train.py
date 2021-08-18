# get the data from the feast feature store
# split train and test data
# train and optimize the model

# from core.data import get_feature_entity_df
from typing import Dict, Sequence, Optional
from pandas.core.frame import DataFrame
from core import data, config, utils, eval
from core.config import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import mlflow
import os
from pathlib import Path
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import tempfile
import joblib
import pandas as pd
import matplotlib.image as mpimg

# import seaborn as sns
# sns.set()


def get_training_data() -> DataFrame:
    """Gets training data from data source

    Returns:
        DataFrame: training data
    """
    feature_entity = data.get_feature_entity_df()

    # get historic data from feature store

    training_data = data.get_historic_features(feature_entity)
    logger.info("training data is fetched")
    # print(training_data.head())
    return training_data


def get_features_and_target() -> DataFrame:
    """splits the data into train and test

    Returns:
        Pipeline: return train and test data
    """

    data = get_training_data()

    print(data.head())

    X = data.drop(["event_timestamp", "customer_id"], axis=1, inplace=False)

    y = data[["credit_card_transactions__Class"]]

    return X, y


def spit_test_train_data() -> DataFrame:
    """Creates test train split

    Returns:
        X_train: training features DataFrame
        X_test : testing features DataFrame
        y_train: training target DataFrame
        y_test : testing target DataFrame
    """

    X, y = get_features_and_target()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    return X_train, X_test, y_train, y_test


def create_model_pipeline() -> Pipeline:
    """Creates the model pipeline

    Returns:
          Pipeline: model pipeline
    """

    pipeline = Pipeline([("scalar", StandardScaler()), ("clf", RandomForestClassifier())])

    # pipeline.fit(X_train, y_train)

    return pipeline


def train_model(
    params_path: Path = Path(config.ARTIFACTS_DIR, config.BEST_MODEL_PARAM),
    experiment_name: Optional[str] = "best",
    run_name: Optional[str] = "model",
    model_dir: Optional[Path] = Path(config.MODEL_DIR),
) -> None:
    """Train a model using the best parameters.

    Args:
        pipeline (Pipeline): model pipeline
        params_path (Path, optional): Best model parameter. Defaults to Path(config.ARTIFACTS_DIR, config.BEST_MODEL_PARAM).
    """

    model_params = utils.load_dict(filepath=params_path)

    X_train, X_test, y_train, y_test = spit_test_train_data()

    # Start run
    # mlflow.set_tracking_uri(config.REMOTE_SERVER_URI)

    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id

        pipeline = create_model_pipeline()

        # Parameters
        pipeline.set_params(**model_params)

        # Fit Train data
        pipeline.fit(X_train, y_train)

        training_score = pipeline.score(X_train, y_train)

        # Evaluations

        training_score = pipeline.score(X_train, y_train)

        testing_score = pipeline.score(X_test, y_test)

        roc_auc, estimator_report, average_precision = eval.evaluation(pipeline, X_test, y_test)

        mlflow.log_params(model_params)
        mlflow.log_metric("training_score", training_score)
        mlflow.log_metric("testing_score", testing_score)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("average_precision", average_precision)

        with tempfile.TemporaryDirectory() as tempDir:
            utils.save_dict(model_params, Path(tempDir, "params.json"))
            joblib.dump(pipeline, Path(tempDir, config.MODEL_NAME))
            classification_report = pd.DataFrame.from_dict(estimator_report).transpose()
            classification_report.to_csv(Path(tempDir, r"classification_report.csv"))
            img = mpimg.imread(Path(config.ARTIFACTS_DIR, "roc_curve.png"))
            mpimg.imsave(Path(tempDir, "roc_curve.png"), img)
            img = mpimg.imread(Path(config.ARTIFACTS_DIR, "precision_recall_curve.png"))
            mpimg.imsave(Path(tempDir, "precision_recall_curve.png"), img)
            mlflow.log_artifacts(tempDir)

        open(Path(model_dir, "run_id.txt"), "w").write(run_id)
        joblib.dump(pipeline, os.path.join(config.MODEL_DIR, config.MODEL_NAME))


if __name__ == "__main__":
    train_model()
