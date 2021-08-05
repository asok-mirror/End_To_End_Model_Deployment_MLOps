# get the data from the feast feature store
# split train and test data
# train and optimize the model

#from core.data import get_feature_entity_df
from typing import Sequence
from pandas.core.frame import DataFrame
import data
import config
from config import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import optuna
import numpy as np
import mlflow
import os


def get_training_data() -> DataFrame:
    """Gets training data from data source

    Returns:
        DataFrame: training data
    """
    feature_entity = data.get_feature_entity_df()

    # get historic data from feature store

    training_data = data.get_historic_features(feature_entity)
    logger.info('training data is fetched')
    # print(training_data.head())
    return training_data


def get_features_and_target() -> Sequence:
    """splits the data into train and test

    Returns:
        Pipeline: return train and test data
    """

    data = get_training_data()

    X = data.drop(['Unnamed: 0', 'customer_id', 'Class', 'created_time'], axis=1, inplace=False)

    y = data[['Class']]

    return X, y


def create_model_pipeline() -> Pipeline:
    """Creates the model pipeline

    Returns:
        Pipeline: model pipeline
    """
    X, y = get_features_and_target()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    pipeline = Pipeline([
        ('scalar', StandardScaler()),
        ('clf', RandomForestClassifier())])

    pipeline.fit(X_train, y_train)

    return pipeline


def objective(trial):
    """Optuna Objective function

    Args:
        trial ([type]): No of trials on model pramas
    Raises:
        optuna.TrialPruned: Early stopping of the optimization trial if poor performance.

    Returns:
        [type]: mean cv score
    """
    clf__n_estimators = trial.suggest_categorical('clf__n_estimators', [int(x) for x in np.linspace(start=200, stop=2000, num=10)])
    clf__min_samples_split = trial.suggest_int('clf__min_samples_split', 2, 5, 10)
    clf__min_samples_leaf = trial.suggest_int('clf__min_samples_leaf', 1, 2, 4)
    clf__bootstrap = trial.suggest_categorical('clf__bootstrap', [False, True])
    clf__max_features = trial.suggest_categorical('clf__max_features', ['auto', 'sqrt'])
    clf__max_depth = trial.suggest_categorical('clf__max_depth', [int(x) for x in np.linspace(10, 110, num=11)].append(None))

    params = {
        'clf__n_estimators': clf__n_estimators,
        'clf__min_samples_split': clf__min_samples_split,
        'clf__min_samples_leaf': clf__min_samples_leaf,
        'clf__bootstrap': clf__bootstrap,
        'clf__max_features': clf__max_features,
        'clf__max_depth': clf__max_depth,
    }

    if trial.should_prune():
        raise optuna.TrialPruned()

    pipeline = create_model_pipeline()
    pipeline.set_params(**params)

    X, y = get_features_and_target()

    return - np.mean(cross_val_score(pipeline, X, y, cv=8, n_jobs=-1))


def mlflow_callback(study, trial):
    """[summary]

    Args:
        study ([type]): optuna study representing optimisation task
        trial ([type]): optuna trail object representing process evaluation of objective function 
    """
    if study.best_trial.number == trial.number:
        joblib.dump(trial.user_attrs["best_model"], os.path.join(config.MODEL_DIR, 'model.pkl'))

    with mlflow.start_run(run_name=study.study_name):
        mlflow.log_params(trial.params)
        mlflow.log_metrics({"accuracy": trial.value})


def optimize_model():
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study()
    study.optimize(objective, timeout=3600, pruner=pruner, n_trials=5, callbacks=[mlflow_callback])


if __name__ == '__main__':
    optimize_model()
