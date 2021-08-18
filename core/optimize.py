
import joblib
from numpy.lib.twodim_base import tri
import optuna
import mlflow
from pandas.core.frame import DataFrame
from core import train, utils, config, eval
import numpy as np
import os
from sklearn.model_selection import cross_val_score
from core.config import logger

def objective(
    X_train: DataFrame,
    X_test: DataFrame,
    y_train: DataFrame,
    y_test: DataFrame,
    X: DataFrame,
    y: DataFrame,
    trial: optuna.trial._trial.Trial,
) -> np.float64:
    """Optuna Objective function

    Args:
        trial ([int]): no of trails
        X_train ([DataFrame]): splited features training dataset
        X_test ([DataFrame]): splited features test dataset
        y_train ([DataFrame]): splited target training dataset
        y_train ([DataFrame]): splited target test dataset
        X ([DataFrame]): features training dataset
        y ([DataFrame]): target training dataset

    Raises:
        optuna.TrialPruned: Early stopping of the optimization trial if poor performance.

    Returns:
        [type]: mean cv score
    """

    clf__n_estimators = trial.suggest_categorical(
        "clf__n_estimators", [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    )
    clf__min_samples_split = trial.suggest_int("clf__min_samples_split", 2, 5, 10)
    clf__min_samples_leaf = trial.suggest_int("clf__min_samples_leaf", 1, 2, 4)
    clf__bootstrap = trial.suggest_categorical("clf__bootstrap", [False, True])
    clf__max_features = trial.suggest_categorical("clf__max_features", ["auto", "sqrt"])
    clf__max_depth = trial.suggest_categorical(
        "clf__max_depth", [int(x) for x in np.linspace(10, 110, num=11)]
    )

    params = {
        "clf__n_estimators": clf__n_estimators,
        "clf__min_samples_split": clf__min_samples_split,
        "clf__min_samples_leaf": clf__min_samples_leaf,
        "clf__bootstrap": clf__bootstrap,
        "clf__max_features": clf__max_features,
        "clf__max_depth": clf__max_depth,
    }

    if trial.should_prune():
        raise optuna.TrialPruned()

    pipeline = train.create_model_pipeline()
    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)
    roc_auc, estimator_report, average_precision = eval.evaluation(pipeline, X_test, y_test)
    trial.set_user_attr(key="roc_auc", value=roc_auc)
    trial.set_user_attr(key="average_precision", value=average_precision)

    return - np.mean(cross_val_score(pipeline, X, y, cv=8, n_jobs=-1))


def mlflow_callback(study, trial) -> None:
    """[summary]

    Args:
        study ([type]): optuna study representing optimisation task
        trial ([type]): optuna trail object representing process evaluation of objective function
    """
    # if study.best_trial.number == trial.number:
    #     joblib.dump(trial.user_attrs["best_model"], os.path.join(config.MODEL_DIR, 'model.pkl'))

    with mlflow.start_run(run_name=study.study_name):
        mlflow.log_params(trial.params)
        mlflow.log_metrics({"accuracy": trial.value})
        mlflow.log_metrics({"roc_auc": trial.user_attrs["roc_auc"]})
        mlflow.log_metrics({"average_precision": trial.user_attrs["average_precision"]})


def optimize_model() -> float:
    """Optuna Hyperparameter search

    Returns:
        float: best trial accuracy
    """
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study()
    X_train, X_test, y_train, y_test = train.spit_test_train_data()
    X, y = train.get_features_and_target()
    study.optimize(
        lambda trial: objective(X_train, X_test, y_train, y_test, X, y, trial),
        timeout=3600,
        n_trials=config.OPTUNA_TRIALS_COUNT,
        callbacks=[mlflow_callback],
    )
    # All trials
    trials_df = study.trials_dataframe()
    print(trials_df)
    trials_df = trials_df.sort_values(["value"], ascending=False)
    print(trials_df)

    # Best trial
    logger.info(f"Best value (mean cv score): {study.best_trial.value}")
    logger.info(f"Best model params: {study.best_trial.params}")
    utils.save_dict(study.best_trial.params, os.path.join(config.ARTIFACTS_DIR, config.BEST_MODEL_PARAM))   
    return  study.best_trial.value

if __name__ == "__main__":
    optimize_model()
