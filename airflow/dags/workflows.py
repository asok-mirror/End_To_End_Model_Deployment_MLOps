

from airflow.decorators import dag
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator, task
from airflow.utils.dates import days_ago
from sqlalchemy.sql.elements import Extract
import os
# sys.path.insert(0, "/c/toolbox/ML OPS/End_To_End_Model_Deployment_MLOps/app")  # TBU
# sys.path.insert(0, "/c/toolbox/ML OPS/End_To_End_Model_Deployment_MLOps/core")  # TBU
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
print(os.path.abspath(os.path.dirname(__file__)))
print("asok kalidass *******************************")  # TBU
print(sys.path)  # TBU
from app import cli
from core import config

# Default DAG args
default_args = {
    "owner": "airflow",
    "catch_up": False,
}

# DataOps Ppeline


@dag(
    dag_id="dataops",
    description="Data related operations.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["dataops"]
)
def dataops():
    """Workflows related to Data Preparation
    """

    # Download Dataset
    download_dataset = PythonOperator(
        task_id="download_dataset",
        python_callable=cli.get_data
    )

    # Feature Store
    END_TS = ""

    materialize_feast_online_store = BashOperator(
        task_id="materialize_feast_online_store",
        bash_command=f"cd {config.BASE_DIR}/features && feast materialize-incremental {END_TS}"
    )

    # Task relationships
    download_dataset >> materialize_feast_online_store


def _evaluate_model():
    return "improved"

# MLOps Ppeline


@dag(
    dag_id="mlops",
    description="ML modelling related operations.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["mlops"]
)
def mlops():
    """Model Training, Optimization and Evaluation

    """

    # Training
    train_model = BashOperator(
        task_id="train_model",
        bash_command="core train-model",
    )

    # Evaluate
    evaluate_model = BranchPythonOperator(  # BranchPythonOperator returns a task_id or [task_ids]
        task_id="evaluate_model",
        python_callable=_evaluate_model,
    )

    # Improved or regressed
    improved = BashOperator(
        task_id="improved",
        bash_command="echo IMPROVED",
    )
    regressed = BashOperator(
        task_id="regressed",
        bash_command="echo REGRESSED",
    )

    # Deploy
    deploy_model = BashOperator(
        task_id="deploy_model",
        bash_command="echo 1",  # push to GitHub to kick off deployment workflows
    )

    # Reset references for monitoring
    set_monitoring_references = BashOperator(
        task_id="set_monitoring_references",
        bash_command="echo 1",  # tagifai set-monitoring-references
    )

    # Notifications (use appropriate operators, ex. EmailOperator)
    notify_teams = BashOperator(task_id="notify_teams", bash_command="echo 1")
    file_report = BashOperator(task_id="file_report", bash_command="echo 1")

    # Task relationships
    train_model >> evaluate_model >> [improved, regressed]
    improved >> [set_monitoring_references, deploy_model, notify_teams]
    regressed >> [notify_teams, file_report]


# Define DAGs
dataops_dag = dataops()
mlops_dag = mlops()
