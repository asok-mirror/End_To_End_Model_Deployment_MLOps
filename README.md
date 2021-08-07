This is a solution base on how to design a scalable ML App with end to end Machine Learning Life Cycle - MLOPS

# Feast 
Feast ML - feature Store- https://docs.feast.dev/getting-started/create-a-feature-repository

# configure python path 
https://stackoverflow.com/questions/35709497/anaconda-python-where-are-the-virtual-environments-stored

# configure jypyter to use git bash in windows 
https://stackoverflow.com/questions/52330492/how-to-change-jupyter-notebook-windows-shell-to-bash

# materialize the data between time ranges
feast materialize 2020-01-01T00:00:00 2021-07-29T00:00:00

# Styling and Formatting code
black .
flake8
isort .

# Airflow
export AIRFLOW_HOME=${PWD}/airflow
AIRFLOW_VERSION=2.1.2
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
# For example: 3.6
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-no-providers-${PYTHON_VERSION}.txt"
# For example: https://raw.githubusercontent.com/apache/airflow/constraints-no-providers-2.1.2/constraints-3.6.txt
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

airflow db init

# Launch Airflow webserver
export AIRFLOW_HOME=${PWD}/airflow
airflow webserver --port 8082

# Kill Process

--windows
netstat -ano|findstr "PID :5000"
taskkill /pid 18264 /f

--Linux
sudo lsof -i:5000
sudo kill PID


<div align="center">
<h1>Work In Progress</h1>
</div>

