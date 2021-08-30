This is a demonstration solution based on designing a scalable ML App adhering to end to end Machine Learning Life Cycle - MLOPS

set python path by creating .env file if you are using VS with MS Python Interpretor
and created the env and install the requirements 

#### Configure python path
```bash
WORKSPACE_FOLDER=/home/asok/toolbox/mlops/End_To_End_Model_Deployment_MLOps
PYTHONPATH=${WORKSPACE_FOLDER}
```

#### Feast Documentation

```bash
Feast ML - feature Store- https://docs.feast.dev/getting-started/create-a-feature-repository
```
Associated feature store configuration is in notebooks/Feast_Feature_Store.ipynb
  
##### Feast materialize the data between time ranges

```bash
feast materialize 2020-01-01T00:00:00 2021-07-29T00:00:00
```
#### Styling and Formatting code

Execute the below commands for auto styling and formatting 

```bash
black .

flake8

isort .
```

#### Airflow

DataOps and MLOps Pipelines are configured as DAGS,  run it using

##### Launch Airflow webserver

```bash
export AIRFLOW_HOME=${PWD}/airflow

airflow webserver --port 8082

````

#### ML Flow

model experiments are caputired in folder /Mlruns and are viewable in UI at

```bash
mlflow ui
```

#### Kill Process

##### --windows

```bash
netstat -ano|findstr "PID :5000"

taskkill /pid 18264 /f
````
  

#####  --Linux

```bash
sudo lsof -i:5000

sudo kill PID
````
  

#### Great Expectations

Run the great expectations checkpoint thought

```bash
great_expectations --v3-api checkpoint run credit_transactions
```
https://docs.greatexpectations.io/docs/deployment_patterns/how_to_run_a_checkpoint_in_airflow

  #### Serving Model
  
Serving model is in folder /serving 
  
#### Fast API

```bash
unicorn app:app --reload
```

##### Swagger UI

```bash
host:port/docs
```

#### Promethous and Grfana

You need to install the prometheous and grafana. 

1) Configure the FAST API end point in the prometheus.ini file 

2) ADD SMTP info for alerts in the graffana.ini file 

After you expose the FastAPI, the metrics are captured in prometheous and are viewable in the graffana. 

You need to configure the grafana dashbord to view the metrics as per the metrics content


#### Future Scope

Looking to include active learning into the pipeline. 
