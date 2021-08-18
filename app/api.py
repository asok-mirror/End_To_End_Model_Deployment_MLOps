from pathlib import Path

import numpy as np
from fastapi import FastAPI, Response
#from prometheus_client import start_http_server
from core import config, utils, predict as prediction
from joblib import load
from schemas import CreditTransactions, FraudDetection
from core import  monitoring 
import uvicorn

#ROOT_DIR = Path(__file__).parent.parent.absolute()

# Define application
app = FastAPI(
    title="Credit card fradulant transaction detection system",
    description="Predict the legitimacy of the credit card transactions",
    version="0.1",
)

#start_http_server(8001)
monitoring.instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)


@app.get("/")
async def root():
    return "Credit card fraud detection system - V1.0"


@app.post("/predict", response_model=FraudDetection)
def predict(response: Response, data: CreditTransactions):
    model_prediction = prediction.api_response(data)
    return FraudDetection(transaction=model_prediction)


@app.get("/healthcheck")
async def healthcheck():
    return {"staus": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)

#locust -f load_test/locustfile.py --host http://127.0.0.1:3000
#export ENABLE_METRICS=true
#/etc$ sudo /etc/init.d/prometheus restart

