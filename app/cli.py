import typer
import warnings
import data, train_and_optimize, predict
from config import logger
from typing import Optional

# Ignore warning
warnings.filterwarnings("ignore")

# Initialize typer app
app = typer.Typer()

@app.command()
def get_data():
    """Download and Persist the data
    """
    data.download_data()
    logger.info("Cli: Dataset downloaded!")

@app.command()
def train_model():
    """train and presist the model
    """
    train_and_optimize.optimize_model()
    logger.info("Cli: Model Trained!")

@app.command()
def predict(data : Optional[list] = [[34681.0,	-0.231080,	-3.201858,	-0.674366,	-0.976167,	-1.302294,	810.00]]):
    """Predicts the transactions

    Args:
        data (Optional[list], optional): Input Data to predict for in order of features: 
                                        Time : float
                                        V1   : float
                                        V2	 : float
                                        V3	 : float
                                        V4	 : float
                                        V5	 : float
                                        Amount                                   
                                        [description]. Defaults to [[34681.0,	-0.231080,	-3.201858,	-0.674366,	-0.976167,	-1.302294,	810.00]].

    Returns:
        [type]: [description] Predited value 0 if not a fradulant transaction or 1 if its a fradulant transaction
    """
    return predict.predict(data)