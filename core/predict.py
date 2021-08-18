import numpy as np
from core import utils, config
from typing import Dict, Union, List, Any
from pathlib import Path
import mlflow
from app import schemas

# JSONType = Union[
#     Dict[str, Any],
#     List[dict, Any],
# ]

# get model from the Mlflow artifacts
#artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri.split("file://")[-1]

#model = utils.load_model(artifact_uri, config.MODEL_NAME)

model = utils.load_model(config.SERVING_MODEL_DIR, config.MODEL_NAME)

def predict(
    data: Union[List[List[float]], np.ndarray],
    run_id: str = open(Path(config.MODEL_DIR, "run_id.txt")).read(),
) -> str:
    """Gets the Prediction

    Args:
        data (Union[List[List[float]], np.ndarray]): features
        run_id (str, optional): mlflow run id to fetch model. Defaults to open(Path(config.MODEL_DIR, "run_id.txt")).read().

    Returns:
        str: Predicted value
    """

    prediction = model.predict(data).tolist()[0]
    return prediction


def form_response(request: Dict) -> str:
    """Gets the Prediction from form submit

    Args:
        request (Dict): feature data from form submit

    Returns:
        str: predicted value
    """

    data = request.values()
    data = [list(map(float, data))]
    return predict(data)


def api_response(request) -> str:
    """Gets the Prediction from form submit

    Args:
        request (Dict): feature data from api request body

    Returns:
        str: predicted value
    """
    request_dict = request.dict()
    features = np.array([request_dict[f] for f in schemas.feature_names]).reshape(1, -1)
    return  predict(features)
    #return {"response": response}
