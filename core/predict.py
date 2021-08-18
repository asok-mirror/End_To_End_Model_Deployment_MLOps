import numpy as np
from core import utils, config
from typing import Dict
from pathlib import Path
import mlflow
from app import schemas

# JSONType = Union[
#     Dict[str, Any],
#     List[dict, Any],
# ]

model = utils.load_model(config.SERVING_MODEL_DIR, config.MODEL_NAME)


def predict(
    data: np.ndarray,
) -> str:
    """Gets the Prediction

    Args:
        data (np.ndarray]): features

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
    return predict(features)
    # return {"response": response}
