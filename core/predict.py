import config
import joblib
import os
import numpy as np
import utils
from typing import Dict, Union, List, Any

JSONType = Union[
    Dict[str, Any],
    List[dict, Any],
]


def predict(data: Union[List[List[float]], np.ndarray]) -> str:
    """Gets the Prediction 

    Returns:
        [type]: predicted value
    """

    model = utils.load_model(config.MODEL_DIR, config.MODEL_NAME)
    prediction = model.predict(data).tolist()[0]
    return prediction


def form_response(request: Dict) -> str:
    """Gets the Prediction from form submit

    Returns:
        [type]: predicted value
    """
    data = request.values()
    data = [list(map(float, data))]
    return predict(data)


def api_response(request: JSONType) -> str:
    """Gets the Prediction from api call

    Returns:
        [type]: predicted value
    """
    data = np.array([list(request.values())])
    response = predict(data)
    return {"response": response}
