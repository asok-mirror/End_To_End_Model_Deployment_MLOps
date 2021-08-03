import config
import joblib
import os
import numpy as np
from typing import Dict, Union, List, Any

JSONType = Union[
    Dict[str, Any],
    List[dict, Any],
]

def predict(data : Union[List[List[int]], np.ndarray]) -> str:
    """Gets the Prediction 

    Returns:
        [type]: predicted value
    """
    model_dir = os.path.join(config.MODEL_DIR, 'model.pkl')
    model = joblib.load(model_dir)
    prediction = model.predict(data).tolist()[0]
    return prediction

def form_response(request : Dict) -> str:
    """Gets the Prediction from form submit

    Returns:
        [type]: predicted value
    """
    data = request.values()
    data = [list(map(int, data))]
    return predict(data)

def api_response(request : JSONType) -> str:
    """Gets the Prediction from api call

    Returns:
        [type]: predicted value
    """
    data = np.array([list(request.values())])
    response = predict(data)
    return { "response" : response }
