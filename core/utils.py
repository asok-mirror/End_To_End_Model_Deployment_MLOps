# Utilities file

import urllib.request
from pathlib import Path
import os
import config
import joblib
from typing import Any, Dict
import json


def download_and_save_data_from_url(url: str, path: str, file_name: str) -> None:
    """Load csv data from the url.

    Args:
        url (str): URL of the data source.
    """

    urllib.request.urlretrieve(url, path + file_name)


def delete_file(filePath: str) -> None:
    """Delete the file from the path

    Args:
        filePath (str): location of the file
    """

    # if dataset already present delete it
    file = Path(filePath)
    file.unlink(missing_ok=True)


def get_data_source_path() -> str:
    """Gets the datasource path

    Returns:
        str: data source path
    """
    return os.path.join(os.path.dirname(os.getcwd()), os.path.join(Path(config.DATA_DIR), config.FILE_NAME))


def load_model(model_path, model_name) -> Any:
    """Loads the model

    Args:
        model_path ([type]): model file location
        model_name ([type]): model name

    Returns:
        Any: model object
    """
    return joblib.load(os.path.join(model_path, model_name))

def save_dict(d: Dict, filepath: str, cls=None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specific location.

    Warning:
        This will overwrite any existing file at `filepath`.

    Args:
        d (Dict): dictionary to save.
        filepath (str): location to save the dictionary to as a JSON file.
        cls (optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): sort keys in dict alphabetically. Defaults to False.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)

def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): JSON's filepath.

    Returns:
        A dictionary with the data loaded.
    """
    with open(filepath) as fp:
        d = json.load(fp)
    return d
