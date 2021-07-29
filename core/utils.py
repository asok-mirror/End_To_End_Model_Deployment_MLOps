#Utilities file

import urllib.request
from pathlib import Path


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
