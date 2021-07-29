# Download the data
# preprocess the data
# store it to feature store

import utils

import config
from config import logger


def download_data():
    """Download the data from URL and save to local drive

    Raises:
        Exception: Error during downloading file from source.
    """

    # if dataset already present then delete it
    utils.delete_file(config.DATA_DIR + config.FILE_NAME)

    try:
        # download and save dataset
        utils.download_and_save_data_from_url(
            url=config.FILE_SOURCE, path=config.DATA_DIR, file_name=config.FILE_NAME
        )
        logger.info("Dataset Downloaded!")
    except Exception as ex:
        logger.error("error in downloading file", ex)
