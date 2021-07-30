# Download the data
# preprocess the data
# store it to feature store

from typing import Dict, List
from pandas.core.frame import DataFrame
import utils
import config
from config import logger
import pandas as pd
from feast import FeatureStore
from datetime import datetime
from pathlib import Path

feature_refs=[
            "credit_card_transactions:V1",
            "credit_card_transactions:V2",
            "credit_card_transactions:V3",
            "credit_card_transactions:V4",
            "credit_card_transactions:V5",
            "credit_card_transactions:Time",
            "credit_card_transactions:Amount",
        ],

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
        logger.info("Dataset downloaded!")
    except Exception as ex:
        logger.error("Error in downloading file", ex)


def get_historic_features(feature_entity: DataFrame) -> DataFrame:
    store = FeatureStore(repo_path=Path(config.BASE_DIR, "features"))
    training_df = store.get_historical_features(
        entity_df=feature_entity,
        feature_refs = feature_refs
    ).to_df()
    return training_df


def get_online_features(feature_rows: List[Dict[str, any]]) -> Dict:
    store = FeatureStore(repo_path=Path(config.BASE_DIR, "features"))
    feature_vector = store.get_online_features(
        feature_refs= feature_refs,
        #entity_rows=[{"customer_id": 3}],
        entity_rows=feature_rows,
    ).to_dict()
    return feature_vector
