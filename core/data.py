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
import os

feature_refs = [
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
    """Gets historic data from feature source

    Returns:
        DataFrame : historic feature data
    """
    store = FeatureStore(repo_path=Path(config.BASE_DIR, "features"))
    training_df = store.get_historical_features(
        entity_df=feature_entity,
        feature_refs=feature_refs[0]
    ).to_df()
    return training_df


def get_online_features(feature_rows: List[Dict[str, any]]) -> Dict:
    """Gets features from feast's online store

    Returns:
        Dict: online featurs from feast 
    """
    store = FeatureStore(repo_path=Path(config.BASE_DIR, "features"))
    feature_vector = store.get_online_features(
        feature_refs=feature_refs[0],
        #entity_rows=[{"customer_id": 3}],
        entity_rows=feature_rows,
    ).to_dict()
    return feature_vector


def get_feature_entity_df() -> DataFrame:
    """Gets feature entity from data source

    Returns:
        DataFrame: feature entity 
    """
    data = pd.read_csv(utils.get_data_source_path())
    customer_tran_id = data['customer_id'].to_list()
    now = datetime.now()
    timestamps = [datetime(now.year, now.month, now.day)] * len(customer_tran_id)
    entity_df = pd.DataFrame.from_dict({"customer_id": customer_tran_id, "event_timestamp": timestamps})
    # print(entity_df.head())
    logger.info("feature entity is fetched")
    return entity_df
