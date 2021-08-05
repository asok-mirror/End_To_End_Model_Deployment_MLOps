# Feature definition

import sys
from datetime import datetime
from pathlib import Path

from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource
from google.protobuf.duration_pb2 import Duration

#sys.path.insert(0, "C:\\toolbox\\ML OPS\\End_To_End_Model_Deployment_MLOps\\core")  # TBU
#print(sys.path)  # TBU
import config

# Read data
START_TIME = "2021-07-28"
feature_source = FileSource(
    path=str(Path(config.DATA_DIR, "features.parquet")), event_timestamp_column="created_time"
)

# Define an entity
feature_entity = Entity(name="customer_id", value_type=ValueType.INT64, description="customer id")

# Define a Feature View
# Can be used for fetching historical data and online serving
feature_details_view = FeatureView(
    name="credit_card_transactions",
    entities=["customer_id"],
    ttl=Duration(
        seconds=(datetime.today() - datetime.strptime(START_TIME, "%Y-%m-%d")).days * 24 * 60 * 60
    ),
    features=[
        Feature(name="V1", dtype=ValueType.FLOAT),
        Feature(name="V2", dtype=ValueType.FLOAT),
        Feature(name="V3", dtype=ValueType.FLOAT),
        Feature(name="V4", dtype=ValueType.FLOAT),
        Feature(name="V5", dtype=ValueType.FLOAT),
        Feature(name="Amount", dtype=ValueType.FLOAT),
        Feature(name="Time", dtype=ValueType.FLOAT),
        Feature(name="Class", dtype=ValueType.FLOAT),
    ],
    online=True,
    input=feature_source,
    tags={"transaction": "fradulant_transaction"},
)
