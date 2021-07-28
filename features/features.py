# Feature definition

from datetime import datetime
from pathlib import Path

from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource
from google.protobuf.duration_pb2 import Duration

import config

# Read data
START_TIME = "2021-07-23"
feature_source = FileSource(
    path=str(Path(config.DATA_DIR, "features.parquet")),
    event_timestamp_column="CREATED_TIME"
)

# Define an entity 
feature_entity = Entity(
    name="ID",
    value_type=ValueType.INT64,
    description="house id"
)

# Define a Feature View 
# Can be used for fetching historical data and online serving
feature_details_view = FeatureView(
    name="house_price_details",
    entities=["ID"],
     ttl=Duration(
        seconds=(datetime.today() - datetime.strptime(START_TIME, "%Y-%m-%d")).days * 24 * 60 * 60
    ),
    features=[
        Feature(name="CRIM", dtype=ValueType.FLOAT),
        Feature(name="ZN", dtype=ValueType.FLOAT),
        Feature(name="INDUS", dtype=ValueType.FLOAT),
        Feature(name="CHAS", dtype=ValueType.FLOAT),
        Feature(name="NOX", dtype=ValueType.FLOAT), 
        Feature(name="RM", dtype=ValueType.FLOAT),
        Feature(name="AGE", dtype=ValueType.FLOAT),
        Feature(name="DIS", dtype=ValueType.FLOAT),
        Feature(name="RAD", dtype=ValueType.FLOAT),
        Feature(name="TAX", dtype=ValueType.FLOAT),
        Feature(name="PTRATIO", dtype=ValueType.FLOAT),
        Feature(name="B", dtype=ValueType.FLOAT),
        Feature(name="LSTAT", dtype=ValueType.FLOAT),
    ],
    online=True,
    input=feature_source,
    tags={}
    )
