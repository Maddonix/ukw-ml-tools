from pymongo.collection import Collection
import pandas as pd
from typing import List

from .crud import get_count_query


def get_count(
    feature_name: str, pre_conditions: List, db_collection: Collection
) -> pd.DataFrame:
    agg = pre_conditions + [get_count_query(feature_name)]
    res = db_collection.aggregate(agg)
    res = [_ for _ in res]
    assert len(res) == 1
    res = res[0]["data"]
    df = {feature_name: [], "count": []}

    print(res)

    for _ in res:
        df[feature_name].append(_[feature_name])
        df["count"].append(_["count"])

    df = pd.DataFrame().from_dict(df)
    df.set_index(feature_name, inplace=True)

    return df
