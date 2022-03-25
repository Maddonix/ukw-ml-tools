from pydantic import BaseModel
from typing import List, Union
from datetime import datetime
import pandas as pd

class TrainDataStats(BaseModel):
    name: str
    date: datetime
    is_val: List[Union[bool, str]]
    origins: List[str]
    labels: List[str]
    count: List[int]
    unique: List[int]

    def stats_df(self, exclude = {"name", "date"}):
        _dict = self.dict(exclude= exclude)
        df = pd.DataFrame.from_dict(_dict, orient = "columns")
        return df