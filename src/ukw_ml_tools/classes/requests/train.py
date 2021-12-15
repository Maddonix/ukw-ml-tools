from pydantic import BaseModel, validator
from ..config import Configuration, AiLabelConfig
from ..train_data import TrainDataDb, TrainData
from typing import Optional
from pathlib import Path


class TrainRequest(BaseModel):
    name: Path
    base_path_models: Path
    train_data_db: TrainDataDb
    train_data: Optional[TrainData]

    @validator("train_data", always=True)
    def set_train_data(cls, v, values):
        v = values["train_data_db"]
        v = TrainData(**v.to_dict_intern())
        return v
